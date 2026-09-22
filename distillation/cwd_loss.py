"""
Channel-Wise Distillation (CWD) Loss for Knowledge Distillation

Paper: "Channel-wise Knowledge Distillation for Dense Prediction"
https://arxiv.org/abs/2011.13256

Implementation follows Eq 4-5 from the paper:
    1. Spatial softmax per channel: φ(y_c)_i = exp(y_i^c / τ) / Σ exp(y_j^c / τ)
    2. KL divergence: L = (τ² / C) × Σ_c KL(teacher || student)

Channel alignment: uses maskbndict from pruned checkpoint to select
matching teacher channels (no 1×1 conv adapters needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class CWDLoss(nn.Module):
    """
    Channel-wise Distillation Loss (paper Eq 4-5).

    Per-channel spatial softmax → KL divergence.
    Temperature is passed per-forward to support dynamic scheduling.
    """

    def forward(self, student_feats: torch.Tensor, teacher_feats: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Args:
            student_feats: [B, C, H, W] (đã align channels)
            teacher_feats: [B, C, H, W] (đã align channels)
            temperature: softmax temperature τ

        Returns:
            Scalar CWD loss
        """
        B, C, H, W = student_feats.shape

        # Flatten spatial: [B, C, H*W]
        s = student_feats.view(B, C, -1)
        t = teacher_feats.view(B, C, -1)

        # Spatial softmax per channel (Eq 4)
        s_log_soft = F.log_softmax(s / temperature, dim=2)  # [B, C, H*W]
        t_soft = F.softmax(t / temperature, dim=2)           # [B, C, H*W]

        # KL divergence per channel (Eq 5)
        # F.kl_div expects log-prob as input, prob as target
        loss = F.kl_div(s_log_soft, t_soft, reduction='none').sum(dim=2)  # [B, C]

        # Scale by τ² and average over batch and channels (Eq 5: τ²/C × Σ_c = τ² × mean_c)
        loss = (temperature ** 2) * loss.mean()

        return loss


class FeatureHook:
    """Passive hook to capture intermediate features from a model layer."""

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        self.features = output


def setup_hooks(model: nn.Module, layer_names: List[str]) -> Dict[str, FeatureHook]:
    """
    Register forward hooks on specified layers.

    Args:
        model: PyTorch model (unwrapped, not AutoBackend)
        layer_names: e.g. ['model.13', 'model.16', 'model.19', 'model.22']

    Returns:
        Dict mapping layer name → FeatureHook
    """
    hooks = {}
    for name, module in model.named_modules():
        if name in layer_names:
            hook = FeatureHook()
            module.register_forward_hook(hook)
            hooks[name] = hook
    return hooks


def build_kd_channel_masks(maskbndict: Dict[str, torch.Tensor], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
    """
    Build boolean masks for channel alignment between teacher and pruned student.

    C3k2 module at index i has output BN = model.{i}.cv2.bn.
    The mask indicates which teacher channels were kept in the student.

    Args:
        maskbndict: Dict from pruned checkpoint, keys like "model.13.cv2.bn",
                    values are float tensors [C_teacher] of 0.0/1.0
        layer_indices: List of layer indices to distill, e.g. [13, 16, 19, 22]

    Returns:
        Dict mapping "model.{i}" → boolean mask [C_teacher]
    """
    masks = {}
    for idx in layer_indices:
        bn_name = f"model.{idx}.cv2.bn"
        if bn_name in maskbndict:
            masks[f"model.{idx}"] = maskbndict[bn_name].bool()
    return masks


class ProjectionAligner(nn.Module):
    """1x1 conv projection layers to align teacher channels → student channels."""

    def __init__(self, channel_pairs: Dict[str, tuple]):
        """
        Args:
            channel_pairs: Dict mapping layer name → (teacher_channels, student_channels)
        """
        super().__init__()
        self.projections = nn.ModuleDict()
        for name, (t_ch, s_ch) in channel_pairs.items():
            # Replace '.' with '_' for nn.ModuleDict key compatibility
            key = name.replace('.', '_')
            self.projections[key] = nn.Conv2d(t_ch, s_ch, kernel_size=1, bias=False)
        # Init with kaiming
        for m in self.projections.values():
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def project(self, name: str, feat: torch.Tensor) -> torch.Tensor:
        key = name.replace('.', '_')
        if key in self.projections:
            return self.projections[key](feat)
        return feat


def build_projection_aligner(
    student_hooks: Dict[str, FeatureHook],
    teacher_hooks: Dict[str, FeatureHook],
    model: nn.Module,
    teacher_model: nn.Module,
    layer_names: List[str],
    device: torch.device,
) -> ProjectionAligner:
    """
    Build ProjectionAligner by running a dummy forward to get channel dims.

    Args:
        student_hooks, teacher_hooks: registered hooks
        model: student model
        teacher_model: teacher model
        layer_names: hooked layer names
        device: torch device

    Returns:
        ProjectionAligner on device
    """
    # Dummy forward to populate hooks
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        model(dummy)
        teacher_model(dummy)

    channel_pairs = {}
    for name in layer_names:
        s_feat = student_hooks[name].features if name in student_hooks else None
        t_feat = teacher_hooks[name].features if name in teacher_hooks else None
        if s_feat is not None and t_feat is not None:
            t_ch = t_feat.shape[1]
            s_ch = s_feat.shape[1]
            if t_ch != s_ch:
                channel_pairs[name] = (t_ch, s_ch)

    aligner = ProjectionAligner(channel_pairs).to(device)
    return aligner


def compute_cwd_loss(
    student_hooks: Dict[str, FeatureHook],
    teacher_hooks: Dict[str, FeatureHook],
    criterion: CWDLoss,
    channel_masks: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Compute CWD loss across all hooked layers.

    Args:
        student_hooks: Dict of student feature hooks
        teacher_hooks: Dict of teacher feature hooks
        criterion: CWDLoss instance
        channel_masks: Dict from build_kd_channel_masks (for channel alignment)
        temperature: Current temperature τ

    Returns:
        Average CWD loss (scalar tensor)
    """
    loss = 0.0
    count = 0

    for name in student_hooks:
        s_feat = student_hooks[name].features
        t_feat = teacher_hooks[name].features

        if s_feat is None or t_feat is None:
            continue

        # Align channels using mask from maskbndict
        if name in channel_masks:
            mask = channel_masks[name]
            t_feat = t_feat[:, mask, :, :]

        # Spatial size mismatch → interpolate teacher to match student
        if s_feat.shape[2:] != t_feat.shape[2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False)

        loss = loss + criterion(s_feat, t_feat, temperature)
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)

    return loss / count


def compute_cwd_loss_proj(
    student_hooks: Dict[str, FeatureHook],
    teacher_hooks: Dict[str, FeatureHook],
    criterion: CWDLoss,
    projection: ProjectionAligner,
    temperature: float,
) -> torch.Tensor:
    """
    Compute CWD loss with projection layer alignment (instead of mask).

    Teacher features are projected via 1x1 conv to match student channels.
    """
    loss = 0.0
    count = 0

    for name in student_hooks:
        s_feat = student_hooks[name].features
        t_feat = teacher_hooks[name].features

        if s_feat is None or t_feat is None:
            continue

        # Project teacher channels → student channels via 1x1 conv
        t_feat = projection.project(name, t_feat)

        # Spatial size mismatch → interpolate
        if s_feat.shape[2:] != t_feat.shape[2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False)

        loss = loss + criterion(s_feat, t_feat, temperature)
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)

    return loss / count
