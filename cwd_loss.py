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


def build_cwd_channel_masks(maskbndict: Dict[str, torch.Tensor], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
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


def compute_cwd_loss(
    student_hooks: Dict[str, FeatureHook],
    teacher_hooks: Dict[str, FeatureHook],
    criterion: CWDLoss,
    channel_masks: Dict[str, torch.Tensor],
    layer_weights: Dict[str, float],
    temperature: float,
) -> torch.Tensor:
    """
    Compute weighted CWD loss across all hooked layers.

    Args:
        student_hooks: Dict of student feature hooks
        teacher_hooks: Dict of teacher feature hooks
        criterion: CWDLoss instance
        channel_masks: Dict from build_cwd_channel_masks (for channel alignment)
        layer_weights: Dict mapping layer name → weight (default 1.0 if not present)
        temperature: Current temperature τ

    Returns:
        Weighted average CWD loss (scalar tensor)
    """
    loss = 0.0
    total_weight = 0.0

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

        w = layer_weights.get(name, 1.0)
        loss = loss + w * criterion(s_feat, t_feat, temperature)
        total_weight += w

    if total_weight == 0.0:
        return torch.tensor(0.0, requires_grad=True)

    return loss / total_weight
