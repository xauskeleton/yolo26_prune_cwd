"""
Knowledge Distillation Losses for YOLO Detection.
=================================================
3 KD methods ngoài CWD:
1. Response KD (Hinton et al. 2015) - KL div trên channel logits
2. FitNets (Romero et al. 2015)    - MSE giữa feature maps
3. MGD (Yang et al. 2022)          - Masked Generative Distillation

Tất cả dùng chung infrastructure với CWD:
- Feature hooks (cwd_loss.py: setup_hooks, FeatureHook)
- Channel alignment (cwd_loss.py: build_kd_channel_masks)
- Warmup + ramp up (trainer.py)

Chỉ khác loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. Response KD - KL divergence on channel logits
# ============================================================================


class ResponseKDLoss(nn.Module):
    """Response-based Knowledge Distillation (Hinton et al. 2015).

    Khác CWD:
    - CWD:  softmax trên spatial per channel → KL div → capture spatial distribution
    - RKD:  softmax trên channels per spatial → KL div → capture channel distribution

    Cho mỗi spatial location (h,w), treat C channels như C-class logits, teacher soft labels = softmax(t/τ), student
    log_softmax(s/τ).
    """

    def forward(self, student_feats: torch.Tensor, teacher_feats: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Args:
            student_feats: [B, C, H, W]
            teacher_feats: [B, C, H, W] (đã align channels)
            temperature: softmax temperature τ.

        Returns:
            Scalar KD loss
        """
        B, C, H, W = student_feats.shape

        # [B, C, H, W] → [B, H*W, C]: mỗi spatial location là 1 "sample" với C "classes"
        s = student_feats.permute(0, 2, 3, 1).reshape(B * H * W, C)
        t = teacher_feats.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Softmax trên channels (dim=-1)
        s_log_soft = F.log_softmax(s / temperature, dim=-1)
        t_soft = F.softmax(t / temperature, dim=-1)

        # KL divergence
        loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean")

        # Scale by τ² (standard KD)
        return (temperature**2) * loss


# ============================================================================
# 2. FitNets - MSE feature matching
# ============================================================================


class FitNetsLoss(nn.Module):
    """FitNets: Hints for Thin Deep Nets (Romero et al. 2015).

    MSE giữa student và teacher feature maps. Normalize features trước khi tính MSE để ổn định training.

    Khi student channels ≠ teacher channels:
    - Dùng channel mask (như CWD) nếu student là pruned model
    - Hoặc dùng 1×1 conv adapter (regressor) để project dimensions
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: normalize features trước MSE (L2 norm per channel). Giúp ổn định khi magnitude features khác nhau
                nhiều.
        """
        super().__init__()
        self.normalize = normalize

    def forward(self, student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_feats: [B, C, H, W]
            teacher_feats: [B, C, H, W] (đã align channels).

        Returns:
            Scalar MSE loss
        """
        if self.normalize:
            # L2 normalize per channel: [B, C, H, W] → norm trên (H,W)
            s = F.normalize(student_feats.flatten(2), dim=2)
            t = F.normalize(teacher_feats.flatten(2), dim=2)
            return F.mse_loss(s, t)

        return F.mse_loss(student_feats, teacher_feats)


# ============================================================================
# 3. MGD - Masked Generative Distillation
# ============================================================================


class MGDGenerator(nn.Module):
    """Lightweight generator cho MGD. Project từ masked student features → teacher feature space.

    Architecture: Conv1x1 → BN → ReLU → Conv1x1
    """

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(student_channels, student_channels, 1),
            nn.BatchNorm2d(student_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(student_channels, teacher_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class MGDLoss(nn.Module):
    """Masked Generative Distillation (Yang et al. 2022).

    Ý tưởng:
    1. Random mask một số channels của student features
    2. Generator (1×1 conv) reconstruct teacher features từ masked student
    3. MSE loss giữa generated và teacher features
    → Ép student học rich representations, không chỉ copy teacher

    Cần tạo 1 MGDGenerator per layer vì channel sizes khác nhau. Generator params được train cùng student (thêm vào
    optimizer).
    """

    def __init__(self, mask_ratio: float = 0.5):
        """
        Args:
            mask_ratio: tỷ lệ channels bị mask (0.0-1.0), default 0.5.
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.generators = nn.ModuleDict()

    def add_generator(self, layer_name: str, student_channels: int, teacher_channels: int):
        """Thêm generator cho 1 layer. Gọi trong setup phase khi biết channel sizes.

        Args:
            layer_name: e.g. "model.13"
            student_channels: số channels student features
            teacher_channels: số channels teacher features
        """
        # ModuleDict không chấp nhận '.' trong key → thay bằng '_'
        key = layer_name.replace(".", "_")
        self.generators[key] = MGDGenerator(student_channels, teacher_channels)

    def forward(self, student_feats: torch.Tensor, teacher_feats: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Args:
            student_feats: [B, C_s, H, W]
            teacher_feats: [B, C_t, H, W] (full teacher, chưa align)
            layer_name: e.g. "model.13".

        Returns:
            Scalar MSE loss
        """
        key = layer_name.replace(".", "_")
        generator = self.generators[key]

        _B, C_s, _H, _W = student_feats.shape

        # Random channel mask: keep (1-mask_ratio) channels
        mask = torch.ones(1, C_s, 1, 1, device=student_feats.device)
        num_masked = int(C_s * self.mask_ratio)
        if num_masked > 0:
            mask_indices = torch.randperm(C_s)[:num_masked]
            mask[:, mask_indices, :, :] = 0.0

        # Mask student features
        masked_student = student_feats * mask

        # Generate → reconstruct teacher
        generated = generator(masked_student)

        # MSE with teacher
        return F.mse_loss(generated, teacher_feats)


# ============================================================================
# Compute functions (tương tự compute_cwd_loss)
# ============================================================================


def compute_response_kd_loss(student_hooks, teacher_hooks, criterion, channel_masks, temperature):
    """Compute Response KD loss across all hooked layers."""
    loss = 0.0
    count = 0

    for name in student_hooks:
        s_feat = student_hooks[name].features
        t_feat = teacher_hooks[name].features
        if s_feat is None or t_feat is None:
            continue

        # Channel alignment
        if name in channel_masks:
            t_feat = t_feat[:, channel_masks[name], :, :]

        # Spatial alignment
        if s_feat.shape[2:] != t_feat.shape[2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode="bilinear", align_corners=False)

        loss = loss + criterion(s_feat, t_feat, temperature)
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)
    return loss / count


def compute_fitnets_loss(student_hooks, teacher_hooks, criterion, channel_masks):
    """Compute FitNets loss across all hooked layers."""
    loss = 0.0
    count = 0

    for name in student_hooks:
        s_feat = student_hooks[name].features
        t_feat = teacher_hooks[name].features
        if s_feat is None or t_feat is None:
            continue

        # Channel alignment
        if name in channel_masks:
            t_feat = t_feat[:, channel_masks[name], :, :]

        # Spatial alignment
        if s_feat.shape[2:] != t_feat.shape[2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode="bilinear", align_corners=False)

        loss = loss + criterion(s_feat, t_feat)
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)
    return loss / count


def compute_mgd_loss(student_hooks, teacher_hooks, criterion, channel_masks):
    """Compute MGD loss across all hooked layers.

    Khác CWD/FitNets: KHÔNG align teacher channels trước. Generator xử lý dimension mismatch (student_ch → teacher_ch).
    """
    loss = 0.0
    count = 0

    for name in student_hooks:
        s_feat = student_hooks[name].features
        t_feat = teacher_hooks[name].features
        if s_feat is None or t_feat is None:
            continue

        # Spatial alignment
        if s_feat.shape[2:] != t_feat.shape[2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode="bilinear", align_corners=False)

        # MGD forward nhận teacher FULL channels (generator project)
        loss = loss + criterion(s_feat, t_feat, name)
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)
    return loss / count
