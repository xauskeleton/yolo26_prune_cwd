"""
DMS (Differentiable Model Scaling) utilities for YOLOv26 pruning.

Implements Differentiable Top-k from paper:
"Differentiable Model Scaling using Differentiable Topk" (ICML 2024)

Core idea:
- Each prunable BN layer gets a learnable parameter `a` (pruning ratio)
- Soft mask = Sigmoid(N * (rank(|gamma|)/N - a))
- Resource constraint ensures target GFLOPs is met
- After search, extract `a` values as per-layer pruning ratios

Usage:
    model.train(data="coco.yaml", dms=True, dms_target=0.3, dms_lambda=1.0, dms_lr=5e-3)
"""

import re

import yaml
import torch
import torch.nn as nn

from ultralytics.utils.ops import make_divisible
from ultralytics.nn.modules.block import Bottleneck, PSABlock


# ============================================================================
# PRUNING UTILITIES (shared by prune.py)
# ============================================================================

def make_divisible_channels(channels: int, max_channels: int, divisor: int) -> int:
    """
    Làm tròn channels đến bội số gần nhất của divisor.
    Dùng make_divisible từ Ultralytics.

    Args:
        channels:     Số channels cần làm tròn
        max_channels: Giới hạn trên (không vượt quá origin)
        divisor:      Số chia (8 hoặc 16)

    Returns:
        int: Số channels đã làm tròn, đảm bảo <= max_channels
    """
    return min(make_divisible(channels, divisor), max_channels)


def get_layer_ratio(layer_name: str, layer_ratio_cfg: dict, default_ratio: float) -> float:
    """
    Lấy prune ratio cho một layer cụ thể.

    Matching theo thứ tự ưu tiên (specific → general):
    1. Exact match:   'model.0.bn' → 0.1
    2. Layer index:   'model.0'    → áp dụng cho tất cả BN trong layer 0
    3. Group name:    'backbone'   → model.0 đến model.9
                      'head'       → model.10 trở đi
                      'detect'     → layer Detect
    4. Default:       global prune_ratio

    Args:
        layer_name:      Tên BN layer, ví dụ 'model.2.cv1.bn'
        layer_ratio_cfg: Dict từ YAML, ví dụ {'model.0': 0.1, 'backbone': 0.2}
        default_ratio:   Global prune ratio nếu không match

    Returns:
        float: prune ratio cho layer này

    Example YAML (layer_ratio.yaml):
        # Giữ nhiều kênh ở layer đầu
        model.0: 0.1
        model.1: 0.1
        # Tỉa mạnh ở head
        model.15: 0.6
        model.18: 0.6
        model.21: 0.6
        # Group rules
        backbone: 0.3
        head: 0.5
        detect: 0.4
    """
    if not layer_ratio_cfg:
        return default_ratio

    # 1. Exact match
    if layer_name in layer_ratio_cfg:
        return float(layer_ratio_cfg[layer_name])

    # 2. Layer index match (model.X)
    match = re.match(r'(model\.\d+)', layer_name)
    if match:
        layer_prefix = match.group(1)
        if layer_prefix in layer_ratio_cfg:
            return float(layer_ratio_cfg[layer_prefix])

        # 3. Group match
        layer_idx = int(re.search(r'model\.(\d+)', layer_name).group(1))

        # detect: Detect head layer (thường là 22)
        if 'detect' in layer_ratio_cfg and layer_idx >= 22:
            return float(layer_ratio_cfg['detect'])
        # backbone: model.0 - model.9
        if 'backbone' in layer_ratio_cfg and layer_idx <= 9:
            return float(layer_ratio_cfg['backbone'])
        # head: model.10 - model.21
        if 'head' in layer_ratio_cfg and 10 <= layer_idx <= 21:
            return float(layer_ratio_cfg['head'])

    return default_ratio


def build_pruned_yaml(cfg, model_size, nc):
    """
    Build pruned YAML dynamically from original model config.

    Hỗ trợ tất cả model sizes (n, s, m, l, x) bằng cách:
    - Đọc cấu trúc từ original YAML
    - Apply depth_multiple cho repeat counts
    - Map module types sang Pruned versions
    - Giữ nguyên structural parameters (k, stride, shortcut, etc.)

    Args:
        cfg (str): Path to original YAML config
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        nc (int): Number of classes (từ loaded model)

    Returns:
        dict: Pruned model config ready for DetectionModelPruned
    """
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)

    # Get scale parameters: [depth_multiple, width_multiple, max_channels]
    depth, width, max_ch = model_yamls['scales'][model_size]

    pruned_yaml = {
        'nc': nc,
        'scales': model_yamls['scales'],
        'scale': model_size,
        'end2end': model_yamls.get('end2end', False),
        'reg_max': model_yamls.get('reg_max', 16),
    }

    def map_layer(f, n, m, args):
        """Map a single YAML layer to its pruned equivalent."""
        # Apply depth to repeat count (giống ultralytics parse_model)
        actual_n = max(round(n * depth), 1) if n > 1 else n

        if m == 'C3k2':
            # C3k2 args: [c2, c3k] hoặc [c2, c3k, e] hoặc [c2, c3k, e, attn]
            # attn=True (arg thứ 4) → dùng C3k2PrunedAttn
            has_attn = len(args) >= 4 and args[3] is True
            if has_attn:
                return [f, actual_n, 'C3k2PrunedAttn', [args[0], True]]
            else:
                return [f, actual_n, 'C3k2Pruned', [args[0], True]]
        elif m == 'SPPF':
            # SPPF args: [c2, k, n_pool, shortcut] → giữ nguyên
            return [f, actual_n, 'SPPFPruned', args]
        elif m == 'C2PSA':
            # C2PSA args: [c2] hoặc [c2, e] → giữ nguyên
            return [f, actual_n, 'C2PSAPruned', args]
        elif m == 'Detect':
            return [f, actual_n, 'DetectPruned', [nc]]
        else:
            # Conv, nn.Upsample, Concat → giữ nguyên
            return [f, actual_n, m, args]

    pruned_yaml['backbone'] = [map_layer(*layer) for layer in model_yamls['backbone']]
    pruned_yaml['head'] = [map_layer(*layer) for layer in model_yamls['head']]

    return pruned_yaml


def build_ignore_bn_list(model):
    """
    Build list of BN layers that should NOT be pruned.

    Rules (same as trainer.py sparsity setup):
    - Bottleneck with residual (add=True): ignore cv2.bn + parent C3k.cv1.bn
    - PSABlock: ignore all internal BNs + parent layer cv1.bn

    Args:
        model: Unwrapped YOLOv26 model

    Returns:
        list: BN layer names to ignore
    """
    ignore = []
    for k, m in model.named_modules():
        if isinstance(m, Bottleneck):
            if m.add:
                ignore.append(k + '.cv2.bn')
                parts = k.split('.')
                if len(parts) >= 2 and parts[-2] == 'm':
                    parent = k.rsplit(".", 2)[0]
                    ignore.append(parent + ".cv1.bn")
        elif isinstance(m, PSABlock):
            for sub_k, sub_m in m.named_modules():
                if isinstance(sub_m, nn.BatchNorm2d):
                    ignore.append(f"{k}.{sub_k}")
            parts = k.split('.')
            if len(parts) >= 2:
                layer_idx = parts[1]
                ignore.append(f"model.{layer_idx}.cv1.bn")
    return list(set(ignore))


def make_soft_mask_hook(bn_name, a_params):
    """
    Create forward hook that applies differentiable soft mask after BN.

    Algorithm per forward:
        1. importance = |BN.gamma|
        2. c' = rank(importance) / N  (detached, no grad)
        3. mask = Sigmoid(N * (c' - a))  (grad flows through a)
        4. output *= mask

    Args:
        bn_name: Name of the BN layer
        a_params: Dict {bn_name: nn.Parameter(a)} shared across all hooks

    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        gamma = module.weight.data.abs()
        N = gamma.shape[0]

        # Step 1-2: Importance normalization (no grad - treat c' as constant)
        with torch.no_grad():
            sorted_idx = gamma.argsort()
            rank = torch.zeros_like(gamma)
            rank[sorted_idx] = torch.arange(N, device=gamma.device, dtype=gamma.dtype)
            c_prime = rank / N  # uniform [0, 1]

        # Step 3: Soft mask (grad flows through a only)
        a = a_params[bn_name]
        mask = torch.sigmoid(N * (c_prime - a))

        # Step 4: Apply mask [1, C, 1, 1] broadcast over batch & spatial
        return output * mask.view(1, -1, 1, 1)

    return hook


def profile_per_layer_flops(model, imgsz=640, device='cuda'):
    """
    Profile FLOPs (MACs) per Conv2d layer using forward hooks.

    Args:
        model: YOLOv26 model (unwrapped)
        imgsz: Input image size (int or [h, w])
        device: Device for dummy input

    Returns:
        (dict, float): {conv_name: flops}, total_flops
    """
    flops_dict = {}
    hooks = []

    def _make_hook(name):
        def _hook(module, inp, output):
            h, w = output.shape[2:]
            flops = (module.in_channels * module.out_channels
                     * module.kernel_size[0] * module.kernel_size[1]
                     * h * w / module.groups)
            flops_dict[name] = flops
        return _hook

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_make_hook(name)))

    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
    dummy = torch.zeros(1, 3, imgsz[0], imgsz[1], device=device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(dummy)
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    total = sum(flops_dict.values())
    return flops_dict, total


def build_conv_bn_mapping(model, ignore_bn_list):
    """
    Build mapping from Conv2d layer name to its output BN info.

    In ultralytics Conv module: conv (Conv2d) + bn (BN2d) + act
    So 'model.0.conv' has output BN at 'model.0.bn'

    Args:
        model: Unwrapped model
        ignore_bn_list: BN layers that are not pruned

    Returns:
        dict: {conv_name: {'out_bn': str, 'is_depthwise': bool}}
    """
    mapping = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name.endswith('.conv'):
            out_bn = name[:-4] + 'bn'  # replace '.conv' → '.bn'
            is_dw = (m.groups == m.in_channels and m.in_channels > 1)
            mapping[name] = {
                'out_bn': out_bn,
                'is_depthwise': is_dw,
            }
    return mapping


def compute_resource_loss(a_params, conv_flops, conv_bn_map, total_flops, target_ratio):
    """
    GFLOPs-based resource constraint (differentiable w.r.t a params).

    Paper Eq. 5-6:
        loss = log(r_c / r_t) if r_c > r_t, else 0
        r_c = effective_flops / total_flops
        r_t = 1 - target_ratio

    For each conv layer:
        - regular:   effective = original × (1 - a_out)²
        - depthwise: effective = original × (1 - a_out)

    Note: Using (1-a_out)² as proxy for (1-a_in)×(1-a_out) because
    neighboring layers tend to have similar pruning ratios.

    Args:
        a_params: Dict {bn_name: nn.Parameter}
        conv_flops: Dict {conv_name: flops} from profiling
        conv_bn_map: Dict {conv_name: {'out_bn', 'is_depthwise'}}
        total_flops: Total original FLOPs
        target_ratio: Target pruning ratio (e.g., 0.3 = remove 30%)

    Returns:
        torch.Tensor: Resource constraint loss (scalar, differentiable)
    """
    device = next(iter(a_params.values())).device
    effective_flops = torch.tensor(0.0, device=device)

    for conv_name, flops in conv_flops.items():
        info = conv_bn_map.get(conv_name)
        if info is None:
            # Conv2d without BN mapping (e.g., Detect head final conv)
            effective_flops = effective_flops + flops
            continue

        out_bn = info['out_bn']
        is_dw = info['is_depthwise']

        a_out = a_params.get(out_bn, None)
        if a_out is None:
            # Ignored BN → no pruning, keep all channels
            effective_flops = effective_flops + flops
            continue

        retain = 1.0 - a_out

        if is_dw:
            ratio = retain
        else:
            ratio = retain * retain

        effective_flops = effective_flops + flops * ratio

    r_c = effective_flops / total_flops
    r_t = 1.0 - target_ratio

    if r_c > r_t:
        return torch.log(r_c / r_t)
    return torch.tensor(0.0, device=device, requires_grad=True)


def compute_l1_loss(model, ignore_bn_list):
    """
    L1 penalty on BN gamma weights (drives unimportant channels to zero).

    Args:
        model: Unwrapped model
        ignore_bn_list: BN layers to skip

    Returns:
        torch.Tensor: Σ|γ| (scalar)
    """
    device = next(model.parameters()).device
    l1 = torch.tensor(0.0, device=device)
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name not in ignore_bn_list:
            l1 = l1 + m.weight.abs().sum()
    return l1


def extract_ratios_from_checkpoint(ckpt_path, save_path='dms_ratios.yaml'):
    """
    Extract learned `a` params from DMS training checkpoint → YAML.

    Output YAML can be used with: prune.py --layer-ratio dms_ratios.yaml

    Args:
        ckpt_path: Path to checkpoint (.pt)
        save_path: Output YAML path

    Returns:
        dict: {bn_name: pruning_ratio}
    """
    import yaml

    ckpt = torch.load(ckpt_path, weights_only=False)
    a_params = ckpt.get('dms_a_params', {})
    if not a_params:
        raise ValueError(f"No dms_a_params found in {ckpt_path}!")

    ratios = {}
    for name, a in a_params.items():
        val = float(a.clamp(0.01, 0.95).item())
        ratios[name] = round(val, 4)

    with open(save_path, 'w') as f:
        yaml.dump(ratios, f, default_flow_style=False, sort_keys=True)

    # Print summary
    avg = sum(ratios.values()) / len(ratios)
    print(f"Extracted {len(ratios)} DMS ratios to {save_path}")
    print(f"  Average pruning ratio: {avg:.4f}")
    print(f"  Min: {min(ratios.values()):.4f}, Max: {max(ratios.values()):.4f}")

    return ratios
