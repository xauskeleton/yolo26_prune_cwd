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
            # Xác định c3k theo logic Ultralytics (tasks.py line 1651-1654):
            # Size m/l/x: force c3k=True | Size n/s: giữ YAML value
            c3k_val = args[1] if len(args) >= 2 else False
            if model_size in ('m', 'l', 'x'):
                c3k_val = True
            if c3k_val:
                return [f, actual_n, 'C3k2Pruned', [args[0], True]]
            else:
                return [f, actual_n, 'C3k2PrunedBn', [args[0], False]]
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


def make_soft_mask_hook(bn_name, a_params, importance='gamma', taylor_buffers=None, conv_module=None):
    """
    Create forward hook that applies differentiable soft mask after BN.

    Algorithm per forward:
        1. importance = |BN.gamma| or taylor buffer or L1 norm of conv weights
        2. c' = rank(importance) / N  (detached, no grad)
        3. mask = Sigmoid(N * (c' - a))  (grad flows through a)
        4. output *= mask
        (if taylor: register backward hook to update taylor buffer with EMA)

    Args:
        bn_name: Name of the BN layer
        a_params: Dict {bn_name: nn.Parameter(a)} shared across all hooks
        importance: 'gamma' (|BN.weight|), 'taylor' ((mask * grad)^2 with EMA),
                    or 'l1' (L1 norm of corresponding Conv filter weights)
        taylor_buffers: Dict {bn_name: Tensor} required when importance='taylor'
        conv_module: nn.Conv2d module, required when importance='l1'

    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        N = module.weight.shape[0]

        # Step 1: Get importance scores
        with torch.no_grad():
            if importance == 'l1' and conv_module is not None:
                # L1 norm per output filter: sum(|weight[i, :, :, :]|)
                scores = conv_module.weight.data.abs().sum(dim=[1, 2, 3])
            elif importance == 'taylor' and taylor_buffers is not None and bn_name in taylor_buffers:
                scores = taylor_buffers[bn_name]
                # Fallback to gamma if taylor is all zeros (first few iters)
                if scores.max() == scores.min():
                    scores = module.weight.data.abs()
            else:
                scores = module.weight.data.abs()

            # Step 2: Rank normalize → uniform [0, 1]
            sorted_idx = scores.argsort()
            rank = torch.zeros_like(scores)
            rank[sorted_idx] = torch.arange(N, device=scores.device, dtype=scores.dtype)
            c_prime = rank / N

        # Step 3: Soft mask (grad flows through a only)
        a = a_params[bn_name]
        mask = torch.sigmoid(N * (c_prime - a))  # shape [C]

        # Step 4: Taylor importance update via backward hook on mask [C]
        if importance == 'taylor' and taylor_buffers is not None and module.training and mask.requires_grad:
            mask_vals = mask.detach()  # save current mask values
            def _taylor_backward_hook(grad):
                # grad shape [C] = d(loss)/d(mask), channel-level
                with torch.no_grad():
                    taylor_new = (mask_vals * grad) ** 2
                    if not taylor_new.isnan().any() and taylor_new.max() != taylor_new.min():
                        taylor_buffers[bn_name] = (
                            taylor_buffers[bn_name] * 0.99 + taylor_new * 0.01
                        )
            mask.register_hook(_taylor_backward_hook)

        # Step 5: Apply mask
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
    Build mapping from Conv2d to output BN + input BN (for exact FLOPs).

    Uses model YAML topology to track which BN feeds into each conv's input.
    FLOPs = in_channels × out_channels × k² × H × W / groups
    After pruning: in_ch_eff = in_ch × (1-a_in), out_ch_eff = out_ch × (1-a_out)

    Args:
        model: Unwrapped model (with .yaml attribute)
        ignore_bn_list: BN layers not pruned

    Returns:
        (dict, dict):
            conv_bn_map: {conv_name: {'out_bn', 'in_bn', 'is_depthwise'}}
            bn_channels: {bn_name: num_features}
    """
    # Collect BN channel counts
    bn_channels = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_channels[name] = m.num_features

    # Parse YAML topology
    yaml_cfg = getattr(model, 'yaml', {})
    layers = yaml_cfg.get('backbone', []) + yaml_cfg.get('head', [])

    # Step 1: Output BN for each top-level layer index
    idx_to_out_bn = {}
    for i, (f, n, m_type, args) in enumerate(layers):
        base = f"model.{i}"
        if m_type == 'Conv':
            idx_to_out_bn[i] = base + '.bn'
        elif m_type in ('C3k2', 'SPPF', 'C2PSA'):
            idx_to_out_bn[i] = base + '.cv2.bn'
        elif m_type == 'nn.Upsample':
            src = f if f >= 0 else i + f
            idx_to_out_bn[i] = idx_to_out_bn.get(src)
        elif m_type == 'Concat':
            src_list = [fi if fi >= 0 else i + fi for fi in (f if isinstance(f, list) else [f])]
            concat_bns = []
            for si in src_list:
                bn = idx_to_out_bn.get(si)
                if isinstance(bn, list):
                    concat_bns.extend(bn)
                elif bn is not None:
                    concat_bns.append(bn)
            idx_to_out_bn[i] = concat_bns

    # Step 2: Input BN(s) for each top-level layer
    idx_to_in_bn = {0: None}  # first layer: no input BN (3ch image)
    for i, (f, n, m_type, args) in enumerate(layers):
        if i == 0:
            continue
        if isinstance(f, list):
            src_list = [fi if fi >= 0 else i + fi for fi in f]
            in_bns = []
            for si in src_list:
                bn = idx_to_out_bn.get(si)
                if isinstance(bn, list):
                    in_bns.extend(bn)
                elif bn is not None:
                    in_bns.append(bn)
            idx_to_in_bn[i] = in_bns
        else:
            src = f if f >= 0 else i + f
            idx_to_in_bn[i] = idx_to_out_bn.get(src)

    # Step 2.5: Detect layer per-scale inputs
    # Detect head receives a list of feature maps [P3, P4, P5].
    # Map each scale index to its backbone output BN.
    detect_scale_inputs = {}  # {layer_idx: {scale_i: bn_name}}
    for i, (f, n, m_type, args) in enumerate(layers):
        if 'Detect' in str(m_type):
            f_list = f if isinstance(f, list) else [f]
            for scale_i, fi in enumerate(f_list):
                src = fi if fi >= 0 else i + fi
                bn = idx_to_out_bn.get(src)
                detect_scale_inputs.setdefault(i, {})[scale_i] = bn

    # Step 3: Per-conv mapping
    mapping = {}
    for name, m in model.named_modules():
        if not (isinstance(m, nn.Conv2d) and name.endswith('.conv')):
            continue

        out_bn = name[:-4] + 'bn'
        is_dw = (m.groups == m.in_channels and m.in_channels > 1)
        parts = name.split('.')
        layer_idx = int(parts[1])
        sub = '.'.join(parts[2:])  # e.g. 'conv', 'cv1.conv', 'm.0.m.0.cv1.conv'

        # Resolve in_bn
        if is_dw:
            in_bn = out_bn                          # depthwise: in = out (tied)
        elif sub == 'conv':
            in_bn = idx_to_in_bn.get(layer_idx)     # simple Conv layer
        elif sub == 'cv1.conv':
            in_bn = idx_to_in_bn.get(layer_idx)     # first conv of C3k2/SPPF/C2PSA
        elif sub == 'cv2.conv':
            m_type = layers[layer_idx][2] if layer_idx < len(layers) else ''
            if m_type in ('SPPF', 'C2PSA'):
                in_bn = f"model.{layer_idx}.cv1.bn"
            elif m_type == 'C3k2':
                # cv2 = cat(left_half + bottleneck outputs) → cv1.bn dominant
                in_bn = f"model.{layer_idx}.cv1.bn"
            else:
                in_bn = None
        elif layer_idx in detect_scale_inputs:
            # Detect head convs (cv2/cv3/one2one_cv2/one2one_cv3)
            in_bn = _resolve_detect_in_bn(name, layer_idx,
                                          detect_scale_inputs[layer_idx])
        else:
            # Internal bottleneck convs
            in_bn = _resolve_internal_in_bn(name, layer_idx, bn_channels)

        mapping[name] = {
            'out_bn': out_bn,
            'in_bn': in_bn,
            'is_depthwise': is_dw,
        }

    return mapping, bn_channels


def _resolve_internal_in_bn(conv_name, layer_idx, bn_channels):
    """
    Resolve in_bn for internal module convs inside C3k2 blocks.

    Determines module type from ACTUAL model structure (not YAML) by checking
    whether cv3.bn exists (C3k has cv3, Bottleneck does not).

    Handles 3 naming patterns:

    Pattern A - 4 sub_parts (m.J.cvN.conv):
      C3k:       cv1/cv2 parallel (both take chunk), cv3 after concat
      Bottleneck: cv1→cv2 sequential

    Pattern B - 6 sub_parts (m.J.m.K.cvN.conv):
      Bottleneck[K] inside C3k[J]

    Pattern C - 5 sub_parts (m.J.K.cvN.conv):
      Attn Sequential: nn.Sequential(Bottleneck, PSABlock)

    Args:
        conv_name: Full conv name (e.g., 'model.6.m.0.cv1.conv')
        layer_idx: Top-level layer index
        bn_channels: Dict {bn_name: num_features} from actual model
    """
    parts = conv_name.split('.')
    sub_parts = parts[2:]  # after 'model.X'
    n_sub = len(sub_parts)

    # ---- Pattern A: m.J.cvN.conv (4 sub_parts) ----
    if n_sub == 4 and sub_parts[0] == 'm':
        j = int(sub_parts[1])
        cv_name = sub_parts[2]  # 'cv1', 'cv2', or 'cv3'

        # C3k cv3: cat(m_out, cv2_out) → proxy: cv1.bn
        if cv_name == 'cv3':
            return f"model.{layer_idx}.m.{j}.cv1.bn"

        # Check actual module type: C3k has cv3.bn, Bottleneck does not
        is_c3k = f"model.{layer_idx}.m.{j}.cv3.bn" in bn_channels

        if is_c3k:
            # C3k cv1 and cv2 are PARALLEL - both take same input
            if j == 0:
                return f"model.{layer_idx}.cv1.bn"  # chunk right_half
            else:
                return f"model.{layer_idx}.m.{j-1}.cv3.bn"  # prev C3k output
        else:
            # Bottleneck[J] directly inside C3k2
            if cv_name == 'cv1':
                if j == 0:
                    return f"model.{layer_idx}.cv1.bn"  # chunk right_half
                else:
                    return f"model.{layer_idx}.m.{j-1}.cv2.bn"  # prev Bottleneck
            elif cv_name == 'cv2':
                return f"model.{layer_idx}.m.{j}.cv1.bn"  # this Bottleneck's cv1

    # ---- Pattern B: m.J.m.K.cvN.conv (6 sub_parts) ----
    # Bottleneck[K] inside C3k[J]
    if n_sub == 6 and sub_parts[0] == 'm' and sub_parts[2] == 'm':
        j = int(sub_parts[1])
        k = int(sub_parts[3])
        cv_name = sub_parts[4]

        if cv_name == 'cv2':
            # Bottleneck cv2 → input from this Bottleneck's cv1
            return conv_name.replace('.cv2.conv', '.cv1.bn')
        elif cv_name == 'cv1':
            if k == 0:
                # First Bottleneck → input from C3k[J].cv1 output
                return f"model.{layer_idx}.m.{j}.cv1.bn"
            else:
                # Later Bottleneck → input from prev Bottleneck's cv2
                return f"model.{layer_idx}.m.{j}.m.{k-1}.cv2.bn"

    # ---- Pattern C: m.J.K.cvN.conv (5 sub_parts) ----
    # Attn: nn.Sequential(Bottleneck[K=0], PSABlock[K=1]) inside C3k2.m[J]
    if n_sub == 5 and sub_parts[0] == 'm':
        j = int(sub_parts[1])   # ModuleList index
        # k = int(sub_parts[2])  # Sequential index (0=Bottleneck)
        cv_name = sub_parts[3]

        if cv_name == 'cv2':
            return conv_name.replace('.cv2.conv', '.cv1.bn')
        elif cv_name == 'cv1':
            if j == 0:
                return f"model.{layer_idx}.cv1.bn"  # chunk right_half
            else:
                # Prev Sequential output (PSABlock) - not pruned → None fallback
                return None

    return None


def _resolve_detect_in_bn(conv_name, layer_idx, scale_inputs):
    """
    Resolve in_bn for Detect head convs.

    Detect head structure per scale I:
        cv2[I] = Sequential(Conv[0], Conv[1], nn.Conv2d[2])
        cv3[I] = Sequential(Sequential(DW[0], PW[1]), Sequential(DW[0], PW[1]), nn.Conv2d[2])
        one2one_cv2/one2one_cv3 follow same structure.

    Args:
        conv_name: e.g. 'model.23.cv2.0.1.conv'
        layer_idx: e.g. 23
        scale_inputs: {0: 'model.16.cv2.bn', 1: 'model.19.cv2.bn', 2: 'model.22.cv2.bn'}
    """
    parts = conv_name.split('.')
    sub_parts = parts[2:-1]  # after 'model.X', before 'conv'
    # e.g. ['cv2', '0', '1'] or ['cv3', '0', '0', '1'] or ['one2one_cv2', '0', '0']

    branch = sub_parts[0]
    indices = sub_parts[1:]

    if branch in ('cv2', 'one2one_cv2'):
        # cv2.I.J → indices = ['I', 'J']
        scale_i = int(indices[0])
        j = int(indices[1])
        if j == 0:
            return scale_inputs.get(scale_i)
        else:
            return f"model.{layer_idx}.{branch}.{scale_i}.{j-1}.bn"

    elif branch in ('cv3', 'one2one_cv3'):
        # cv3.I.J.K → indices = ['I', 'J', 'K']
        scale_i = int(indices[0])
        j = int(indices[1])
        k = int(indices[2])
        if k == 0:
            # DW conv — normally caught by is_dw check before reaching here.
            # Provide correct source anyway for robustness.
            if j == 0:
                return scale_inputs.get(scale_i)
            else:
                return f"model.{layer_idx}.{branch}.{scale_i}.{j-1}.1.bn"
        else:
            # Pointwise after DW
            return f"model.{layer_idx}.{branch}.{scale_i}.{j}.0.bn"

    return None


def compute_resource_loss(a_params, conv_flops, conv_bn_map, bn_channels,
                          total_flops, target_ratio):
    """
    GFLOPs-based resource constraint (differentiable w.r.t a params).

    Paper Eq. 5-6:
        loss = log(r_e / r_t) if r_e > r_t, else 0

    Exact per-conv FLOPs:
        - regular:   effective = original × (1-a_in) × (1-a_out)
        - depthwise: effective = original × (1-a_out)
        - concat in: weighted average retention by channel count

    Args:
        a_params:     Dict {bn_name: nn.Parameter}
        conv_flops:   Dict {conv_name: flops} from profiling
        conv_bn_map:  Dict {conv_name: {'out_bn', 'in_bn', 'is_depthwise'}}
        bn_channels:  Dict {bn_name: num_features}
        total_flops:  Total original FLOPs
        target_ratio: Target pruning ratio (e.g., 0.3 = remove 30%)

    Returns:
        torch.Tensor: Resource constraint loss (scalar, differentiable)
    """
    device = next(iter(a_params.values())).device
    effective_flops = torch.tensor(0.0, device=device)

    for conv_name, flops in conv_flops.items():
        info = conv_bn_map.get(conv_name)
        if info is None:
            effective_flops = effective_flops + flops
            continue

        out_bn = info['out_bn']
        in_bn = info.get('in_bn')
        is_dw = info['is_depthwise']

        # Output retention
        a_out = a_params.get(out_bn)
        retain_out = (1.0 - a_out) if a_out is not None else 1.0

        # Compute ratio
        if is_dw:
            ratio = retain_out
        elif in_bn is None:
            # Cannot resolve input → assume input not prunable
            ratio = retain_out
        elif isinstance(in_bn, list):
            # Concat input: weighted average retention
            total_ch = 0.0
            weighted_retain = torch.tensor(0.0, device=device)
            for bn_name in in_bn:
                ch = bn_channels.get(bn_name, 0)
                a_in = a_params.get(bn_name)
                r = (1.0 - a_in) if a_in is not None else 1.0
                total_ch += ch
                weighted_retain = weighted_retain + ch * r
            retain_in = weighted_retain / total_ch if total_ch > 0 else 1.0
            ratio = retain_in * retain_out
        else:
            # Single input BN
            a_in = a_params.get(in_bn)
            retain_in = (1.0 - a_in) if a_in is not None else 1.0
            ratio = retain_in * retain_out

        effective_flops = effective_flops + flops * ratio

    r_e = effective_flops / total_flops
    r_t = 1.0 - target_ratio

    if r_e > r_t:
        return torch.log(r_e / r_t)
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


def extract_ratios_from_checkpoint(ckpt_path, save_path='dms_ratios.yaml', divisor=8):
    """
    Extract learned `a` params from DMS training checkpoint → YAML.

    Clamp per layer theo paper: a ∈ [0, 1 - x_min/x_max]
    x_min = divisor (minimum channels to keep), x_max = original channels.

    Output YAML can be used with: prune.py --layer-ratio dms_ratios.yaml

    Args:
        ckpt_path: Path to checkpoint (.pt)
        save_path: Output YAML path
        divisor: Minimum channels to keep per layer (default 8)

    Returns:
        dict: {bn_name: pruning_ratio}
    """
    import yaml

    ckpt = torch.load(ckpt_path, weights_only=False)
    a_params = ckpt.get('dms_a_params', {})
    if not a_params:
        raise ValueError(f"No dms_a_params found in {ckpt_path}!")

    # Get BN channel counts from model state_dict
    state_dict = ckpt.get('model', ckpt).state_dict() if hasattr(ckpt.get('model', {}), 'state_dict') else ckpt.get('state_dict', {})
    bn_channels = {}
    for key, val in state_dict.items():
        if key.endswith('.bn.weight') or key.endswith('.bn.running_mean'):
            bn_name = key.rsplit('.', 1)[0]  # remove .weight/.running_mean
            bn_channels[bn_name] = val.shape[0]

    ratios = {}
    for name, a in a_params.items():
        n_channels = bn_channels.get(name, 256)
        a_max = 1.0 - divisor / n_channels
        val = float(a.clamp(0.0, a_max).item())
        ratios[name] = round(val, 4)

    with open(save_path, 'w') as f:
        yaml.dump(ratios, f, default_flow_style=False, sort_keys=True)

    # Print summary
    avg = sum(ratios.values()) / len(ratios)
    print(f"Extracted {len(ratios)} DMS ratios to {save_path} (divisor={divisor})")
    print(f"  Average pruning ratio: {avg:.4f}")
    print(f"  Min: {min(ratios.values()):.4f}, Max: {max(ratios.values()):.4f}")

    return ratios
