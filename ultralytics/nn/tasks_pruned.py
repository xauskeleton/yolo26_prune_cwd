# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pruned model parsing and building for YOLOv26."""

import contextlib
from copy import deepcopy

import torch
import torch.nn as nn

from ultralytics.nn.tasks import BaseModel
from ultralytics.nn.modules.conv import Conv, DWConv, Concat
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.modules.block_pruned import (
    BottleneckPruned,
    C3kPruned,
    C3k2Pruned,
    C3k2PrunedAttn,
    SPPFPruned,
    C2PSAPruned,
)

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights, scale_img


class DetectionModelPruned(BaseModel):
    """
    YOLOv26 Detection Model - Pruned version.

    forward继承BaseModel
    """

    def __init__(self, maskbndict, cfg, ch=3, nc=None, verbose=True):
        """
        Initialize the YOLOv26 detection model with the given config and parameters.

        Args:
            maskbndict (dict): Pruning masks dict
            cfg (dict): Model config từ YAML
            ch (int): Input channels (default: 3 for RGB)
            nc (int): Number of classes (optional, từ cfg nếu không có)
            verbose (bool): Print model info
        """
        super().__init__()
        self.yaml = cfg
        # 注意这里一定要deepcopy一下cfg, 因为当模型剪枝过程全部结束时我们要把模型保存下来, 包括模型配置信息
        # 所以我们并不想改变模型配置信息, 如果改变, 当再次用配置信息去构建网络的时候就会出错
        self.model, self.save, self.current_to_prev = parse_model_pruned(
            maskbndict, deepcopy(cfg), ch, verbose
        )
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, DetectPruned):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            # DetectPruned expects list of tensors as input
            # Create dummy inputs for each detection scale
            with torch.no_grad():
                # Tạm thời chuyển sang eval mode để forward trả về list of tensors
                self.eval()

                # Tạo dummy input tensor
                dummy_input = torch.zeros(1, ch, s, s)

                # Forward qua toàn bộ model để lấy feature maps
                # DetectPruned nhận list of feature maps từ các scales
                forward_output = self.forward(dummy_input)

                # Trong eval mode, forward trả về (y, preds) hoặc y
                # y là prediction output, preds là dict chứa feature maps
                if isinstance(forward_output, tuple):
                    y, preds = forward_output
                    # Lấy feature maps từ preds
                    if isinstance(preds, dict):
                        if 'one2many' in preds:
                            feats = preds['one2many'].get('feats', [])
                        else:
                            feats = preds.get('feats', [])
                    else:
                        feats = []
                else:
                    feats = []

                # Tính stride từ feature maps
                if feats and len(feats) > 0:
                    m.stride = torch.tensor([s / x.shape[-2] for x in feats])
                else:
                    # Fallback: tính stride dựa trên số scales
                    # Giả định 3 scales với stride 8, 16, 32
                    m.stride = torch.tensor([8., 16., 32.])

                # Quay lại training mode
                self.train()

            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)


def parse_model_pruned(maskbndict, d, ch, verbose=True):
    """
    Parse pruned model từ YAML config và pruning masks.

    网络构建(ch是个列表, 记录着每一层的输出通道数; current_to_prev是一个字典,
    记录着{某一bn层的名字: 该bn层连接的上一(多)bn层的名字}):

    YOLOv26 modules:
        - Conv: 基础卷积层
        - DWConv: Depthwise卷积 (cv3 branch中使用)
        - BottleneckPruned: 带残差的Bottleneck
        - C3kPruned: C3k模块
        - C3k2Pruned: C3k2模块 (split + C3k modules)
        - SPPFPruned: SPPF模块
        - C2PSAPruned: C2PSA模块 (split + PSA)
        - DetectPruned: 检测头 (DWConv pattern for cv3)

    Args:
        maskbndict (dict): Pruning masks
            Format: {'model.{i}.bn': tensor([True, False, ...]), ...}
        d (dict): Model dict từ YAML
        ch (int): Input channels (3 for RGB)
        verbose (bool): Print layer info

    Returns:
        (nn.Sequential, list, dict): Model, save list, current_to_prev dict
    """
    import ast

    # Parse args
    nc = d.get('nc', 80)
    reg_max = d.get('reg_max', 1)  # YOLOv26 default
    end2end = d.get('end2end', False)
    act = d.get('activation')
    depth = d.get('depth_multiple', 1.0)

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<50}{'arguments':<30}")

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]

    # Track dependencies: {某一bn层的名字: 该bn层连接的上一(多)bn层的名字}
    current_to_prev = {}
    # Track layer indices: {索引: 本层最后一个bn层的名字}
    idx_to_bn_layer_name = {}
    prev_bn_layer_name = None
    prev_module = None

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # Parse args
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        base_name = f'model.{i}'

        # ═══════════════════════════════════════════════════════
        # Module parsing
        # ═══════════════════════════════════════════════════════

        if m == 'Conv':
            # ─────────── CONV ───────────
            c1 = ch[f]
            bn_layer_name = base_name + '.bn'
            mask = maskbndict[bn_layer_name]
            c2 = torch.sum(mask).int().item()
            # args từ YAML: [original_c2, k, s, ...] → bỏ args[0], giữ k, s, ...
            args = [c1, c2, *args[1:]]  # FIX: skip args[0] (original channel spec)

            # Track dependencies
            if i == 0:
                prev_bn_layer_name = bn_layer_name
            else:
                current_to_prev[bn_layer_name] = prev_bn_layer_name
                prev_bn_layer_name = bn_layer_name
            idx_to_bn_layer_name[i] = bn_layer_name

            m = Conv

        elif m == 'DWConv':
            # ─────────── DWCONV ───────────
            c1 = ch[f]
            bn_layer_name = base_name + '.bn'
            mask = maskbndict[bn_layer_name]
            c2 = torch.sum(mask).int().item()
            # args từ YAML: [original_c2, k, s, ...] → bỏ args[0]
            args = [c1, c2, *args[1:]]  # FIX: skip args[0]

            # Track dependencies
            if prev_bn_layer_name:
                current_to_prev[bn_layer_name] = prev_bn_layer_name
            prev_bn_layer_name = bn_layer_name
            idx_to_bn_layer_name[i] = bn_layer_name

            m = DWConv

        elif m == 'BottleneckPruned':
            # ─────────── BOTTLENECK PRUNED ───────────
            c1 = ch[f]
            cv1_bn_name = base_name + '.cv1.bn'
            cv2_bn_name = base_name + '.cv2.bn'

            cv1_mask = maskbndict[cv1_bn_name]
            cv2_mask = maskbndict[cv2_bn_name]

            cv1out = torch.sum(cv1_mask).int().item()
            cv2out = torch.sum(cv2_mask).int().item()

            args = [c1, cv1out, cv2out, *args]
            c2 = cv2out

            # Track dependencies
            current_to_prev[cv1_bn_name] = prev_bn_layer_name
            current_to_prev[cv2_bn_name] = cv1_bn_name
            prev_bn_layer_name = cv2_bn_name
            idx_to_bn_layer_name[i] = cv2_bn_name

            m = BottleneckPruned

        elif m == 'C3kPruned':
            # ─────────── C3K PRUNED ───────────
            c1 = ch[f]
            cv1_bn_name = base_name + '.cv1.bn'
            cv2_bn_name = base_name + '.cv2.bn'
            cv3_bn_name = base_name + '.cv3.bn'

            cv1_mask = maskbndict[cv1_bn_name]
            cv2_mask = maskbndict[cv2_bn_name]
            cv3_mask = maskbndict[cv3_bn_name]

            cv1out = torch.sum(cv1_mask).int().item()
            cv2out = torch.sum(cv2_mask).int().item()
            cv3out = torch.sum(cv3_mask).int().item()

            # Parse bottleneck channels (giống C3k2Pruned parse C3k modules)
            bottleneck_indices = []
            for j in range(10):  # Max 10 bottlenecks
                test_key = base_name + f'.m.{j}.cv1.bn'
                if test_key in maskbndict:
                    bottleneck_indices.append(j)
                else:
                    break

            n_bottlenecks = len(bottleneck_indices)
            bottleneck_cv1outs = []
            bottleneck_cv2outs = []

            for j in bottleneck_indices:
                bn_cv1_name = base_name + f'.m.{j}.cv1.bn'
                bn_cv2_name = base_name + f'.m.{j}.cv2.bn'

                bottleneck_cv1outs.append(torch.sum(maskbndict[bn_cv1_name]).int().item())
                bottleneck_cv2outs.append(torch.sum(maskbndict[bn_cv2_name]).int().item())

            args = [c1, cv1out, cv2out, cv3out,
                    bottleneck_cv1outs, bottleneck_cv2outs,
                    n_bottlenecks, *args]
            c2 = cv3out

            # Track dependencies
            current_to_prev[cv1_bn_name] = prev_bn_layer_name
            current_to_prev[cv2_bn_name] = prev_bn_layer_name
            current_to_prev[cv3_bn_name] = [cv1_bn_name, cv2_bn_name]

            # Track bottlenecks
            for j in bottleneck_indices:
                bn_cv1_name = base_name + f'.m.{j}.cv1.bn'
                bn_cv2_name = base_name + f'.m.{j}.cv2.bn'

                if j == 0:
                    current_to_prev[bn_cv1_name] = cv1_bn_name
                else:
                    prev_bn_cv2 = base_name + f'.m.{j - 1}.cv2.bn'
                    current_to_prev[bn_cv1_name] = prev_bn_cv2

                current_to_prev[bn_cv2_name] = bn_cv1_name

            prev_bn_layer_name = cv3_bn_name
            idx_to_bn_layer_name[i] = cv3_bn_name

            m = C3kPruned
            n = 1  # C3kPruned đã nhận n trong args

        elif m == 'C3k2Pruned':
            # ─────────── C3K2 PRUNED ───────────
            c1 = ch[f]

            # Get masks
            cv1_bn_name = base_name + '.cv1.bn'
            cv1_mask = maskbndict[cv1_bn_name]
            cv1out = torch.sum(cv1_mask).int().item()

            # Split sections
            cv1_split_sections = [
                torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(),
                torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()
            ]

            # Inner C3k modules - DETECT ĐỘNG số modules thực tế
            # Vì pruned model có thể có ít hơn n modules (ví dụ YAML n=3 nhưng chỉ có m.0)
            c3k_indices = []
            for j in range(10):  # Max 10 C3k modules
                test_key = base_name + f'.m.{j}.cv1.bn'
                if test_key in maskbndict:
                    c3k_indices.append(j)
                else:
                    break  # Không còn module nào nữa

            n_c3k = len(c3k_indices)
            c3k_cv1outs = []
            c3k_cv2outs = []
            c3k_cv3outs = []
            c3k_bottleneck_cv1outs = []  # List of lists
            c3k_bottleneck_cv2outs = []  # List of lists
            c3k_n_bottlenecks = []  # Số bottleneck cho mỗi C3k

            for j in c3k_indices:
                c3k_cv1_name = base_name + f'.m.{j}.cv1.bn'
                c3k_cv2_name = base_name + f'.m.{j}.cv2.bn'
                c3k_cv3_name = base_name + f'.m.{j}.cv3.bn'

                c3k_cv1outs.append(torch.sum(maskbndict[c3k_cv1_name]).int().item())
                c3k_cv2outs.append(torch.sum(maskbndict[c3k_cv2_name]).int().item())
                c3k_cv3outs.append(torch.sum(maskbndict[c3k_cv3_name]).int().item())

                # Parse bottlenecks bên trong C3k này
                bn_indices = []
                for k in range(10):  # Max 10 bottlenecks per C3k
                    bn_key = base_name + f'.m.{j}.m.{k}.cv1.bn'
                    if bn_key in maskbndict:
                        bn_indices.append(k)
                    else:
                        break

                bn_cv1outs = []
                bn_cv2outs = []
                for k in bn_indices:
                    bn_cv1_key = base_name + f'.m.{j}.m.{k}.cv1.bn'
                    bn_cv2_key = base_name + f'.m.{j}.m.{k}.cv2.bn'
                    bn_cv1outs.append(torch.sum(maskbndict[bn_cv1_key]).int().item())
                    bn_cv2outs.append(torch.sum(maskbndict[bn_cv2_key]).int().item())

                c3k_bottleneck_cv1outs.append(bn_cv1outs)
                c3k_bottleneck_cv2outs.append(bn_cv2outs)
                c3k_n_bottlenecks.append(len(bn_indices))

            # cv2
            cv2_bn_name = base_name + '.cv2.bn'
            cv2_mask = maskbndict[cv2_bn_name]
            cv2out = torch.sum(cv2_mask).int().item()

            # Parse n_bottlenecks from args
            # YAML: [256, True] → bool, skip (dùng default)
            # YAML: [256, 2, True] → int, n_bottlenecks=2
            # Note: isinstance(True, int) = True trong Python, nên phải exclude bool
            if len(args) >= 2 and isinstance(args[1], int) and not isinstance(args[1], bool):
                n_bottlenecks = args[1]
                rest_args = args[2:]  # shortcut, g, k, e
            else:
                n_bottlenecks = 2  # default
                rest_args = args[1:]  # shortcut, e

            args = [
                c1, cv1out, cv1_split_sections,
                c3k_cv1outs, c3k_cv2outs, c3k_cv3outs,
                c3k_bottleneck_cv1outs, c3k_bottleneck_cv2outs, c3k_n_bottlenecks,
                cv2out, n_c3k, n_bottlenecks, *rest_args
            ]
            c2 = cv2out

            # Track dependencies
            current_to_prev[cv1_bn_name] = prev_bn_layer_name
            if prev_module == 'Concat':
                fx = ([f if f >= 0 else i + f] if isinstance(f, int)
                      else [ix if ix >= 0 else i + ix for ix in f])
                all_bns = []
                for ix in fx:
                    bn = idx_to_bn_layer_name[ix]
                    if isinstance(bn, list):
                        all_bns.extend(bn)
                    else:
                        all_bns.append(bn)
                current_to_prev[cv1_bn_name] = all_bns

            prev_bn_layer_name = cv1_bn_name

            # Track C3k modules
            prev_bn_layer_names_for_cv2 = [cv1_bn_name]
            for j_idx, j in enumerate(c3k_indices):
                c3k_cv1_name = base_name + f'.m.{j}.cv1.bn'
                c3k_cv2_name = base_name + f'.m.{j}.cv2.bn'
                c3k_cv3_name = base_name + f'.m.{j}.cv3.bn'

                current_to_prev[c3k_cv1_name] = prev_bn_layer_name
                current_to_prev[c3k_cv2_name] = prev_bn_layer_name
                current_to_prev[c3k_cv3_name] = [c3k_cv1_name, c3k_cv2_name]

                # Track bottlenecks bên trong C3k này
                bn_count = c3k_n_bottlenecks[j_idx]
                for k in range(bn_count):
                    bn_cv1_key = base_name + f'.m.{j}.m.{k}.cv1.bn'
                    bn_cv2_key = base_name + f'.m.{j}.m.{k}.cv2.bn'

                    if k == 0:
                        current_to_prev[bn_cv1_key] = c3k_cv1_name
                    else:
                        prev_bn_cv2_key = base_name + f'.m.{j}.m.{k - 1}.cv2.bn'
                        current_to_prev[bn_cv1_key] = prev_bn_cv2_key

                    current_to_prev[bn_cv2_key] = bn_cv1_key

                prev_bn_layer_name = c3k_cv3_name
                prev_bn_layer_names_for_cv2.append(c3k_cv3_name)

            current_to_prev[cv2_bn_name] = prev_bn_layer_names_for_cv2
            prev_bn_layer_name = cv2_bn_name
            idx_to_bn_layer_name[i] = cv2_bn_name

            m = C3k2Pruned
            n = 1

        elif m == 'C3k2PrunedAttn':
            # ─────────── C3K2 PRUNED ATTN ───────────
            # Dành cho C3k2 có attn=True: m = Sequential(Bottleneck, PSABlock)
            # cv1.bn và PSABlock BNs nằm trong ignore_bn_list → không prune
            c1 = ch[f]

            # cv1 - KHÔNG prune (trong ignore_bn_list)
            cv1_bn_name = base_name + '.cv1.bn'
            cv1_mask = maskbndict[cv1_bn_name]   # mask = all ones
            cv1out = torch.sum(cv1_mask).int().item()

            cv1_split_sections = [
                torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(),
                torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()
            ]

            # Tìm Sequential(Bottleneck, PSABlock) blocks
            # Key: model.X.m.{j}.0.cv1.bn (Bottleneck tại vị trí j trong Sequential)
            seq_indices = []
            for j in range(10):
                test_key = base_name + f'.m.{j}.0.cv1.bn'
                if test_key in maskbndict:
                    seq_indices.append(j)
                else:
                    break

            n_blocks = len(seq_indices)
            bottleneck_cv1outs = []
            for j in seq_indices:
                bn_cv1_name = base_name + f'.m.{j}.0.cv1.bn'
                bottleneck_cv1outs.append(torch.sum(maskbndict[bn_cv1_name]).int().item())

            # cv2 - CÓ THỂ prune
            cv2_bn_name = base_name + '.cv2.bn'
            cv2_mask = maskbndict[cv2_bn_name]
            cv2out = torch.sum(cv2_mask).int().item()

            # Parse args - [1024, True] → shortcut=True
            shortcut = args[1] if len(args) >= 2 and isinstance(args[1], bool) else True

            args = [c1, cv1out, cv1_split_sections, n_blocks, bottleneck_cv1outs, cv2out, shortcut]
            c2 = cv2out

            # Track dependencies
            current_to_prev[cv1_bn_name] = prev_bn_layer_name
            if prev_module == 'Concat':
                fx = ([f if f >= 0 else i + f] if isinstance(f, int)
                      else [ix if ix >= 0 else i + ix for ix in f])
                all_bns = []
                for ix in fx:
                    bn = idx_to_bn_layer_name[ix]
                    if isinstance(bn, list):
                        all_bns.extend(bn)
                    else:
                        all_bns.append(bn)
                current_to_prev[cv1_bn_name] = all_bns

            # Track Bottleneck cv1 và cv2 trong Sequential
            prev_bn_for_cv2 = [cv1_bn_name]
            for j in seq_indices:
                bn_cv1_name = base_name + f'.m.{j}.0.cv1.bn'
                bn_cv2_name = base_name + f'.m.{j}.0.cv2.bn'
                # Bottleneck.cv1 nhận right_half của cv1 (tracking qua cv1_bn_name)
                current_to_prev[bn_cv1_name] = cv1_bn_name
                # Bottleneck.cv2 nhận output của cv1
                current_to_prev[bn_cv2_name] = bn_cv1_name
                prev_bn_for_cv2.append(bn_cv2_name)

            # cv2 phụ thuộc vào cv1 + tất cả Bottleneck outputs (qua PSABlock = right_half)
            current_to_prev[cv2_bn_name] = prev_bn_for_cv2

            prev_bn_layer_name = cv2_bn_name
            idx_to_bn_layer_name[i] = cv2_bn_name

            m = C3k2PrunedAttn
            n = 1

        elif m == 'SPPFPruned':
            # ─────────── SPPF PRUNED ───────────
            c1 = ch[f]
            cv1_bn_name = base_name + '.cv1.bn'
            cv2_bn_name = base_name + '.cv2.bn'

            cv1_mask = maskbndict[cv1_bn_name]
            cv2_mask = maskbndict[cv2_bn_name]

            cv1out = torch.sum(cv1_mask).int().item()
            cv2out = torch.sum(cv2_mask).int().item()

            # Parse args - YAML: [1024, 5] or [1024, 5, 3, False]
            # args[0] = channels (bỏ)
            # args[1] = k (kernel size)
            # args[2] = n (optional, default 3)
            # args[3] = shortcut (optional, default False)
            k = args[1] if len(args) >= 2 else 5
            n_param = args[2] if len(args) >= 3 else 3
            shortcut = args[3] if len(args) >= 4 else False

            args = [c1, cv1out, cv2out, k, n_param, shortcut]
            c2 = cv2out

            # Track dependencies
            current_to_prev[cv1_bn_name] = prev_bn_layer_name
            current_to_prev[cv2_bn_name] = cv1_bn_name
            prev_bn_layer_name = cv2_bn_name
            idx_to_bn_layer_name[i] = cv2_bn_name

            m = SPPFPruned
            n = 1  # SPPFPruned đã nhận n trong args

        elif m == 'C2PSAPruned':
            # ─────────── C2PSA PRUNED ───────────
            c1 = ch[f]

            # cv1 + split
            cv1_bn_name = base_name + '.cv1.bn'
            cv1_mask = maskbndict[cv1_bn_name]
            cv1out = torch.sum(cv1_mask).int().item()

            cv1_split_sections = [
                torch.sum(cv1_mask.chunk(2, 0)[0]).int().item(),
                torch.sum(cv1_mask.chunk(2, 0)[1]).int().item()
            ]

            # cv2
            cv2_bn_name = base_name + '.cv2.bn'
            cv2_mask = maskbndict[cv2_bn_name]
            cv2out = torch.sum(cv2_mask).int().item()

            # Auto-detect PSABlock count từ masks (robust hơn dùng YAML n)
            n_psa = 0
            for j in range(10):
                test_key = base_name + f'.m.{j}.attn.qkv.bn'
                if test_key in maskbndict:
                    n_psa += 1
                else:
                    break
            actual_n = n_psa if n_psa > 0 else n

            # Parse args - YAML: [1024] or [1024, 0.5]
            # args[0] = channels (bỏ), args[1] = e (optional)
            e = args[1] if len(args) >= 2 else 0.5

            args = [c1, cv1out, cv1_split_sections, cv2out, actual_n, e]
            c2 = cv2out

            # Track dependencies
            current_to_prev[cv1_bn_name] = prev_bn_layer_name
            prev_bn_layer_name = cv1_bn_name
            # PSA modules don't have BN, cv2 connects to cv1
            current_to_prev[cv2_bn_name] = cv1_bn_name
            prev_bn_layer_name = cv2_bn_name
            idx_to_bn_layer_name[i] = cv2_bn_name

            m = C2PSAPruned
            n = 1  # C2PSAPruned đã nhận n trong args, không wrap trong Sequential

        elif m == 'DetectPruned':
            # ─────────── DETECT PRUNED ───────────
            # YOLOv26 Detect với DWConv pattern

            input_chs = [ch[x] for x in f]
            nl = len(f)  # 3 scales

            # cv2 (bbox branch) - simple Conv
            cv2x0_outs = []
            cv2x1_outs = []
            cv2x0_bn_names = []
            cv2x1_bn_names = []
            cv2x2_conv_names = []

            for scale_idx in range(nl):
                cv2x0_name = base_name + f'.cv2.{scale_idx}.0.bn'
                cv2x1_name = base_name + f'.cv2.{scale_idx}.1.bn'
                cv2x2_name = base_name + f'.cv2.{scale_idx}.2'

                cv2x0_outs.append(torch.sum(maskbndict[cv2x0_name]).int().item())
                cv2x1_outs.append(torch.sum(maskbndict[cv2x1_name]).int().item())

                cv2x0_bn_names.append(cv2x0_name)
                cv2x1_bn_names.append(cv2x1_name)
                cv2x2_conv_names.append(cv2x2_name)

            # cv3 (class branch) - DWConv pattern
            cv3x0_dw_outs = []
            cv3x0_pw_outs = []
            cv3x1_dw_outs = []
            cv3x1_pw_outs = []
            cv3x0_dw_bn_names = []
            cv3x0_pw_bn_names = []
            cv3x1_dw_bn_names = []
            cv3x1_pw_bn_names = []
            cv3x2_conv_names = []

            for scale_idx in range(nl):
                cv3x0_dw_name = base_name + f'.cv3.{scale_idx}.0.0.bn'  # DWConv
                cv3x0_pw_name = base_name + f'.cv3.{scale_idx}.0.1.bn'  # Conv
                cv3x1_dw_name = base_name + f'.cv3.{scale_idx}.1.0.bn'  # DWConv
                cv3x1_pw_name = base_name + f'.cv3.{scale_idx}.1.1.bn'  # Conv
                cv3x2_name = base_name + f'.cv3.{scale_idx}.2'

                cv3x0_dw_outs.append(torch.sum(maskbndict[cv3x0_dw_name]).int().item())
                cv3x0_pw_outs.append(torch.sum(maskbndict[cv3x0_pw_name]).int().item())
                cv3x1_dw_outs.append(torch.sum(maskbndict[cv3x1_dw_name]).int().item())
                cv3x1_pw_outs.append(torch.sum(maskbndict[cv3x1_pw_name]).int().item())

                cv3x0_dw_bn_names.append(cv3x0_dw_name)
                cv3x0_pw_bn_names.append(cv3x0_pw_name)
                cv3x1_dw_bn_names.append(cv3x1_dw_name)
                cv3x1_pw_bn_names.append(cv3x1_pw_name)
                cv3x2_conv_names.append(cv3x2_name)

            # one2one branches (if end2end)
            one2one_cv2x0_outs = None
            one2one_cv2x1_outs = None
            one2one_cv3x0_dw_outs = None
            one2one_cv3x0_pw_outs = None
            one2one_cv3x1_dw_outs = None
            one2one_cv3x1_pw_outs = None

            if end2end:
                one2one_cv2x0_outs = []
                one2one_cv2x1_outs = []
                one2one_cv3x0_dw_outs = []
                one2one_cv3x0_pw_outs = []
                one2one_cv3x1_dw_outs = []
                one2one_cv3x1_pw_outs = []

                for scale_idx in range(nl):
                    # one2one_cv2
                    o2o_cv2x0_name = base_name + f'.one2one_cv2.{scale_idx}.0.bn'
                    o2o_cv2x1_name = base_name + f'.one2one_cv2.{scale_idx}.1.bn'

                    if o2o_cv2x0_name in maskbndict and o2o_cv2x1_name in maskbndict:
                        one2one_cv2x0_outs.append(torch.sum(maskbndict[o2o_cv2x0_name]).int().item())
                        one2one_cv2x1_outs.append(torch.sum(maskbndict[o2o_cv2x1_name]).int().item())
                    else:
                        one2one_cv2x0_outs.append(cv2x0_outs[scale_idx])
                        one2one_cv2x1_outs.append(cv2x1_outs[scale_idx])

                    # one2one_cv3
                    o2o_cv3x0_dw_name = base_name + f'.one2one_cv3.{scale_idx}.0.0.bn'
                    o2o_cv3x0_pw_name = base_name + f'.one2one_cv3.{scale_idx}.0.1.bn'
                    o2o_cv3x1_dw_name = base_name + f'.one2one_cv3.{scale_idx}.1.0.bn'
                    o2o_cv3x1_pw_name = base_name + f'.one2one_cv3.{scale_idx}.1.1.bn'

                    if all(k in maskbndict for k in [o2o_cv3x0_dw_name, o2o_cv3x0_pw_name,
                                                     o2o_cv3x1_dw_name, o2o_cv3x1_pw_name]):
                        one2one_cv3x0_dw_outs.append(torch.sum(maskbndict[o2o_cv3x0_dw_name]).int().item())
                        one2one_cv3x0_pw_outs.append(torch.sum(maskbndict[o2o_cv3x0_pw_name]).int().item())
                        one2one_cv3x1_dw_outs.append(torch.sum(maskbndict[o2o_cv3x1_dw_name]).int().item())
                        one2one_cv3x1_pw_outs.append(torch.sum(maskbndict[o2o_cv3x1_pw_name]).int().item())
                    else:
                        one2one_cv3x0_dw_outs.append(cv3x0_dw_outs[scale_idx])
                        one2one_cv3x0_pw_outs.append(cv3x0_pw_outs[scale_idx])
                        one2one_cv3x1_dw_outs.append(cv3x1_dw_outs[scale_idx])
                        one2one_cv3x1_pw_outs.append(cv3x1_pw_outs[scale_idx])

            args = [
                cv2x0_outs, cv2x1_outs,
                cv3x0_dw_outs, cv3x0_pw_outs, cv3x1_dw_outs, cv3x1_pw_outs,
                one2one_cv2x0_outs, one2one_cv2x1_outs,
                one2one_cv3x0_dw_outs, one2one_cv3x0_pw_outs,
                one2one_cv3x1_dw_outs, one2one_cv3x1_pw_outs,
                nc, reg_max, tuple(input_chs)
            ]

            # Track dependencies
            for scale_idx in range(nl):
                # cv2 branch
                current_to_prev[cv2x0_bn_names[scale_idx]] = idx_to_bn_layer_name[f[scale_idx]]
                current_to_prev[cv2x1_bn_names[scale_idx]] = cv2x0_bn_names[scale_idx]
                current_to_prev[cv2x2_conv_names[scale_idx]] = cv2x1_bn_names[scale_idx]

                # cv3 branch
                current_to_prev[cv3x0_dw_bn_names[scale_idx]] = idx_to_bn_layer_name[f[scale_idx]]
                current_to_prev[cv3x0_pw_bn_names[scale_idx]] = cv3x0_dw_bn_names[scale_idx]
                current_to_prev[cv3x1_dw_bn_names[scale_idx]] = cv3x0_pw_bn_names[scale_idx]
                current_to_prev[cv3x1_pw_bn_names[scale_idx]] = cv3x1_dw_bn_names[scale_idx]
                current_to_prev[cv3x2_conv_names[scale_idx]] = cv3x1_pw_bn_names[scale_idx]

            # one2one branches (if end2end)
            if end2end:
                for scale_idx in range(nl):
                    # one2one_cv2 branch
                    o2o_cv2x0_name = base_name + f'.one2one_cv2.{scale_idx}.0.bn'
                    o2o_cv2x1_name = base_name + f'.one2one_cv2.{scale_idx}.1.bn'
                    o2o_cv2x2_name = base_name + f'.one2one_cv2.{scale_idx}.2'
                    current_to_prev[o2o_cv2x0_name] = idx_to_bn_layer_name[f[scale_idx]]
                    current_to_prev[o2o_cv2x1_name] = o2o_cv2x0_name
                    current_to_prev[o2o_cv2x2_name] = o2o_cv2x1_name

                    # one2one_cv3 branch
                    o2o_cv3x0_dw_name = base_name + f'.one2one_cv3.{scale_idx}.0.0.bn'
                    o2o_cv3x0_pw_name = base_name + f'.one2one_cv3.{scale_idx}.0.1.bn'
                    o2o_cv3x1_dw_name = base_name + f'.one2one_cv3.{scale_idx}.1.0.bn'
                    o2o_cv3x1_pw_name = base_name + f'.one2one_cv3.{scale_idx}.1.1.bn'
                    o2o_cv3x2_name = base_name + f'.one2one_cv3.{scale_idx}.2'
                    current_to_prev[o2o_cv3x0_dw_name] = idx_to_bn_layer_name[f[scale_idx]]
                    current_to_prev[o2o_cv3x0_pw_name] = o2o_cv3x0_dw_name
                    current_to_prev[o2o_cv3x1_dw_name] = o2o_cv3x0_pw_name
                    current_to_prev[o2o_cv3x1_pw_name] = o2o_cv3x1_dw_name
                    current_to_prev[o2o_cv3x2_name] = o2o_cv3x1_pw_name

            c2 = nc
            m = DetectPruned

        elif m == 'nn.Upsample':
            # ─────────── UPSAMPLE ───────────
            c2 = ch[f]
            idx_to_bn_layer_name[i] = idx_to_bn_layer_name[i - 1]
            m = nn.Upsample

        elif m == 'Concat':
            # ─────────── CONCAT ───────────
            c2 = sum(ch[x] for x in f)

            # Track indices for Concat
            # Concat không có BN riêng → lưu TẤT CẢ BN từ các inputs
            # để layer sau (C3k2Pruned) có thể tạo đúng in_channels_mask
            fx = [ix if ix >= 0 else i + ix for ix in f]
            concat_bns = []
            for ix in fx:
                if ix in idx_to_bn_layer_name:
                    bn = idx_to_bn_layer_name[ix]
                    if isinstance(bn, list):
                        concat_bns.extend(bn)
                    else:
                        concat_bns.append(bn)
                else:
                    concat_bns.append(prev_bn_layer_name)

            idx_to_bn_layer_name[i] = concat_bns

            m = Concat

        else:
            raise ValueError(f"ERROR ❌ module {m} not supported in parse_model_pruned.")

        prev_module = m if isinstance(m, str) else m.__name__

        # ═══════════════════════════════════════════════════════
        # Create module
        # ═══════════════════════════════════════════════════════
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t

        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<50}{str(args):<30}')

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)

        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save), current_to_prev


if __name__ == "__main__":
    import prune

    dummies = torch.randn([4, 3, 640, 640], dtype=torch.float32).cuda()
    opt = prune.parse_opt()
    maskbndict, pruned_yaml = prune.main(opt)
    model = DetectionModelPruned(maskbndict, pruned_yaml, ch=3)
    model.train().cuda()
    out = model(dummies)