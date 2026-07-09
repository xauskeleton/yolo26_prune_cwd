# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pruned model head modules for YOLOv26."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv

__all__ = ("DetectPruned",)


class DetectPruned(nn.Module):
    """YOLOv26 Detect head - Pruned version.

    Khác với Detect gốc:
    - Nhận channel sizes tuyệt đối từ pruning masks thay vì tính từ công thức
    - cv3 dùng DWConv pattern: (DWConv → Conv) → (DWConv → Conv) → Conv2d
    - Mỗi scale có thể có channels khác nhau
    - Hỗ trợ one2one branches với channels riêng biệt

    Structure:
        cv2 (bbox branch):
            Conv(ch[i] → cv2x0_out, k=3)
            Conv(cv2x0_out → cv2x1_out, k=3)
            Conv2d(cv2x1_out → 4*reg_max, k=1)

        cv3 (class branch):
            Sequential:
                DWConv(ch[i] → cv3x0_dw_out, k=3, g=ch[i])
                Conv(cv3x0_dw_out → cv3x0_pw_out, k=1)
            Sequential:
                DWConv(cv3x0_pw_out → cv3x1_dw_out, k=3, g=cv3x0_pw_out)
                Conv(cv3x1_dw_out → cv3x1_pw_out, k=1)
            Conv2d(cv3x1_pw_out → nc, k=1)

    Args:
        # cv2 one2many
        cv2x0_outs (list[int]): Output channels của cv2 layer 0 cho mỗi scale
        cv2x1_outs (list[int]): Output channels của cv2 layer 1 cho mỗi scale

        # cv3 one2many
        cv3x0_dw_outs (list[int]): DWConv output của cv3 layer 0 cho mỗi scale
        cv3x0_pw_outs (list[int]): Conv output của cv3 layer 0 cho mỗi scale
        cv3x1_dw_outs (list[int]): DWConv output của cv3 layer 1 cho mỗi scale
        cv3x1_pw_outs (list[int]): Conv output của cv3 layer 1 cho mỗi scale

        # one2one branches (optional, nếu end2end=True)
        one2one_cv2x0_outs (list[int]): cv2 layer 0 output cho one2one
        one2one_cv2x1_outs (list[int]): cv2 layer 1 output cho one2one
        one2one_cv3x0_dw_outs (list[int]): cv3 layer 0 DWConv output cho one2one
        one2one_cv3x0_pw_outs (list[int]): cv3 layer 0 Conv output cho one2one
        one2one_cv3x1_dw_outs (list[int]): cv3 layer 1 DWConv output cho one2one
        one2one_cv3x1_pw_outs (list[int]): cv3 layer 1 Conv output cho one2one
        nc (int): Number of classes
        reg_max (int): DFL channels
        ch (tuple): Input channels từ backbone [P3, P4, P5]

    Examples:
        >>> # Tạo DetectPruned với channels đã prune
        >>> detect = DetectPruned(
        ...     cv2x0_outs=[64, 64, 64],
        ...     cv2x1_outs=[64, 64, 64],
        ...     cv3x0_dw_outs=[256, 512, 512],
        ...     cv3x0_pw_outs=[80, 80, 80],
        ...     cv3x1_dw_outs=[80, 80, 80],
        ...     cv3x1_pw_outs=[80, 80, 80],
        ...     nc=80,
        ...     reg_max=1,
        ...     ch=(256, 512, 512),
        ... )
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 512, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(
        self,
        # cv2 one2many
        cv2x0_outs,
        cv2x1_outs,
        # cv3 one2many
        cv3x0_dw_outs,
        cv3x0_pw_outs,
        cv3x1_dw_outs,
        cv3x1_pw_outs,
        # one2one (optional)
        one2one_cv2x0_outs=None,
        one2one_cv2x1_outs=None,
        one2one_cv3x0_dw_outs=None,
        one2one_cv3x0_pw_outs=None,
        one2one_cv3x1_dw_outs=None,
        one2one_cv3x1_pw_outs=None,
        # common
        nc=80,
        reg_max=1,
        ch=(),
    ):
        super().__init__()

        if reg_max != 1:
            import warnings

            warnings.warn(
                f"YOLO26 uses reg_max=1 (no DFL), but got reg_max={reg_max}. This may cause compatibility issues."
            )

        self.nc = nc
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)

        # Validate inputs
        assert len(cv2x0_outs) == self.nl, f"cv2x0_outs length {len(cv2x0_outs)} != nl {self.nl}"
        assert len(cv2x1_outs) == self.nl, f"cv2x1_outs length {len(cv2x1_outs)} != nl {self.nl}"
        assert len(cv3x0_dw_outs) == self.nl, f"cv3x0_dw_outs length {len(cv3x0_dw_outs)} != nl {self.nl}"
        assert len(cv3x0_pw_outs) == self.nl, f"cv3x0_pw_outs length {len(cv3x0_pw_outs)} != nl {self.nl}"
        assert len(cv3x1_dw_outs) == self.nl, f"cv3x1_dw_outs length {len(cv3x1_dw_outs)} != nl {self.nl}"
        assert len(cv3x1_pw_outs) == self.nl, f"cv3x1_pw_outs length {len(cv3x1_pw_outs)} != nl {self.nl}"

        # ═══════════════════════════════════════
        # cv2 - Bbox branch (one2many)
        # ═══════════════════════════════════════
        self.cv2 = nn.ModuleList()
        for i in range(self.nl):
            self.cv2.append(
                nn.Sequential(
                    Conv(ch[i], cv2x0_outs[i], 3),  # Layer 0
                    Conv(cv2x0_outs[i], cv2x1_outs[i], 3),  # Layer 1
                    nn.Conv2d(cv2x1_outs[i], 4 * reg_max, 1),  # Final
                )
            )

        # ═══════════════════════════════════════
        # cv3 - Class branch (one2many)
        # ═══════════════════════════════════════
        self.cv3 = nn.ModuleList()
        for i in range(self.nl):
            self.cv3.append(
                nn.Sequential(
                    # Layer 0: DWConv → Conv
                    nn.Sequential(DWConv(ch[i], cv3x0_dw_outs[i], 3), Conv(cv3x0_dw_outs[i], cv3x0_pw_outs[i], 1)),
                    # Layer 1: DWConv → Conv
                    nn.Sequential(
                        DWConv(cv3x0_pw_outs[i], cv3x1_dw_outs[i], 3), Conv(cv3x1_dw_outs[i], cv3x1_pw_outs[i], 1)
                    ),
                    # Final: Conv2d
                    nn.Conv2d(cv3x1_pw_outs[i], nc, 1),
                )
            )

        # DFL layer
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        # ═══════════════════════════════════════
        # one2one branches (nếu có)
        # ═══════════════════════════════════════
        if one2one_cv2x0_outs is not None:
            # Validate one2one inputs
            assert len(one2one_cv2x0_outs) == self.nl
            assert len(one2one_cv2x1_outs) == self.nl
            assert len(one2one_cv3x0_dw_outs) == self.nl
            assert len(one2one_cv3x0_pw_outs) == self.nl
            assert len(one2one_cv3x1_dw_outs) == self.nl
            assert len(one2one_cv3x1_pw_outs) == self.nl

            # Build one2one_cv2
            self.one2one_cv2 = nn.ModuleList()
            for i in range(self.nl):
                self.one2one_cv2.append(
                    nn.Sequential(
                        Conv(ch[i], one2one_cv2x0_outs[i], 3),
                        Conv(one2one_cv2x0_outs[i], one2one_cv2x1_outs[i], 3),
                        nn.Conv2d(one2one_cv2x1_outs[i], 4 * reg_max, 1),
                    )
                )

            # Build one2one_cv3
            self.one2one_cv3 = nn.ModuleList()
            for i in range(self.nl):
                self.one2one_cv3.append(
                    nn.Sequential(
                        nn.Sequential(
                            DWConv(ch[i], one2one_cv3x0_dw_outs[i], 3),
                            Conv(one2one_cv3x0_dw_outs[i], one2one_cv3x0_pw_outs[i], 1),
                        ),
                        nn.Sequential(
                            DWConv(one2one_cv3x0_pw_outs[i], one2one_cv3x1_dw_outs[i], 3),
                            Conv(one2one_cv3x1_dw_outs[i], one2one_cv3x1_pw_outs[i], 1),
                        ),
                        nn.Conv2d(one2one_cv3x1_pw_outs[i], nc, 1),
                    )
                )

    @property
    def one2many(self):
        """Returns the one-to-many head components."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one branches."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one_cv2")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module = None, cls_head: nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities."""
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        from ultralytics.utils.tal import make_anchors

        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        from ultralytics.utils.tal import dist2bbox

        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end and not self.xyxy, dim=1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        import math

        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions."""
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores."""
        batch_size, anchors, nc = scores.shape
        k = max_det if self.export else min(max_det, anchors)

        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices

        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None
