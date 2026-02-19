import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import PSABlock  # ← THÊM DÒNG NÀY!

__all__ = (
    'BottleneckPruned',
    'C3kPruned',
    'C3k2Pruned',
    'C3k2PrunedAttn',
    'SPPFPruned',
    'C2PSAPruned',
)


class BottleneckPruned(nn.Module):
    """
    Pruned Bottleneck block.

    Khác với Bottleneck gốc, class này nhận channel sizes tuyệt đối
    thay vì tính từ expansion ratio.

    Args:
        cv1in (int): Input channels cho cv1
        cv1out (int): Output channels của cv1
        cv2out (int): Output channels của cv2 (cũng là output của block)
        shortcut (bool): Có sử dụng residual connection không
        g (int): Groups cho convolution
        k (tuple): Kernel sizes cho (cv1, cv2)
        e (float): Expansion ratio (không dùng trong pruned version, giữ lại để tương thích)
    """

    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, k[0], 1)
        self.cv2 = Conv(cv1out, cv2out, k[1], 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        """Forward pass với optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3kPruned(nn.Module):
    """
    Pruned C3k block - CSP Bottleneck with customizable kernel sizes.

    C3k structure (kế thừa từ C3):
        Input ┬→ cv1(c1→c_) → Sequential Bottleneck(c_→c_) → m_out ┐
              └→ cv2(c1→c_) ────────────────────────────────────────→ cv2_out
                                                                      ↓
                                                          concat(2*c_) → cv3(2*c_→c2)

    Args:
        cv1in (int): Input channels
        cv1out (int): Output channels của cv1 (= c_)
        cv2out (int): Output channels của cv2 (= c_)
        cv3out (int): Output channels của cv3 (= c2, final output)
        bottleneck_cv1outs (list[int]): cv1 output cho mỗi bottleneck
        bottleneck_cv2outs (list[int]): cv2 output cho mỗi bottleneck
        n (int): Number of Bottleneck blocks
        shortcut (bool): Shortcut cho các bottleneck bên trong
        g (int): Groups
        k (int): Kernel size cho bottleneck

    Notes:
        - Mỗi bottleneck có thể có channels khác nhau sau khi prune
        - Bottleneck chain: bn[0] nhận cv1out, bn[i] nhận output của bn[i-1]
    """

    def __init__(self, cv1in, cv1out, cv2out, cv3out,
                 bottleneck_cv1outs, bottleneck_cv2outs,
                 n=1, shortcut=True, g=1, k=3):
        super().__init__()

        # QUAN TRỌNG: Thứ tự phải khớp với C3 gốc: cv1 → cv2 → cv3 → m

        # Parallel branches
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)

        # Concat và final conv - KHAI BÁO TRƯỚC m
        # cv3 input = cv1out + cv2out (concat 2 branches)
        self.cv3 = Conv(cv1out + cv2out, cv3out, 1, 1)

        # Bottleneck sequence với pruned channels
        bottlenecks = []
        for i in range(n):
            if i == 0:
                bn_input = cv1out
            else:
                bn_input = bottleneck_cv2outs[i - 1]

            bn_cv1out = bottleneck_cv1outs[i]
            bn_cv2out = bottleneck_cv2outs[i]

            bottlenecks.append(
                BottleneckPruned(bn_input, bn_cv1out, bn_cv2out,
                                 shortcut, g, k=(k, k), e=1.0)
            )

        self.m = nn.Sequential(*bottlenecks)

    def forward(self, x):
        """Forward pass qua C3k structure."""
        # Parallel branches
        y1 = self.cv1(x)
        y2 = self.cv2(x)

        # Sequential bottlenecks (output = input vì add=True)
        y1 = self.m(y1)

        # Concat và final conv
        return self.cv3(torch.cat((y1, y2), 1))


class C3k2Pruned(nn.Module):
    """
    Pruned C3k2 block - Faster CSP implementation với C3k module.

    C3k2 structure (kế thừa từ C2f, chỉ khác m = C3k):
        Input → cv1 → chunk(2) → [left_half, right_half]
                         ↓              ↓
                       list      → m[0] (C3k) → list
                                         ↓
                                       m[1] (C3k) → ...
                                         ↓
                             concat all → cv2 → output

    Args:
        cv1in (int): Input channels
        cv1out (int): Output channels của cv1
        cv1_split_sections (tuple): (left_half_channels, right_half_channels) sau split
        c3k_cv1outs (list[int]): cv1 output của mỗi C3k module
        c3k_cv2outs (list[int]): cv2 output của mỗi C3k module
        c3k_cv3outs (list[int]): cv3 output (final) của mỗi C3k module
        c3k_bottleneck_cv1outs (list[list[int]]): bottleneck cv1 outputs cho mỗi C3k
        c3k_bottleneck_cv2outs (list[list[int]]): bottleneck cv2 outputs cho mỗi C3k
        c3k_n_bottlenecks (list[int]): số bottleneck cho mỗi C3k
        cv2out (int): Output channels của cv2 (final output)
        n (int): Number of C3k modules
        n_bottlenecks (int): Number of bottlenecks bên trong mỗi C3k (không dùng, giữ lại cho tương thích)
        shortcut (bool): Shortcut cho bottleneck trong C3k
        g (int): Groups
        k (int): Kernel size cho bottleneck trong C3k

    Notes:
        - Giống C2fPruned structure
        - m = C3kPruned thay vì BottleneckPruned
        - C3k chain: C3k[0] nhận right_half, C3k[i] nhận output của C3k[i-1]
        - Mỗi C3k có số bottleneck và channels riêng
    """

    def __init__(self, cv1in, cv1out, cv1_split_sections,
                 c3k_cv1outs, c3k_cv2outs, c3k_cv3outs,
                 c3k_bottleneck_cv1outs, c3k_bottleneck_cv2outs, c3k_n_bottlenecks,
                 cv2out, n=1, n_bottlenecks=2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()

        self.cv1_split_sections = cv1_split_sections

        # QUAN TRỌNG: Thứ tự khai báo phải KHỚP với original C2f
        # Original C2f order: cv1 → cv2 → m

        # First conv
        self.cv1 = Conv(cv1in, cv1out, 1, 1)

        # Final conv - KHAI BÁO TRƯỚC m để khớp thứ tự với original!
        cv2_input = cv1out + sum(c3k_cv3outs)
        self.cv2 = Conv(cv2_input, cv2out, 1, 1)

        # C3k modules - chain structure
        self.m = nn.ModuleList()
        for i in range(n):
            # C3k đầu tiên nhận right_half từ split
            if i == 0:
                c3k_input = cv1_split_sections[1]
            else:
                # C3k tiếp theo nhận output của C3k trước
                c3k_input = c3k_cv3outs[i - 1]

            self.m.append(
                C3kPruned(
                    cv1in=c3k_input,
                    cv1out=c3k_cv1outs[i],
                    cv2out=c3k_cv2outs[i],
                    cv3out=c3k_cv3outs[i],
                    bottleneck_cv1outs=c3k_bottleneck_cv1outs[i],
                    bottleneck_cv2outs=c3k_bottleneck_cv2outs[i],
                    n=c3k_n_bottlenecks[i],
                    shortcut=shortcut,
                    g=g,
                    k=k
                )
            )

    def forward(self, x):
        """
        Forward pass qua C3k2 structure.

        Flow:
            x → cv1 → split → [left, right]
            right → C3k[0] → out0
            out0 → C3k[1] → out1
            ...
            concat(left, right, out0, out1, ...) → cv2 → output
        """
        # cv1 forward và split
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))

        # C3k chain
        y.extend(m(y[-1]) for m in self.m)

        # Concat và final conv
        return self.cv2(torch.cat(y, 1))


class C3k2PrunedAttn(nn.Module):
    """
    Pruned C3k2 với attention blocks (attn=True variant).

    Dành cho C3k2 có m = Sequential(Bottleneck, PSABlock) - layer 22 trong YOLOv26.
    PSABlock KHÔNG được prune - cv1.bn của layer này nằm trong ignore_bn_list.

    C3k2 forward (kế thừa C2f):
        Input → cv1 → split → [left(256), right(256)]
        right → Sequential(BottleneckPruned, PSABlock) → out0(256)
        concat(left, right, out0) → cv2 → output

    Args:
        cv1in (int): Input channels
        cv1out (int): Output channels của cv1 (KHÔNG prune, trong ignore_bn_list)
        cv1_split_sections (list): [left, right] - cả 2 cố định (= cv1out//2)
        n_blocks (int): Số lượng Sequential(Bottleneck, PSABlock) blocks
        bottleneck_cv1outs (list[int]): cv1 output (pruned) của mỗi Bottleneck
        cv2out (int): Output channels của cv2 (final output, CÓ THỂ prune)
        shortcut (bool): Shortcut cho Bottleneck bên trong
        g (int): Groups

    Notes:
        - cv1.bn nằm trong ignore_bn_list → right_half cố định = original
        - Bottleneck cv2.bn nằm trong ignore_bn_list (add=True) → cv2out = right_half
        - PSABlock BNs nằm trong ignore_bn_list → không prune
        - cv2_input = cv1out + right_half * n_blocks
    """

    def __init__(self, cv1in, cv1out, cv1_split_sections, n_blocks,
                 bottleneck_cv1outs, cv2out, shortcut=True, g=1):
        super().__init__()

        self.cv1_split_sections = cv1_split_sections
        right_half = cv1_split_sections[1]  # Cố định (không prune)

        # cv1: không prune (cv1.bn trong ignore_bn_list)
        # Khai báo theo thứ tự gốc C2f: cv1 → cv2 → m
        self.cv1 = Conv(cv1in, cv1out, 1, 1)

        # cv2_input = left + right + PSABlock_out * n_blocks
        #           = cv1out + right_half * n_blocks
        cv2_input = cv1out + right_half * n_blocks
        self.cv2 = Conv(cv2_input, cv2out, 1, 1)

        # Sequential(BottleneckPruned, PSABlock) - PSABlock dùng right_half cố định
        self.m = nn.ModuleList()
        for i in range(n_blocks):
            bottleneck = BottleneckPruned(
                cv1in=right_half,
                cv1out=bottleneck_cv1outs[i],
                cv2out=right_half,       # Không prune (cv2.bn ignore, add=True)
                shortcut=shortcut, g=g, k=(3, 3), e=1.0
            )
            psa = PSABlock(
                c=right_half,
                attn_ratio=0.5,
                num_heads=max(right_half // 64, 1)
            )
            self.m.append(nn.Sequential(bottleneck, psa))

    def forward(self, x):
        """Forward: cv1 → split → [left, right]; right qua m blocks; concat → cv2."""
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPFPruned(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer - Pruned version.

    Tái sử dụng từ YOLOv8 với một số điều chỉnh cho YOLOv26:
    - Thêm param n (number of pooling iterations)
    - Thêm param shortcut (residual connection)
    - cv1 không có activation (act=False)

    Args:
        cv1in (int): Input channels
        cv1out (int): Hidden channels (sau cv1)
        cv2out (int): Output channels
        k (int): Kernel size cho max pooling
        n (int): Number of pooling iterations
        shortcut (bool): Whether to use shortcut connection
    """

    def __init__(self, cv1in, cv1out, cv2out, k=5, n=3, shortcut=False):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1, act=False)  # NO activation!
        # cv2 input = cv1out * (n+1) vì concat [x, y1, y2, y3, ...]
        self.cv2 = Conv(cv1out * (n + 1), cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        """Apply sequential pooling operations and return concatenated result."""
        y = [self.cv1(x)]
        # Sequential pooling n times
        y.extend(self.m(y[-1]) for _ in range(self.n))

        # Concat và final conv
        y = self.cv2(torch.cat(y, 1))

        # Optional shortcut
        return y + x if self.add else y


class C2PSAPruned(nn.Module):
    """
    Pruned C2PSA - C2 module với PSA attention blocks.

    [... docstring ...]
    """

    def __init__(self, cv1in, cv1out, cv1_split_sections, cv2out, n=1, e=0.5):
        super().__init__()

        # Validations
        assert len(cv1_split_sections) == 2, "Must have [left, right]"
        assert sum(cv1_split_sections) == cv1out, \
            f"Split sum {sum(cv1_split_sections)} != cv1out {cv1out}"
        assert cv1_split_sections[1] % 2 == 0, \
            f"Right half {cv1_split_sections[1]} must be even"

        self.cv1_split_sections = cv1_split_sections
        self.cv1 = Conv(cv1in, cv1out, 1, 1)

        # QUAN TRỌNG: cv2 phải khai báo TRƯỚC m để khớp thứ tự với C2PSA gốc
        self.cv2 = Conv(cv1out, cv2out, 1)

        # ═══ TÌM num_heads HỢP LỆ ═══
        c_psa = cv1_split_sections[1]
        target_heads = max(c_psa // 64, 1)

        # Tìm ước trong range [1, 32]
        valid_heads = [h for h in range(1, min(33, c_psa + 1))
                       if c_psa % h == 0]

        if not valid_heads:
            num_heads = 1
        else:
            num_heads = min(valid_heads,
                            key=lambda x: abs(x - target_heads))

        # Validation
        assert c_psa % num_heads == 0, \
            f"FATAL: c_psa={c_psa} not divisible by num_heads={num_heads}"
        # ════════════════════════════

        # PSABlock modules
        self.m = nn.Sequential(*[
            PSABlock(
                c=c_psa,
                attn_ratio=0.5,
                num_heads=num_heads,
                shortcut=True
            )
            for _ in range(n)
        ])

    # ═══ THÊM PHẦN NÀY ═══
    def forward(self, x):
        """
        Forward pass qua C2PSA structure.

        Args:
            x (torch.Tensor): Input [B, cv1in, H, W]

        Returns:
            torch.Tensor: Output [B, cv2out, H, W]
        """
        # cv1
        y = self.cv1(x)

        # Split
        a, b = y.split(self.cv1_split_sections, dim=1)

        # PSABlock chain
        b = self.m(b)

        # Concat và cv2
        return self.cv2(torch.cat([a, b], 1))
    # ═════════════════════