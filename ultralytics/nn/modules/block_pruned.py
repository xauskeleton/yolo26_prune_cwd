import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import PSABlock  # ← THÊM DÒNG NÀY!

__all__ = (
    'BottleneckPruned',
    'C3kPruned',
    'C3k2Pruned',
    'SPPFPruned',
    'C2PSAPruned'  # ← THÊM DÒNG NÀY!
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
        n (int): Number of Bottleneck blocks
        shortcut (bool): Shortcut cho các bottleneck bên trong
        g (int): Groups
        k (int): Kernel size cho bottleneck

    Notes:
        - Bottleneck bên trong có e=1.0 và shortcut=True → add=True
        - Bottleneck với add=True: cv1.bn và cv2.bn vào ignore_bn_list (KHÔNG prune)
        - Vậy tất cả bottleneck đều: c_ → c_ → c_ (channels không đổi)
        - Chỉ prune: cv1.bn, cv2.bn, cv3.bn của C3k chính
    """

    def __init__(self, cv1in, cv1out, cv2out, cv3out, n=1, shortcut=True, g=1, k=3):
        super().__init__()

        # Parallel branches (cùng output channels)
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)

        # Bottleneck sequence
        # Tất cả bottleneck đều: cv1out → cv1out (vì add=True)
        # Inner bottleneck KHÔNG được prune (ignore_bn_list)
        self.m = nn.Sequential(
            *(BottleneckPruned(cv1out, cv1out, cv1out, shortcut, g, k=(k, k), e=1.0)
              for _ in range(n))
        )

        # Concat và final conv
        # cv3 input = cv1out + cv2out (concat 2 branches)
        self.cv3 = Conv(cv1out + cv2out, cv3out, 1, 1)

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
        cv2out (int): Output channels của cv2 (final output)
        n (int): Number of C3k modules
        n_bottlenecks (int): Number of bottlenecks bên trong mỗi C3k
        shortcut (bool): Shortcut cho bottleneck trong C3k
        g (int): Groups
        k (int): Kernel size cho bottleneck trong C3k

    Notes:
        - Giống C2fPruned structure
        - m = C3kPruned thay vì BottleneckPruned
        - C3k chain: C3k[0] nhận right_half, C3k[i] nhận output của C3k[i-1]
    """

    def __init__(self, cv1in, cv1out, cv1_split_sections,
                 c3k_cv1outs, c3k_cv2outs, c3k_cv3outs,
                 cv2out, n=1, n_bottlenecks=2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()

        self.cv1_split_sections = cv1_split_sections

        # First conv
        self.cv1 = Conv(cv1in, cv1out, 1, 1)

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
                    n=n_bottlenecks,
                    shortcut=shortcut,
                    g=g,
                    k=k
                )
            )

        # Final conv
        # Forward flow: y = [left, right, m[0](right), m[1](m[0]), ...]
        # cv2 input = concat all y = left + right + all C3k outputs
        #
        # Trong pruned version:
        # - left = cv1_split_sections[0]
        # - right = cv1_split_sections[1]
        # - m[i] output = c3k_cv3outs[i]
        # Total = cv1_split_sections[0] + cv1_split_sections[1] + sum(c3k_cv3outs)
        #       = cv1out + sum(c3k_cv3outs)
        cv2_input = cv1out + sum(c3k_cv3outs)
        self.cv2 = Conv(cv2_input, cv2out, 1, 1)

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

    Chiến lược:
        - cv1: CÓ THỂ cắt cả input và output channels
        - Split: KHÔNG đối xứng sau pruning
        - PSABlock: GIỮ NGUYÊN (dùng class gốc từ ultralytics)
        - cv2: CÓ THỂ cắt output channels

    C2PSA structure:
        Input → cv1 → split([a, b])
                  a → unchanged
                  b → PSABlock[0] → PSABlock[1] → ... → b'
                concat(a, b') → cv2 → output

    Args:
        cv1in (int): Input channels (từ SPPF đã cắt)
        cv1out (int): Output channels của cv1 (từ mask)
        cv1_split_sections (tuple): (left_channels, right_channels) sau split
                                    VD: [512, 512] hoặc [480, 544] nếu prune không đều
        cv2out (int): Output channels của cv2 (từ mask)
        n (int): Number of PSABlock modules
        e (float): Expansion ratio (không dùng, giữ để tương thích)

    Notes:
        - cv1_split_sections[1] (right half) phải là số CHẴN vì PSABlock cần
        - PSABlock có residual → input = output = cv1_split_sections[1]
        - cv2 input = cv1out (concat a + b)

    Example:
        Model gốc:
            C2PSA(1024, 1024, n=2)
            cv1: 1024 → 2048, split: [1024, 1024]
            cv2: 2048 → 1024

        Sau prune:
            C2PSAPruned(512, 1024, [512, 512], 512, n=2)
            cv1: 512 → 1024, split: [512, 512]
            cv2: 1024 → 512
    """

    def __init__(self, cv1in, cv1out, cv1_split_sections, cv2out, n=1, e=0.5):
        super().__init__()

        # Kiểm tra split sections
        assert len(cv1_split_sections) == 2, "cv1_split_sections must be [left, right]"
        assert sum(cv1_split_sections) == cv1out, \
            f"Sum of split_sections {sum(cv1_split_sections)} must equal cv1out {cv1out}"

        # Right half phải chẵn (PSABlock requirement)
        assert cv1_split_sections[1] % 2 == 0, \
            f"Right half {cv1_split_sections[1]} must be even for PSABlock"

        self.cv1_split_sections = cv1_split_sections

        # cv1: Conv đầu tiên
        # Input có thể đã bị cắt (từ SPPF)
        # Output có thể bị cắt (từ mask)
        self.cv1 = Conv(cv1in, cv1out, 1, 1)

        # PSABlock modules
        # Input/Output = cv1_split_sections[1] (right half)
        # Dùng PSABlock GỐC từ Ultralytics
        c_psa = cv1_split_sections[1]
        self.m = nn.Sequential(*[
            PSABlock(
                c=c_psa,
                attn_ratio=0.5,
                num_heads=max(c_psa // 64, 1),
                shortcut=True
            )
            for _ in range(n)
        ])

        # cv2: Final conv
        # Input = cv1out (concat left + right)
        # Output có thể bị cắt (từ mask)
        self.cv2 = Conv(cv1out, cv2out, 1)

    def forward(self, x):
        """
        Forward pass qua C2PSA structure.

        Flow:
            x → cv1 → split([a, b])
            a: unchanged (left half)
            b: → PSABlock chain → b' (right half)
            concat(a, b') → cv2 → output

        Args:
            x (torch.Tensor): Input tensor [B, cv1in, H, W]

        Returns:
            torch.Tensor: Output tensor [B, cv2out, H, W]
        """
        # cv1 forward
        y = self.cv1(x)  # [B, cv1out, H, W]

        # Split thành 2 nửa (KHÔNG đối xứng sau pruning)
        a, b = y.split(self.cv1_split_sections, dim=1)
        # a: [B, left_channels, H, W]
        # b: [B, right_channels, H, W]

        # PSABlock chain (sequential, có residual)
        # b không đổi size vì PSABlock có residual
        b = self.m(b)  # [B, right_channels, H, W]

        # Concat và final conv
        out = self.cv2(torch.cat([a, b], 1))  # [B, cv2out, H, W]

        return out