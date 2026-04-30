import torch
import torch.nn as nn

class TASD(nn.Module):

    def __init__(self, in_channels=128, code_channels=42, num_groups=8):
        super().__init__()

        self.code_ch = code_channels

        # shared projections (xs3 → code)
        self.head_et  = self._make_head(in_channels, code_channels, num_groups)
        self.head_ed  = self._make_head(in_channels, code_channels, num_groups)
        self.head_ncr = self._make_head(in_channels, code_channels, num_groups)

        # project shared code to match private channels
        self.proj = nn.Conv3d(code_channels, in_channels, kernel_size=1, bias=False)

        # gating (per branch)
        self.gate_et  = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gate_ed  = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gate_ncr = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def _make_head(in_ch, out_ch, num_groups):
        g = min(num_groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, xa3, xb3, xs3):
        """
        xa3: private stream A (T1/T1ce)
        xb3: private stream B (T2/FLAIR)
        xs3: shared stream
        """

        # shared codes (42ch)
        s_et  = self.head_et(xs3)
        s_ed  = self.head_ed(xs3)
        s_ncr = self.head_ncr(xs3)

        # project to 128ch
        s_et  = self.proj(s_et)
        s_ed  = self.proj(s_ed)
        s_ncr = self.proj(s_ncr)

        # gated fusion (VERY IMPORTANT)
        g_et  = torch.sigmoid(self.gate_et(xa3))
        g_ed  = torch.sigmoid(self.gate_ed(xb3))
        g_ncr = torch.sigmoid(self.gate_ncr(xa3))

        z_et  = xa3 + g_et  * s_et
        z_ed  = xb3 + g_ed  * s_ed
        z_ncr = xa3 + g_ncr * s_ncr

        return z_et, z_ed, z_ncr


class TASD_1(nn.Module):
    """
    Ablation T1 — Additive Fusion, No Gating
    ─────────────────────────────────────────────────────────────────────────
    Removes the sigmoid gate entirely. The shared task code is added directly
    to the private stream with no per-channel modulation:

        z_et  = xa3 + s_et
        z_ed  = xb3 + s_ed
        z_ncr = xa3 + s_ncr

    Everything else is identical to TASD: three per-task heads extract shared
    codes from xs3, proj maps 42 → 128 ch, and the same private/shared stream
    routing is preserved.

    What this ablation isolates
    ────────────────────────────
    If TASD_1 ≈ full TASD  →  the sigmoid gate adds no value; plain additive
                               injection of the shared code is sufficient.
    If TASD_1 < full TASD  →  the gate is doing meaningful channel selection,
                               either suppressing noise or amplifying relevant
                               task-specific channels.

    Compared to TASD_2: TASD_1 has NO gate at all; TASD_2 has a gate but
    sourced from xs3 instead of xa3/xb3. The gap (T1 → T2 → full) cleanly
    attributes gains to (a) having a gate and (b) where the gate comes from.

    Differences from original TASD
    ─────────────────────────────────
    • gate_et / gate_ed / gate_ncr removed
    • sigmoid(gate(x_private)) * s  replaced by plain  s
    • proj + task heads unchanged
    """

    def __init__(self, in_channels=128, code_channels=42, num_groups=8):
        super().__init__()
        self.code_ch = code_channels

        self.head_et = self._make_head(in_channels, code_channels, num_groups)
        self.head_ed = self._make_head(in_channels, code_channels, num_groups)
        self.head_ncr = self._make_head(in_channels, code_channels, num_groups)

        self.proj = nn.Conv3d(code_channels, in_channels, kernel_size=1, bias=False)
        # No gate convolutions

    @staticmethod
    def _make_head(in_ch, out_ch, num_groups):
        g = min(num_groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, xa3, xb3, xs3):
        # shared codes projected to 128 ch
        s_et = self.proj(self.head_et(xs3))
        s_ed = self.proj(self.head_ed(xs3))
        s_ncr = self.proj(self.head_ncr(xs3))

        # Direct additive fusion — NO sigmoid gate
        z_et = xa3 + s_et
        z_ed = xb3 + s_ed
        z_ncr = xa3 + s_ncr

        return z_et, z_ed, z_ncr


# ══════════════════════════════════════════════════════════════════════════════
# TASD_2  —  T3: Symmetric Shared Gating
# ══════════════════════════════════════════════════════════════════════════════

class TASD_2(nn.Module):
    """
    Ablation T3 — Symmetric Shared-Stream Gating
    ─────────────────────────────────────────────────────────────────────────
    Keeps gated fusion but changes the gate SOURCE from private streams
    (xa3 / xb3) to the shared stream (xs3) for all three tasks:

        g_et  = sigmoid(gate_et (xs3))      # was sigmoid(gate_et (xa3))
        g_ed  = sigmoid(gate_ed (xs3))      # was sigmoid(gate_ed (xb3))
        g_ncr = sigmoid(gate_ncr(xs3))      # was sigmoid(gate_ncr(xa3))

        z_et  = xa3 + g_et  * s_et
        z_ed  = xb3 + g_ed  * s_ed
        z_ncr = xa3 + g_ncr * s_ncr

    The gate now modulates how much of the shared code enters each private
    stream — but using shared context to compute that modulation, rather than
    private context. This is a symmetric design: xs3 both generates the code
    AND controls its injection weight.

    What this ablation isolates
    ────────────────────────────
    Compared to TASD_1  : adds a gate         — does ANY gate help vs none?
    Compared to full TASD: changes gate source — does the gate need to "see"
                           the private stream (xa3/xb3) to be effective, or
                           is a shared-stream gate equally good?

    The clean 3-way paper story:
        TASD_1 (T1) : no gate at all             ← simplest
        TASD_2 (T3) : gate from shared stream    ← mid
        TASD   (full): gate from private stream  ← full

    Differences from original TASD
    ─────────────────────────────────
    • gate_et / gate_ed / gate_ncr take xs3 as input (not xa3 / xb3)
    • Everything else — heads, proj, z formula — is unchanged
    """

    def __init__(self, in_channels=128, code_channels=42, num_groups=8):
        super().__init__()
        self.code_ch = code_channels

        self.head_et = self._make_head(in_channels, code_channels, num_groups)
        self.head_ed = self._make_head(in_channels, code_channels, num_groups)
        self.head_ncr = self._make_head(in_channels, code_channels, num_groups)

        self.proj = nn.Conv3d(code_channels, in_channels, kernel_size=1, bias=False)

        # Gates still exist but will receive xs3 instead of xa3/xb3
        self.gate_et = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gate_ed = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gate_ncr = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def _make_head(in_ch, out_ch, num_groups):
        g = min(num_groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, xa3, xb3, xs3):
        # shared codes projected to 128 ch (identical to original)
        s_et = self.proj(self.head_et(xs3))
        s_ed = self.proj(self.head_ed(xs3))
        s_ncr = self.proj(self.head_ncr(xs3))

        # Gate sourced from xs3 for ALL tasks  ← key difference
        g_et = torch.sigmoid(self.gate_et(xs3))
        g_ed = torch.sigmoid(self.gate_ed(xs3))
        g_ncr = torch.sigmoid(self.gate_ncr(xs3))

        # Private stream + gated shared code (same formula as original)
        z_et = xa3 + g_et * s_et
        z_ed = xb3 + g_ed * s_ed
        z_ncr = xa3 + g_ncr * s_ncr

        return z_et, z_ed, z_ncr