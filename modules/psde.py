import torch
import torch.nn as nn


class PSDE(nn.Module):

    """
     Private-to-Shared Disentanglement Enhancement
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        # project private features into shared channel space
        self.proj_a = nn.Conv3d(channels, channels, 1, bias=False)
        self.proj_b = nn.Conv3d(channels, channels, 1, bias=False)

        # channel attention (SE-style) on each private residual
        self.ca_a = self._make_ca(channels, reduction)
        self.ca_b = self._make_ca(channels, reduction)

        # learnable injection scalars — small init so shared dominates early
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

        # refinement conv after injection
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True))

    @staticmethod
    def _make_ca(channels, reduction):
        """Lightweight channel attention: GAP → FC → ReLU → FC → Sigmoid."""
        mid = max(channels // reduction, 4)
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid())

    def forward(self, f_a, f_s, f_b):

        # step 1: project private into shared space
        p_a = self.proj_a(f_a)  # (B, C, H, W, D)
        p_b = self.proj_b(f_b)

        # step 2: extract private-exclusive residuals
        r_a = f_a - p_a  # what private A has beyond shared projection
        r_b = f_b - p_b

        # step 3: channel attention on residuals
        # ca output shape: (B, C) → reshape to (B, C, 1, 1, 1) for broadcast
        w_a = self.ca_a(r_a).view(-1, r_a.shape[1], 1, 1, 1)
        w_b = self.ca_b(r_b).view(-1, r_b.shape[1], 1, 1, 1)
        r_a = r_a * w_a
        r_b = r_b * w_b

        # step 4: inject into shared with learnable scalars
        f_s_enriched = f_s + self.alpha * r_a + self.beta * r_b

        # step 5: refinement
        f_s_enriched = self.refine(f_s_enriched)


        return f_s_enriched


# ══════════════════════════════════════════════════════════════════════════════
# PSDE_1  —  P2: Direct Addition  (no residual, no channel attention)
# ══════════════════════════════════════════════════════════════════════════════

class PSDE_1(nn.Module):
    """
    Ablation P2 — Direct Addition
    ──────────────────────────────────────────────────────────────────────────
    Simplest possible alternative to PSDE: skip the projection/residual
    disentanglement and the channel-attention entirely.  Raw private features
    f_a and f_b are injected directly into the shared stream f_s with a pair
    of learnable scalar weights (same init as original PSDE).

        f_s_enriched = f_s + alpha * f_a + beta * f_b   → refine

    What this ablation isolates
    ───────────────────────────
    If PSDE_1 ≈ full PSDE  →  the residual disentanglement AND channel
    attention add no value; raw injection is sufficient.
    If PSDE_1 < full PSDE  →  at least one of those two mechanisms matters
    (use PSDE_2 to split them further).

    Differences from original PSDE
    ────────────────────────────────
    • No proj_a / proj_b           (no projection into shared space)
    • No residual extraction       (r = f - proj(f) removed)
    • No channel-attention (ca_a / ca_b)
    • Injects f_a / f_b directly   (not attention-weighted residuals)
    • Refinement conv kept         (fair comparison — same param count there)
    """

    def __init__(self, channels, reduction=4):  # reduction kept for API compat
        super().__init__()

        # learnable injection scalars — same init as original
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

        # refinement conv (kept identical to original)
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True))

    def forward(self, f_a, f_s, f_b):
        # Direct injection — no projection, no residual, no attention
        f_s_enriched = f_s + self.alpha * f_a + self.beta * f_b
        f_s_enriched = self.refine(f_s_enriched)
        return f_s_enriched


# ══════════════════════════════════════════════════════════════════════════════
# PSDE_2  —  P3: Residual Projection Only  (no channel attention)
# ══════════════════════════════════════════════════════════════════════════════

class PSDE_2(nn.Module):
    """
    Ablation P3 — Projection + Residual, No Channel Attention
    ──────────────────────────────────────────────────────────────────────────
    Keeps the full residual-disentanglement step from the original PSDE
    (project → subtract → residual) but removes the channel-attention (SE)
    weights applied to those residuals before injection.

        r_a = f_a - proj_a(f_a)          # private-exclusive residual
        r_b = f_b - proj_b(f_b)
        f_s_enriched = f_s + alpha * r_a + beta * r_b   → refine

    What this ablation isolates
    ───────────────────────────
    Compared to PSDE_1:   adds residual disentanglement  — does subtracting
                          the shared projection help?
    Compared to full PSDE: removes channel attention      — is the CA on top
                           of the residual actually necessary?

    The clean 3-way comparison becomes:
        PSDE_1  (P2) : raw injection        ← simplest
        PSDE_2  (P3) : residual only        ← mid
        PSDE    (full): residual + CA       ← full

    Differences from original PSDE
    ────────────────────────────────
    • proj_a / proj_b kept         (projection into shared space)
    • Residual extraction kept     (r = f - proj(f))
    • No channel-attention (ca_a / ca_b)  ← only change
    • Injects raw (unweighted) residuals  (not CA-weighted)
    • Refinement conv kept
    """

    def __init__(self, channels, reduction=4):  # reduction kept for API compat
        super().__init__()

        # projection convolutions (same as original)
        self.proj_a = nn.Conv3d(channels, channels, 1, bias=False)
        self.proj_b = nn.Conv3d(channels, channels, 1, bias=False)

        # learnable injection scalars — same init as original
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

        # refinement conv (identical to original)
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True))

    def forward(self, f_a, f_s, f_b):
        # Step 1: project private into shared space
        p_a = self.proj_a(f_a)
        p_b = self.proj_b(f_b)

        # Step 2: extract private-exclusive residuals
        r_a = f_a - p_a
        r_b = f_b - p_b

        # Step 3: inject residuals directly — NO channel attention
        f_s_enriched = f_s + self.alpha * r_a + self.beta * r_b
        f_s_enriched = self.refine(f_s_enriched)
        return f_s_enriched
