"""
Ablation Models for BrainTumorSegNet
=====================================
All six ablation variants in one file.

    AblationV1  — Single Encoder / Single Decoder
                  No stem. Raw cat(4ch) → conv_block(4→16) → 32 → 64 → 128.
                  Dropout3d at enc2/enc3. Single decoder, one 3-ch head.

    AblationV2  — Four Encoders / Single Decoder
                  No stem. Each modality: conv_block(1→16) → 32 → 64 → 128.
                  Dropout3d at enc2/enc3 per branch. Fused at every scale.
                  Single shared decoder, one 3-ch head.

    AblationV3  — Three Encoders / Three Decoders  (no PSDE / TASD / Epi)
                  MSGS stems + DRB encoders + Dropout3d at enc2/enc3.
                  PSDE replaced by direct branch-S skips.
                  TASD replaced by cat(384) + three 1×1 projections.

    AblationV4  — V3 + PSDE  (no TASD / Epi)
                  Adds real PSDE. Dropout order: encode → PSDE → dropout → pool.

    AblationV4_1 — V4 with PSDE_1  (P2: Direct Addition)
                   No projection, no residual, no channel attention.

    AblationV4_2 — V4 with PSDE_2  (P3: Residual Only, no CA)
                   Projection + residual kept, channel attention removed.

    AblationV5  — V3 + TASD  (no PSDE / Epi)
                  Adds real TASD bottleneck task split.

    AblationV5_1 — V5 with TASD_1  (T1: No gating, additive only)
                   Shared code added directly, no sigmoid gate.

    AblationV5_2 — V5 with TASD_2  (T3: Symmetric shared-stream gating)
                   Gate exists but sourced from xs3 for all tasks.

    AblationV6  — V3 + PSDE + TASD  (no Epi)
                  Full architecture minus epistemic loss only.

All models:
    - Dropout3d at enc2/enc3 (same positions as full model) for MC inference
    - mc_forward()        — identical API to BrainTumorSegNet.mc_forward()
    - forward_mc_train()  — no-op wrapper → (outputs, None) for train compat

Dropout order (V3-V6, matches original model exactly):
    Level 1: encode → (PSDE if applicable) → pool          [no dropout]
    Level 2: encode → (PSDE if applicable) → dropout → pool
    Level 3: encode → (PSDE if applicable) → dropout → pool

Shared:
    AblationLoss — Dice + CE only. Drop-in for BrainTumorLoss.

Usage in train.py:
    from ablation_models import (AblationV1, AblationV2, AblationV3,
                                  AblationV4,  AblationV4_1, AblationV4_2,
                                  AblationV5,  AblationV5_1, AblationV5_2,
                                  AblationV6,  AblationLoss)
    model_dict = {
        'ablation_v1'  : AblationV1,
        'ablation_v2'  : AblationV2,
        'ablation_v3'  : AblationV3,
        'ablation_v4'  : AblationV4,
        'ablation_v4_1': AblationV4_1,
        'ablation_v4_2': AblationV4_2,
        'ablation_v5'  : AblationV5,
        'ablation_v5_1': AblationV5_1,
        'ablation_v5_2': AblationV5_2,
        'ablation_v6'  : AblationV6,
    }
    loss_dict = { k: AblationLoss for k in model_dict }
"""

import torch
import torch.nn as nn
from modules.blocks import MSGS, DRB
from modules.psde   import PSDE, PSDE_1, PSDE_2
from modules.tasd   import TASD, TASD_1, TASD_2


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv3d → GroupNorm(min(8,ch)) → ReLU layers."""
    g = min(8, out_ch)
    return nn.Sequential(
        nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.ReLU(inplace=True),
    )


def _fuse_proj(in_ch: int, out_ch: int) -> nn.Sequential:
    """1×1 conv to fuse multi-branch concatenated features."""
    g = min(8, out_ch)
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.ReLU(inplace=True),
    )


def _task_head() -> nn.Sequential:
    """1×1 bottleneck projection 384 → 128 ch (replaces TASD in V3/V4)."""
    return nn.Sequential(
        nn.Conv3d(384, 128, 1, bias=False),
        nn.GroupNorm(8, 128),
        nn.ReLU(inplace=True),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Shared MC-Dropout mixin
# ══════════════════════════════════════════════════════════════════════════════

class _MCMixin:
    """
    Provides mc_forward() for all ablation models.
    Identical API to BrainTumorSegNet.mc_forward():
        - sets only Dropout3d layers to train mode, keeps norm layers in eval
        - runs n_passes stochastic forward passes
        - returns mean predictions + per-voxel epistemic variance maps
    """

    n_passes: int  # must be set in subclass __init__

    def mc_forward(self, t1, t1ce, t2, flair, n_passes=None):
        n_passes = n_passes or self.n_passes

        # activate only Dropout3d, keep BN/GN in eval
        for m in self.modules():
            if isinstance(m, nn.Dropout3d):
                m.train()

        et_preds, ed_preds, ncr_preds = [], [], []

        with torch.no_grad():
            for _ in range(n_passes):
                out = self.forward(t1, t1ce, t2, flair)
                et_preds.append(torch.sigmoid(out['mu_et']))
                ed_preds.append(torch.sigmoid(out['mu_ed']))
                ncr_preds.append(torch.sigmoid(out['mu_ncr']))

        et_stack  = torch.stack(et_preds)     # (T, B, 1, H, W, D)
        ed_stack  = torch.stack(ed_preds)
        ncr_stack = torch.stack(ncr_preds)

        mu_et  = et_stack.mean(dim=0)
        mu_ed  = ed_stack.mean(dim=0)
        mu_ncr = ncr_stack.mean(dim=0)

        # epistemic uncertainty: (1/T) Σ (ŷ(t) - ȳ)²
        var_et  = ((et_stack  - mu_et.unsqueeze(0))  ** 2).mean(dim=0)
        var_ed  = ((ed_stack  - mu_ed.unsqueeze(0))  ** 2).mean(dim=0)
        var_ncr = ((ncr_stack - mu_ncr.unsqueeze(0)) ** 2).mean(dim=0)

        return {
            'mu_et'  : mu_et,
            'mu_ed'  : mu_ed,
            'mu_ncr' : mu_ncr,
            'var_et' : var_et,
            'var_ed' : var_ed,
            'var_ncr': var_ncr,
            'seg'    : torch.cat([mu_et, mu_ed, mu_ncr], dim=1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ablation V1 — Single Encoder / Single Decoder
# ══════════════════════════════════════════════════════════════════════════════

class AblationV1(_MCMixin, nn.Module):
    """
    Single-encoder / single-decoder U-Net. No stem.
    Dropout3d at enc2 and enc3 (same positions as full model).

    Input     : cat(T1, T1ce, T2, FLAIR) → 4 ch
    Enc1      : conv_block(4,  16)   full resolution           → s1
    Enc2      : conv_block(16, 32) → Dropout3d → MaxPool3d(2)  → s2 (dropped)
    Enc3      : conv_block(32, 64) → Dropout3d → MaxPool3d(2)  → s3 (dropped)
    Enc4      : conv_block(64, 128) → MaxPool3d(2)             → s4, z

    Decoder (skip connections use pre-dropout features for richer gradients):
        up0/dec0 : cat(128, s4=128) = 256 → 128
        up1/dec1 : cat(128, s3=64)  = 192 → 128
        up2/dec2 : cat(64,  s2=32)  = 96  → 64
        up3/final: cat(32,  s1=16)  = 48  → 16
        out      : 16 → 3
    """

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__()
        self.n_passes = n_passes

        self.enc1   = _conv_block(4,   16)
        self.pool1  = nn.MaxPool3d(2)

        self.enc2   = _conv_block(16,  32)
        self.drop2  = nn.Dropout3d(mc_dropout_p)
        self.pool2  = nn.MaxPool3d(2)

        self.enc3   = _conv_block(32,  64)
        self.drop3  = nn.Dropout3d(mc_dropout_p)
        self.pool3  = nn.MaxPool3d(2)

        self.enc4   = _conv_block(64,  128)
        self.pool4  = nn.MaxPool3d(2)

        self.up0    = nn.ConvTranspose3d(128, 128, 2, 2)
        self.dec0   = _conv_block(256, 128)
        self.up1    = nn.ConvTranspose3d(128, 128, 2, 2)
        self.dec1   = _conv_block(192, 128)
        self.up2    = nn.ConvTranspose3d(128, 64,  2, 2)
        self.dec2   = _conv_block(96,  64)
        self.up3    = nn.ConvTranspose3d(64,  32,  2, 2)
        self.final  = _conv_block(48,  16)
        self.out_conv = nn.Conv3d(16, 3, 1)

    def forward(self, t1, t1ce, t2, flair):
        x   = torch.cat([t1, t1ce, t2, flair], dim=1)

        s1  = self.enc1(x)                           # (B, 16,  H,    W,    D   )
        e2  = self.enc2(self.pool1(s1))
        d2  = self.drop2(e2)
        s2  = d2                                     # (B, 32,  H/2,  W/2,  D/2 ) — skip
        e3  = self.enc3(self.pool2(d2))
        d3  = self.drop3(e3)
        s3  = d3                                     # (B, 64,  H/4,  W/4,  D/4 )
        s4  = self.enc4(self.pool3(d3))              # (B, 128, H/8,  W/8,  D/8 )
        z   = self.pool4(s4)                         # (B, 128, H/16, W/16, D/16)

        x = self.dec0(torch.cat([self.up0(z),  s4], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x = self.final(torch.cat([self.up3(x), s1], dim=1))

        seg = self.out_conv(x)
        return {'mu_et': seg[:, 0:1], 'mu_ed': seg[:, 1:2],
                'mu_ncr': seg[:, 2:3], 'seg': seg}

    def forward_mc_train(self, t1, t1ce, t2, flair, T=3):
        return self.forward(t1, t1ce, t2, flair), None


# ══════════════════════════════════════════════════════════════════════════════
# Ablation V2 — Four Encoders / Single Decoder
# ══════════════════════════════════════════════════════════════════════════════

class _ModalityEncoder(nn.Module):
    """
    Four-level encoder for one 1-ch modality. No stem.
    Dropout3d at enc2 and enc3.
        enc1 : 1  → 16   full resolution
        enc2 : 16 → 32   → Dropout3d → MaxPool3d(2)
        enc3 : 32 → 64   → Dropout3d → MaxPool3d(2)
        enc4 : 64 → 128  → MaxPool3d(2)
    """

    def __init__(self, mc_dropout_p=0.1):
        super().__init__()
        self.enc1  = _conv_block(1,  16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2  = _conv_block(16, 32)
        self.drop2 = nn.Dropout3d(mc_dropout_p)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3  = _conv_block(32, 64)
        self.drop3 = nn.Dropout3d(mc_dropout_p)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4  = _conv_block(64, 128)
        self.pool4 = nn.MaxPool3d(2)

    def forward(self, x):
        s1  = self.enc1(x)
        e2  = self.enc2(self.pool1(s1))
        d2  = self.drop2(e2)
        e3  = self.enc3(self.pool2(d2))
        d3  = self.drop3(e3)
        s4  = self.enc4(self.pool3(d3))
        bot = self.pool4(s4)
        # return skips (post-dropout at enc2/enc3, consistent with V1)
        return s1, d2, d3, s4, bot


class AblationV2(_MCMixin, nn.Module):
    """
    Four-encoder / single-decoder. No stem. Dropout3d at enc2/enc3.

    Each modality encoded independently via _ModalityEncoder.
    Fused at every scale: cat(4×ch) + 1×1 proj → target ch.
        fuse_s1  : 4×16  = 64  → 16 ch
        fuse_s2  : 4×32  = 128 → 32 ch
        fuse_s3  : 4×64  = 256 → 64 ch
        fuse_s4  : 4×128 = 512 → 128 ch
        fuse_bot : 4×128 = 512 → 128 ch

    Decoder:
        up0/dec0 : cat(128, fs4=128) = 256 → 128
        up1/dec1 : cat(128, fs3=64)  = 192 → 128
        up2/dec2 : cat(64,  fs2=32)  = 96  → 64
        up3/final: cat(32,  fs1=16)  = 48  → 16
        out      : 16 → 3
    """

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__()
        self.n_passes = n_passes

        self.enc_t1    = _ModalityEncoder(mc_dropout_p)
        self.enc_t1ce  = _ModalityEncoder(mc_dropout_p)
        self.enc_t2    = _ModalityEncoder(mc_dropout_p)
        self.enc_flair = _ModalityEncoder(mc_dropout_p)

        self.fuse_s1  = _fuse_proj(4 * 16,  16)
        self.fuse_s2  = _fuse_proj(4 * 32,  32)
        self.fuse_s3  = _fuse_proj(4 * 64,  64)
        self.fuse_s4  = _fuse_proj(4 * 128, 128)
        self.fuse_bot = _fuse_proj(4 * 128, 128)

        self.up0    = nn.ConvTranspose3d(128, 128, 2, 2)
        self.dec0   = _conv_block(256, 128)
        self.up1    = nn.ConvTranspose3d(128, 128, 2, 2)
        self.dec1   = _conv_block(192, 128)
        self.up2    = nn.ConvTranspose3d(128, 64,  2, 2)
        self.dec2   = _conv_block(96,  64)
        self.up3    = nn.ConvTranspose3d(64,  32,  2, 2)
        self.final  = _conv_block(48,  16)
        self.out_conv = nn.Conv3d(16, 3, 1)

    def forward(self, t1, t1ce, t2, flair):
        s1_1, s2_1, s3_1, s4_1, b1 = self.enc_t1(t1)
        s1_2, s2_2, s3_2, s4_2, b2 = self.enc_t1ce(t1ce)
        s1_3, s2_3, s3_3, s4_3, b3 = self.enc_t2(t2)
        s1_4, s2_4, s3_4, s4_4, b4 = self.enc_flair(flair)

        fs1 = self.fuse_s1 (torch.cat([s1_1, s1_2, s1_3, s1_4], dim=1))
        fs2 = self.fuse_s2 (torch.cat([s2_1, s2_2, s2_3, s2_4], dim=1))
        fs3 = self.fuse_s3 (torch.cat([s3_1, s3_2, s3_3, s3_4], dim=1))
        fs4 = self.fuse_s4 (torch.cat([s4_1, s4_2, s4_3, s4_4], dim=1))
        z   = self.fuse_bot(torch.cat([b1,   b2,   b3,   b4  ], dim=1))

        x = self.dec0(torch.cat([self.up0(z),  fs4], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  fs3], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  fs2], dim=1))
        x = self.final(torch.cat([self.up3(x), fs1], dim=1))

        seg = self.out_conv(x)
        return {'mu_et': seg[:, 0:1], 'mu_ed': seg[:, 1:2],
                'mu_ncr': seg[:, 2:3], 'seg': seg}

    def forward_mc_train(self, t1, t1ce, t2, flair, T=3):
        return self.forward(t1, t1ce, t2, flair), None


# ══════════════════════════════════════════════════════════════════════════════
# Base class for V3–V6
# ══════════════════════════════════════════════════════════════════════════════

class _V3Base(_MCMixin, nn.Module):
    """
    Shared base for AblationV3–V6.
    MSGS stems + DRB encoders + Dropout3d at enc2/enc3 + three DRB decoders.
    Subclasses override _encode() (PSDE or not) and forward() (TASD or not).
    """

    def __init__(self, mc_dropout_p=0.1, n_passes=10):
        super().__init__()
        self.n_passes = n_passes

        # ── Stems ─────────────────────────────────────────────────────────────
        self.stem_t1    = MSGS(1, 16)
        self.stem_t1ce  = MSGS(1, 16)
        self.stem_t2    = MSGS(1, 16)
        self.stem_flair = MSGS(1, 16)

        self.highres_fusion = nn.Sequential(
            nn.Conv3d(64, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(inplace=True))
        self.shared_proj = nn.Sequential(
            nn.Conv3d(64, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(inplace=True))

        # ── Encoder branch A ──────────────────────────────────────────────────
        self.enc_a1  = DRB(32, 32);   self.pool_a1 = nn.MaxPool3d(2)
        self.enc_a2  = DRB(32, 64);   self.drop_a2 = nn.Dropout3d(mc_dropout_p)
        self.pool_a2 = nn.MaxPool3d(2)
        self.enc_a3  = DRB(64, 128);  self.drop_a3 = nn.Dropout3d(mc_dropout_p)
        self.pool_a3 = nn.MaxPool3d(2)

        # ── Encoder branch B ──────────────────────────────────────────────────
        self.enc_b1  = DRB(32, 32);   self.pool_b1 = nn.MaxPool3d(2)
        self.enc_b2  = DRB(32, 64);   self.drop_b2 = nn.Dropout3d(mc_dropout_p)
        self.pool_b2 = nn.MaxPool3d(2)
        self.enc_b3  = DRB(64, 128);  self.drop_b3 = nn.Dropout3d(mc_dropout_p)
        self.pool_b3 = nn.MaxPool3d(2)

        # ── Encoder branch S ──────────────────────────────────────────────────
        self.enc_s1  = DRB(32, 32);   self.pool_s1 = nn.MaxPool3d(2)
        self.enc_s2  = DRB(32, 64);   self.drop_s2 = nn.Dropout3d(mc_dropout_p)
        self.pool_s2 = nn.MaxPool3d(2)
        self.enc_s3  = DRB(64, 128);  self.drop_s3 = nn.Dropout3d(mc_dropout_p)
        self.pool_s3 = nn.MaxPool3d(2)

        # ── Decoders ──────────────────────────────────────────────────────────
        def _make_decoder():
            return nn.ModuleDict({
                "up0"  : nn.ConvTranspose3d(128, 128, 2, 2),
                "dec0" : DRB(256, 128),
                "up1"  : nn.ConvTranspose3d(128, 128, 2, 2),
                "dec1" : DRB(192, 128),
                "up2"  : nn.ConvTranspose3d(128, 64,  2, 2),
                "dec2" : DRB(96,  64),
                "up3"  : nn.ConvTranspose3d(64,  32,  2, 2),
                "final": DRB(64,  16),
                "out"  : nn.Conv3d(16, 1, 1),
            })
        self.dec_et  = _make_decoder()
        self.dec_ed  = _make_decoder()
        self.dec_ncr = _make_decoder()

    def _stem(self, t1, t1ce, t2, flair):
        f1h, f1 = self.stem_t1(t1)
        f2h, f2 = self.stem_t1ce(t1ce)
        f3h, f3 = self.stem_t2(t2)
        f4h, f4 = self.stem_flair(flair)
        high_res = self.highres_fusion(torch.cat([f1h, f2h, f3h, f4h], dim=1))
        inp_a    = torch.cat([f1, f2], dim=1)
        inp_b    = torch.cat([f3, f4], dim=1)
        inp_s    = self.shared_proj(torch.cat([f1, f2, f3, f4], dim=1))
        return high_res, inp_a, inp_b, inp_s

    def _decode(self, z, dec, es1, es2, es3, high_res):
        x = dec["dec0"](torch.cat([dec["up0"](z),  es3],      dim=1))
        x = dec["dec1"](torch.cat([dec["up1"](x),  es2],      dim=1))
        x = dec["dec2"](torch.cat([dec["up2"](x),  es1],      dim=1))
        x = dec["final"](torch.cat([dec["up3"](x), high_res], dim=1))
        return dec["out"](x)

    def forward_mc_train(self, t1, t1ce, t2, flair, T=3):
        return self.forward(t1, t1ce, t2, flair), None


# ══════════════════════════════════════════════════════════════════════════════
# Ablation V3 — no PSDE / no TASD
# ══════════════════════════════════════════════════════════════════════════════

class AblationV3(_V3Base):
    """V3: MSGS + DRB + MC-Dropout. No PSDE, no TASD.
    Dropout order: encode → dropout → pool  (no PSDE to insert between)."""

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.task_et  = _task_head()
        self.task_ed  = _task_head()
        self.task_ncr = _task_head()

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1: encode → pool  (no dropout, no PSDE) ────────────────────
        ea1 = self.enc_a1(inp_a);  xa1 = self.pool_a1(ea1)
        eb1 = self.enc_b1(inp_b);  xb1 = self.pool_b1(eb1)
        es1 = self.enc_s1(inp_s);  xs1 = self.pool_s1(es1)

        # ── Level 2: encode → dropout → pool ─────────────────────────────────
        ea2 = self.enc_a2(xa1);  da2 = self.drop_a2(ea2);  xa2 = self.pool_a2(da2)
        eb2 = self.enc_b2(xb1);  db2 = self.drop_b2(eb2);  xb2 = self.pool_b2(db2)
        es2 = self.enc_s2(xs1);  ds2 = self.drop_s2(es2);  xs2 = self.pool_s2(ds2)

        # ── Level 3: encode → dropout → pool ─────────────────────────────────
        ea3 = self.enc_a3(xa2);  da3 = self.drop_a3(ea3);  xa3 = self.pool_a3(da3)
        eb3 = self.enc_b3(xb2);  db3 = self.drop_b3(eb3);  xb3 = self.pool_b3(db3)
        es3 = self.enc_s3(xs2);  ds3 = self.drop_s3(es3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        fused   = torch.cat([xa3, xb3, xs3], dim=1)
        out_et  = self._decode(self.task_et(fused),  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(self.task_ed(fused),  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(self.task_ncr(fused), self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


# ══════════════════════════════════════════════════════════════════════════════
# Ablation V4 — PSDE only (no TASD)
# ══════════════════════════════════════════════════════════════════════════════

class AblationV4(_V3Base):
    """V4: MSGS + DRB + MC-Dropout + PSDE. No TASD.
    Dropout order: encode → PSDE → dropout → pool  (matches original model)."""

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.psde1 = PSDE(32)
        self.psde2 = PSDE(64)
        self.psde3 = PSDE(128)
        self.task_et  = _task_head()
        self.task_ed  = _task_head()
        self.task_ncr = _task_head()

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1: encode → PSDE → pool  (no dropout at level 1) ──────────
        ea1 = self.enc_a1(inp_a)
        eb1 = self.enc_b1(inp_b)
        es1 = self.enc_s1(inp_s)
        es1 = self.psde1(ea1, es1, eb1)
        xa1 = self.pool_a1(ea1);  xb1 = self.pool_b1(eb1);  xs1 = self.pool_s1(es1)

        # ── Level 2: encode → PSDE → dropout → pool ──────────────────────────
        ea2 = self.enc_a2(xa1)
        eb2 = self.enc_b2(xb1)
        es2 = self.enc_s2(xs1)
        es2 = self.psde2(ea2, es2, eb2)
        da2 = self.drop_a2(ea2);  db2 = self.drop_b2(eb2);  ds2 = self.drop_s2(es2)
        xa2 = self.pool_a2(da2);  xb2 = self.pool_b2(db2);  xs2 = self.pool_s2(ds2)

        # ── Level 3: encode → PSDE → dropout → pool ──────────────────────────
        ea3 = self.enc_a3(xa2)
        eb3 = self.enc_b3(xb2)
        es3 = self.enc_s3(xs2)
        es3 = self.psde3(ea3, es3, eb3)
        da3 = self.drop_a3(ea3);  db3 = self.drop_b3(eb3);  ds3 = self.drop_s3(es3)
        xa3 = self.pool_a3(da3);  xb3 = self.pool_b3(db3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        fused   = torch.cat([xa3, xb3, xs3], dim=1)
        out_et  = self._decode(self.task_et(fused),  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(self.task_ed(fused),  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(self.task_ncr(fused), self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


class AblationV4_1(_V3Base):
    """
    V4 variant using PSDE_1 (Direct Addition, ablation P2).

    PSDE_1 replaces the full PSDE at every encoder level.
    Everything else — stems, DRB encoders, dropout positions, task heads,
    decoders — is identical to AblationV4.

    Dropout order (same as V4 / full model):
        Level 1 : encode → PSDE_1 → pool             [no dropout]
        Level 2 : encode → PSDE_1 → dropout → pool
        Level 3 : encode → PSDE_1 → dropout → pool

    What changes vs AblationV4
    ────────────────────────────
    • psde1/2/3 are PSDE_1 instances  instead of PSDE
    • No residual extraction          (r = f - proj(f) removed)
    • No channel attention            (ca_a, ca_b removed)
    • Raw f_a / f_b injected directly into f_s
    """

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.psde1    = PSDE_1(32)
        self.psde2    = PSDE_1(64)
        self.psde3    = PSDE_1(128)
        self.task_et  = _task_head()
        self.task_ed  = _task_head()
        self.task_ncr = _task_head()

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1: encode → PSDE_1 → pool  (no dropout at level 1) ─────────
        ea1 = self.enc_a1(inp_a)
        eb1 = self.enc_b1(inp_b)
        es1 = self.enc_s1(inp_s)
        es1 = self.psde1(ea1, es1, eb1)
        xa1 = self.pool_a1(ea1);  xb1 = self.pool_b1(eb1);  xs1 = self.pool_s1(es1)

        # ── Level 2: encode → PSDE_1 → dropout → pool ─────────────────────────
        ea2 = self.enc_a2(xa1)
        eb2 = self.enc_b2(xb1)
        es2 = self.enc_s2(xs1)
        es2 = self.psde2(ea2, es2, eb2)
        da2 = self.drop_a2(ea2);  db2 = self.drop_b2(eb2);  ds2 = self.drop_s2(es2)
        xa2 = self.pool_a2(da2);  xb2 = self.pool_b2(db2);  xs2 = self.pool_s2(ds2)

        # ── Level 3: encode → PSDE_1 → dropout → pool ─────────────────────────
        ea3 = self.enc_a3(xa2)
        eb3 = self.enc_b3(xb2)
        es3 = self.enc_s3(xs2)
        es3 = self.psde3(ea3, es3, eb3)
        da3 = self.drop_a3(ea3);  db3 = self.drop_b3(eb3);  ds3 = self.drop_s3(es3)
        xa3 = self.pool_a3(da3);  xb3 = self.pool_b3(db3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        fused   = torch.cat([xa3, xb3, xs3], dim=1)
        out_et  = self._decode(self.task_et(fused),  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(self.task_ed(fused),  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(self.task_ncr(fused), self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


class AblationV4_2(_V3Base):
    """
    V4 variant using PSDE_2 (Residual Only, ablation P3).

    PSDE_2 replaces the full PSDE at every encoder level.
    Everything else — stems, DRB encoders, dropout positions, task heads,
    decoders — is identical to AblationV4.

    Dropout order (same as V4 / full model):
        Level 1 : encode → PSDE_2 → pool             [no dropout]
        Level 2 : encode → PSDE_2 → dropout → pool
        Level 3 : encode → PSDE_2 → dropout → pool

    What changes vs AblationV4
    ────────────────────────────
    • psde1/2/3 are PSDE_2 instances  instead of PSDE
    • Projection + residual extraction kept  (r = f - proj(f))
    • Channel attention removed              (ca_a, ca_b removed)
    • Unweighted residuals injected into f_s (no SE gating)

    What changes vs AblationV4_1
    ──────────────────────────────
    • Adds residual disentanglement on top of direct injection
    • Lets you attribute the gap (V4_1 → V4_2) purely to the residual step
    """

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.psde1    = PSDE_2(32)
        self.psde2    = PSDE_2(64)
        self.psde3    = PSDE_2(128)
        self.task_et  = _task_head()
        self.task_ed  = _task_head()
        self.task_ncr = _task_head()

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1: encode → PSDE_2 → pool  (no dropout at level 1) ─────────
        ea1 = self.enc_a1(inp_a)
        eb1 = self.enc_b1(inp_b)
        es1 = self.enc_s1(inp_s)
        es1 = self.psde1(ea1, es1, eb1)
        xa1 = self.pool_a1(ea1);  xb1 = self.pool_b1(eb1);  xs1 = self.pool_s1(es1)

        # ── Level 2: encode → PSDE_2 → dropout → pool ─────────────────────────
        ea2 = self.enc_a2(xa1)
        eb2 = self.enc_b2(xb1)
        es2 = self.enc_s2(xs1)
        es2 = self.psde2(ea2, es2, eb2)
        da2 = self.drop_a2(ea2);  db2 = self.drop_b2(eb2);  ds2 = self.drop_s2(es2)
        xa2 = self.pool_a2(da2);  xb2 = self.pool_b2(db2);  xs2 = self.pool_s2(ds2)

        # ── Level 3: encode → PSDE_2 → dropout → pool ─────────────────────────
        ea3 = self.enc_a3(xa2)
        eb3 = self.enc_b3(xb2)
        es3 = self.enc_s3(xs2)
        es3 = self.psde3(ea3, es3, eb3)
        da3 = self.drop_a3(ea3);  db3 = self.drop_b3(eb3);  ds3 = self.drop_s3(es3)
        xa3 = self.pool_a3(da3);  xb3 = self.pool_b3(db3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        fused   = torch.cat([xa3, xb3, xs3], dim=1)
        out_et  = self._decode(self.task_et(fused),  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(self.task_ed(fused),  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(self.task_ncr(fused), self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


# ══════════════════════════════════════════════════════════════════════════════
# Ablation V5 — TASD only (no PSDE)
# ══════════════════════════════════════════════════════════════════════════════

class AblationV5(_V3Base):
    """V5: MSGS + DRB + MC-Dropout + TASD. No PSDE.
    Dropout order: encode → dropout → pool  (no PSDE)."""

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.tasd = TASD(128, 42)

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1 ───────────────────────────────────────────────────────────
        ea1 = self.enc_a1(inp_a);  xa1 = self.pool_a1(ea1)
        eb1 = self.enc_b1(inp_b);  xb1 = self.pool_b1(eb1)
        es1 = self.enc_s1(inp_s);  xs1 = self.pool_s1(es1)

        # ── Level 2 ───────────────────────────────────────────────────────────
        ea2 = self.enc_a2(xa1);  da2 = self.drop_a2(ea2);  xa2 = self.pool_a2(da2)
        eb2 = self.enc_b2(xb1);  db2 = self.drop_b2(eb2);  xb2 = self.pool_b2(db2)
        es2 = self.enc_s2(xs1);  ds2 = self.drop_s2(es2);  xs2 = self.pool_s2(ds2)

        # ── Level 3 ───────────────────────────────────────────────────────────
        ea3 = self.enc_a3(xa2);  da3 = self.drop_a3(ea3);  xa3 = self.pool_a3(da3)
        eb3 = self.enc_b3(xb2);  db3 = self.drop_b3(eb3);  xb3 = self.pool_b3(db3)
        es3 = self.enc_s3(xs2);  ds3 = self.drop_s3(es3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        z_et, z_ed, z_ncr = self.tasd(xa3, xb3, xs3)
        out_et  = self._decode(z_et,  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(z_ed,  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(z_ncr, self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


class AblationV5_1(_V3Base):
    """
    V5 variant using TASD_1 (No gating, additive fusion, ablation T1).

    TASD_1 replaces the full TASD at the bottleneck.
    Everything else — stems, DRB encoders, dropout positions, decoders —
    is identical to AblationV5.

    Dropout order (same as V5, no PSDE):
        Level 1 : encode → pool              [no dropout]
        Level 2 : encode → dropout → pool
        Level 3 : encode → dropout → pool

    What changes vs AblationV5
    ────────────────────────────
    • tasd is TASD_1  (gate_et / gate_ed / gate_ncr removed)
    • Fusion: z = x_private + s_task  (no sigmoid modulation)

    What changes vs AblationV5_2
    ──────────────────────────────
    • V5_2 still has a gate (sourced from xs3)
    • V5_1 has NO gate at all → cleanest no-gate baseline
    """

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.tasd = TASD_1(128, 42)

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1 ───────────────────────────────────────────────────────────
        ea1 = self.enc_a1(inp_a);  xa1 = self.pool_a1(ea1)
        eb1 = self.enc_b1(inp_b);  xb1 = self.pool_b1(eb1)
        es1 = self.enc_s1(inp_s);  xs1 = self.pool_s1(es1)

        # ── Level 2 ───────────────────────────────────────────────────────────
        ea2 = self.enc_a2(xa1);  da2 = self.drop_a2(ea2);  xa2 = self.pool_a2(da2)
        eb2 = self.enc_b2(xb1);  db2 = self.drop_b2(eb2);  xb2 = self.pool_b2(db2)
        es2 = self.enc_s2(xs1);  ds2 = self.drop_s2(es2);  xs2 = self.pool_s2(ds2)

        # ── Level 3 ───────────────────────────────────────────────────────────
        ea3 = self.enc_a3(xa2);  da3 = self.drop_a3(ea3);  xa3 = self.pool_a3(da3)
        eb3 = self.enc_b3(xb2);  db3 = self.drop_b3(eb3);  xb3 = self.pool_b3(db3)
        es3 = self.enc_s3(xs2);  ds3 = self.drop_s3(es3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        z_et, z_ed, z_ncr = self.tasd(xa3, xb3, xs3)
        out_et  = self._decode(z_et,  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(z_ed,  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(z_ncr, self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


class AblationV5_2(_V3Base):
    """
    V5 variant using TASD_2 (Symmetric shared-stream gating, ablation T3).

    TASD_2 replaces the full TASD at the bottleneck.
    Everything else — stems, DRB encoders, dropout positions, decoders —
    is identical to AblationV5.

    Dropout order (same as V5, no PSDE):
        Level 1 : encode → pool              [no dropout]
        Level 2 : encode → dropout → pool
        Level 3 : encode → dropout → pool

    What changes vs AblationV5
    ────────────────────────────
    • tasd is TASD_2
    • Gate source: xs3 for all tasks  (was xa3 for ET/NCR, xb3 for ED)
    • Formula: z = x_private + sigmoid(gate(xs3)) * s_task

    What changes vs AblationV5_1
    ──────────────────────────────
    • V5_1 has no gate; V5_2 adds a gate
    • Gap (V5_1 → V5_2) isolates the pure effect of having ANY gate
    • Gap (V5_2 → V5)   isolates the effect of gate source (shared vs private)
    """

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.tasd = TASD_2(128, 42)

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1 ───────────────────────────────────────────────────────────
        ea1 = self.enc_a1(inp_a);  xa1 = self.pool_a1(ea1)
        eb1 = self.enc_b1(inp_b);  xb1 = self.pool_b1(eb1)
        es1 = self.enc_s1(inp_s);  xs1 = self.pool_s1(es1)

        # ── Level 2 ───────────────────────────────────────────────────────────
        ea2 = self.enc_a2(xa1);  da2 = self.drop_a2(ea2);  xa2 = self.pool_a2(da2)
        eb2 = self.enc_b2(xb1);  db2 = self.drop_b2(eb2);  xb2 = self.pool_b2(db2)
        es2 = self.enc_s2(xs1);  ds2 = self.drop_s2(es2);  xs2 = self.pool_s2(ds2)

        # ── Level 3 ───────────────────────────────────────────────────────────
        ea3 = self.enc_a3(xa2);  da3 = self.drop_a3(ea3);  xa3 = self.pool_a3(da3)
        eb3 = self.enc_b3(xb2);  db3 = self.drop_b3(eb3);  xb3 = self.pool_b3(db3)
        es3 = self.enc_s3(xs2);  ds3 = self.drop_s3(es3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        z_et, z_ed, z_ncr = self.tasd(xa3, xb3, xs3)
        out_et  = self._decode(z_et,  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(z_ed,  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(z_ncr, self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


# ══════════════════════════════════════════════════════════════════════════════
# Ablation V6 — PSDE + TASD (no Epi loss only)
# ══════════════════════════════════════════════════════════════════════════════

class AblationV6(_V3Base):
    """V6: MSGS + DRB + MC-Dropout + PSDE + TASD. Full arch, no epistemic loss.
    Dropout order: encode → PSDE → dropout → pool  (matches original model)."""

    def __init__(self, mc_dropout_p=0.1, n_passes=10, **kwargs):
        super().__init__(mc_dropout_p, n_passes)
        self.psde1 = PSDE(32)
        self.psde2 = PSDE(64)
        self.psde3 = PSDE(128)
        self.tasd  = TASD(128, 42)

    def _encode(self, inp_a, inp_b, inp_s):
        # ── Level 1: encode → PSDE → pool ────────────────────────────────────
        ea1 = self.enc_a1(inp_a)
        eb1 = self.enc_b1(inp_b)
        es1 = self.enc_s1(inp_s)
        es1 = self.psde1(ea1, es1, eb1)
        xa1 = self.pool_a1(ea1);  xb1 = self.pool_b1(eb1);  xs1 = self.pool_s1(es1)

        # ── Level 2: encode → PSDE → dropout → pool ──────────────────────────
        ea2 = self.enc_a2(xa1)
        eb2 = self.enc_b2(xb1)
        es2 = self.enc_s2(xs1)
        es2 = self.psde2(ea2, es2, eb2)
        da2 = self.drop_a2(ea2);  db2 = self.drop_b2(eb2);  ds2 = self.drop_s2(es2)
        xa2 = self.pool_a2(da2);  xb2 = self.pool_b2(db2);  xs2 = self.pool_s2(ds2)

        # ── Level 3: encode → PSDE → dropout → pool ──────────────────────────
        ea3 = self.enc_a3(xa2)
        eb3 = self.enc_b3(xb2)
        es3 = self.enc_s3(xs2)
        es3 = self.psde3(ea3, es3, eb3)
        da3 = self.drop_a3(ea3);  db3 = self.drop_b3(eb3);  ds3 = self.drop_s3(es3)
        xa3 = self.pool_a3(da3);  xb3 = self.pool_b3(db3);  xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    def forward(self, t1, t1ce, t2, flair):
        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)
        z_et, z_ed, z_ncr = self.tasd(xa3, xb3, xs3)
        out_et  = self._decode(z_et,  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self._decode(z_ed,  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self._decode(z_ncr, self.dec_ncr, es1, es2, es3, high_res)
        return {'mu_et': out_et, 'mu_ed': out_ed, 'mu_ncr': out_ncr,
                'seg': torch.cat([out_et, out_ed, out_ncr], dim=1)}


# ══════════════════════════════════════════════════════════════════════════════
# Shared Loss  (drop-in for BrainTumorLoss)
# ══════════════════════════════════════════════════════════════════════════════

class AblationLoss(nn.Module):
    """
    Dice + CE only. Drop-in for BrainTumorLoss:
      - accepts preds_stack kwarg (ignored)
      - accepts lam_epi kwarg (ignored)
      - always returns 'epi': 0.0 for consistent CSV logging
    """

    def __init__(self, lam_dice=1.0, lam_ce=0.5, lam_epi=0.0, smooth=1e-5):
        super().__init__()
        self.lam_dice = lam_dice
        self.lam_ce   = lam_ce
        self.smooth   = smooth
        self.ce       = nn.BCEWithLogitsLoss()

    def _dice(self, pred, target):
        pred = torch.sigmoid(pred)
        num  = 2 * (pred * target).sum(dim=(2, 3, 4)) + self.smooth
        den  = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + self.smooth
        return (1 - num / den).mean()

    def forward(self, outputs, targets, preds_stack=None):
        seg  = outputs['seg']
        p_et,  p_ed,  p_ncr  = seg[:, 0:1],     seg[:, 1:2],     seg[:, 2:3]
        t_et,  t_ed,  t_ncr  = targets[:, 0:1], targets[:, 1:2], targets[:, 2:3]
        l_dice = (self._dice(p_et, t_et) + self._dice(p_ed, t_ed) + self._dice(p_ncr, t_ncr)) / 3
        l_ce   = (self.ce(p_et, t_et)   + self.ce(p_ed, t_ed)   + self.ce(p_ncr, t_ncr))   / 3
        total  = self.lam_dice * l_dice + self.lam_ce * l_ce
        return total, {'dice': l_dice.item(), 'ce': l_ce.item(), 'epi': 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(1, 1, 128, 128, 128, device=device)

    models = [
        ("V1",   AblationV1),
        ("V2",   AblationV2),
        ("V3",   AblationV3),
        ("V4",   AblationV4),
        ("V4_1", AblationV4_1),
        ("V4_2", AblationV4_2),
        ("V5",   AblationV5),
        ("V5_1", AblationV5_1),
        ("V5_2", AblationV5_2),
        ("V6",   AblationV6),
    ]

    for name, cls in models:
        m      = cls().to(device)
        out    = m(dummy, dummy, dummy, dummy)
        mc     = m.mc_forward(dummy, dummy, dummy, dummy, n_passes=3)
        n      = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_drop = sum(1 for mod in m.modules() if isinstance(mod, nn.Dropout3d))
        print(f"{name:5s} | seg: {out['seg'].shape} | "
              f"var_et: {mc['var_et'].shape} | "
              f"dropout_layers: {n_drop} | "
              f"params: {n / 1_000_000:.2f}M")