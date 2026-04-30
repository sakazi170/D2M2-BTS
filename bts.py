import torch
import torch.nn as nn
from modules.blocks import MSGS, DRB
from modules.psde import PSDE
from modules.tasd import TASD


class BrainTumorSegNet(nn.Module):

    def __init__(self, mc_dropout_p=0.1, n_passes=10):
        super().__init__()

        self.n_passes = n_passes

        # STEMS
        self.stem_t1    = MSGS(1, 16)
        self.stem_t1ce  = MSGS(1, 16)
        self.stem_t2    = MSGS(1, 16)
        self.stem_flair = MSGS(1, 16)

        self.highres_fusion = nn.Sequential(
            nn.Conv3d(64, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        self.shared_proj = nn.Sequential(
            nn.Conv3d(64, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # ENCODERS
        self.enc_a1  = DRB(32, 32)
        self.pool_a1 = nn.MaxPool3d(2)
        self.enc_a2  = DRB(32, 64)
        self.drop_a2 = nn.Dropout3d(mc_dropout_p)
        self.pool_a2 = nn.MaxPool3d(2)
        self.enc_a3  = DRB(64, 128)
        self.drop_a3 = nn.Dropout3d(mc_dropout_p)
        self.pool_a3 = nn.MaxPool3d(2)

        self.enc_b1  = DRB(32, 32)
        self.pool_b1 = nn.MaxPool3d(2)
        self.enc_b2  = DRB(32, 64)
        self.drop_b2 = nn.Dropout3d(mc_dropout_p)
        self.pool_b2 = nn.MaxPool3d(2)
        self.enc_b3  = DRB(64, 128)
        self.drop_b3 = nn.Dropout3d(mc_dropout_p)
        self.pool_b3 = nn.MaxPool3d(2)

        self.enc_s1  = DRB(32, 32)
        self.pool_s1 = nn.MaxPool3d(2)
        self.enc_s2  = DRB(32, 64)
        self.drop_s2 = nn.Dropout3d(mc_dropout_p)
        self.pool_s2 = nn.MaxPool3d(2)
        self.enc_s3  = DRB(64, 128)
        self.drop_s3 = nn.Dropout3d(mc_dropout_p)
        self.pool_s3 = nn.MaxPool3d(2)

        # PSDE
        self.psde1 = PSDE(32)
        self.psde2 = PSDE(64)
        self.psde3 = PSDE(128)

        # TASD
        self.tasd = TASD(128, 42)


        def make_decoder():
            return nn.ModuleDict({
                "up0"  : nn.ConvTranspose3d(128, 128, 2, 2),
                "dec0" : DRB(256, 128),
                "up1"  : nn.ConvTranspose3d(128, 128, 2, 2),
                "dec1" : DRB(192, 128),
                "up2"  : nn.ConvTranspose3d(128, 64,  2, 2),
                "dec2" : DRB(96,  64),
                "up3"  : nn.ConvTranspose3d(64,  32,  2, 2),
                "final": DRB(64,  16),
                "out"  : nn.Conv3d(16, 1, 1)
            })

        self.dec_et  = make_decoder()
        self.dec_ed  = make_decoder()
        self.dec_ncr = make_decoder()

    # ── decode ────────────────────────────────────────────────────────────────
    def decode(self, z, decoder, es1, es2, es3, high_res):

        x = decoder["up0"](z)
        x = decoder["dec0"](torch.cat([x, es3], dim=1))

        x = decoder["up1"](x)
        x = decoder["dec1"](torch.cat([x, es2], dim=1))

        x = decoder["up2"](x)
        x = decoder["dec2"](torch.cat([x, es1], dim=1))

        x = decoder["up3"](x)
        x = decoder["final"](torch.cat([x, high_res], dim=1))

        return decoder["out"](x)

    # ── encoder shared logic ──────────────────────────────────────────────────
    def _encode(self, inp_a, inp_b, inp_s):

        # ===== Level 1 =====
        ea1 = self.enc_a1(inp_a)
        xa1 = self.pool_a1(ea1)

        eb1 = self.enc_b1(inp_b)
        xb1 = self.pool_b1(eb1)

        es1 = self.enc_s1(inp_s)
        es1 = self.psde1(ea1, es1, eb1)
        xs1 = self.pool_s1(es1)

        # ===== Level 2 =====
        ea2 = self.enc_a2(xa1)
        eb2 = self.enc_b2(xb1)
        es2 = self.enc_s2(xs1)

        es2 = self.psde2(ea2, es2, eb2)

        da2 = self.drop_a2(ea2)
        db2 = self.drop_b2(eb2)
        ds2 = self.drop_s2(es2)

        xa2 = self.pool_a2(da2)
        xb2 = self.pool_b2(db2)
        xs2 = self.pool_s2(ds2)

        # ===== Level 3 =====
        ea3 = self.enc_a3(xa2)
        eb3 = self.enc_b3(xb2)
        es3 = self.enc_s3(xs2)

        es3 = self.psde3(ea3, es3, eb3)

        da3 = self.drop_a3(ea3)
        db3 = self.drop_b3(eb3)
        ds3 = self.drop_s3(es3)

        xa3 = self.pool_a3(da3)
        xb3 = self.pool_b3(db3)
        xs3 = self.pool_s3(ds3)

        return es1, ds2, ds3, xa3, xb3, xs3

    # ── stem shared logic ─────────────────────────────────────────────────────
    def _stem(self, t1, t1ce, t2, flair):
        f1h, f1 = self.stem_t1(t1)
        f2h, f2 = self.stem_t1ce(t1ce)
        f3h, f3 = self.stem_t2(t2)
        f4h, f4 = self.stem_flair(flair)

        high_res = self.highres_fusion(
            torch.cat([f1h, f2h, f3h, f4h], dim=1))

        inp_a = torch.cat([f1, f2], dim=1)
        inp_b = torch.cat([f3, f4], dim=1)
        inp_s = self.shared_proj(
            torch.cat([f1, f2, f3, f4], dim=1))

        return high_res, inp_a, inp_b, inp_s

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, t1, t1ce, t2, flair):

        high_res, inp_a, inp_b, inp_s = self._stem(t1, t1ce, t2, flair)
        es1, es2, es3, xa3, xb3, xs3  = self._encode(inp_a, inp_b, inp_s)

        z_et, z_ed, z_ncr = self.tasd(xa3, xb3, xs3)

        out_et  = self.decode(z_et,  self.dec_et,  es1, es2, es3, high_res)
        out_ed  = self.decode(z_ed,  self.dec_ed,  es1, es2, es3, high_res)
        out_ncr = self.decode(z_ncr, self.dec_ncr, es1, es2, es3, high_res)

        return {
            'mu_et' : out_et,
            'mu_ed' : out_ed,
            'mu_ncr': out_ncr,
            'seg'   : torch.cat([out_et, out_ed, out_ncr], dim=1),
        }

    # ── MC forward for training (T passes, returns preds_stack) ──────────────
    def forward_mc_train(self, t1, t1ce, t2, flair, T=3):
        preds = []

        # pass 1 — keep graph for main loss backprop
        out1 = self.forward(t1, t1ce, t2, flair)
        preds.append(torch.sigmoid(out1['seg']))  # keep as-is (attached to graph)

        # pass 2..T — detach immediately, only need probs for variance
        for _ in range(T - 1):
            with torch.no_grad():
                out = self.forward(t1, t1ce, t2, flair)
                preds.append(torch.sigmoid(out['seg']).detach())

        preds_stack = torch.stack(preds)  # (T, B, 3, H, W, D)
        mean_seg = preds_stack.mean(dim=0)

        return {
            'mu_et': mean_seg[:, 0:1],
            'mu_ed': mean_seg[:, 1:2],
            'mu_ncr': mean_seg[:, 2:3],
            'seg': out1['seg'],  # use pass 1 logits for dice+ce
        }, preds_stack

    # ── MC forward for inference (uncertainty maps) ───────────────────────────
    def mc_forward(self, t1, t1ce, t2, flair, n_passes=None):

        n_passes = n_passes or self.n_passes

        # activate only dropout layers, keep norm layers in eval
        for m in self.modules():
            if isinstance(m, nn.Dropout3d):
                m.train()

        et_preds  = []
        ed_preds  = []
        ncr_preds = []

        with torch.no_grad():
            for _ in range(n_passes):
                out = self.forward(t1, t1ce, t2, flair)
                et_preds.append(torch.sigmoid(out['mu_et']))
                ed_preds.append(torch.sigmoid(out['mu_ed']))
                ncr_preds.append(torch.sigmoid(out['mu_ncr']))

        # (n_passes, B, 1, H, W, D)
        et_stack  = torch.stack(et_preds)
        ed_stack  = torch.stack(ed_preds)
        ncr_stack = torch.stack(ncr_preds)

        mu_et  = et_stack.mean(dim=0)
        mu_ed  = ed_stack.mean(dim=0)
        mu_ncr = ncr_stack.mean(dim=0)

        # epistemic uncertainty — paper formula: (1/T) Σ (ŷ_i(t) - ȳ_i)²
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


# ── Loss ──────────────────────────────────────────────────────────────────────
class BrainTumorLoss(nn.Module):

    def __init__(self, lam_dice=1.0, lam_ce=0.5, lam_epi=0.01, smooth=1e-5):
        super().__init__()
        self.lam_dice = lam_dice
        self.lam_ce   = lam_ce
        self.lam_epi  = lam_epi
        self.smooth   = smooth
        self.ce       = nn.BCEWithLogitsLoss()

    # ── dice ──────────────────────────────────────────────────────────────────
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        num  = 2 * (pred * target).sum(dim=(2, 3, 4)) + self.smooth
        den  = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + self.smooth
        return (1 - num / den).mean()

    # ── epistemic loss ────────────────────────────────────────────────────────
    def epistemic_loss(self, preds_stack, targets):
        """
        Implements the paper formula:
            L_epistemic = (1/T) Σ (ŷ_i(t) - ȳ_i)²

        Modified into a proper training objective:
            - penalize HIGH variance on correct voxels (be confident when right)
            - penalize LOW  variance on wrong  voxels (be uncertain when wrong)

        Args:
            preds_stack : (T, B, 3, H, W, D) sigmoid probs across T passes
            targets     : (B, 3, H, W, D)    ground truth binary mask
        """
        # ȳ_i — mean prediction  (B, 3, H, W, D)
        mean_pred = preds_stack.mean(dim=0)

        # paper formula: (1/T) Σ (ŷ_i(t) - ȳ_i)²
        variance = ((preds_stack - mean_pred.unsqueeze(0)) ** 2).mean(dim=0)

        # correctness mask
        correct = ((mean_pred > 0.5).float() == targets).float()  # 1=correct
        wrong   = 1.0 - correct                                    # 1=wrong

        # correct voxels → minimize variance (confident when right)
        loss_correct = (variance * correct).sum() / \
                       (correct.sum() + self.smooth)

        # wrong voxels → maximize variance (uncertain when wrong)
        # negate so minimizing loss = maximizing variance on wrong voxels
        loss_wrong = (variance * wrong).sum() / \
                     (wrong.sum() + self.smooth)

        # subtract wrong term: minimizing (correct - 0.1*wrong) achieves both goals
        return loss_correct - 0.1 * loss_wrong

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, outputs, targets, preds_stack=None):
        """
        Args:
            outputs     : dict with 'seg' key  (B, 3, H, W, D) logits
            targets     : (B, 3, H, W, D) binary ground truth
            preds_stack : (T, B, 3, H, W, D) optional — enables epistemic loss
        """
        seg = outputs['seg']

        t_et  = targets[:, 0:1]
        t_ed  = targets[:, 1:2]
        t_ncr = targets[:, 2:3]

        p_et  = seg[:, 0:1]
        p_ed  = seg[:, 1:2]
        p_ncr = seg[:, 2:3]

        l_dice = (
            self.dice_loss(p_et,  t_et)  +
            self.dice_loss(p_ed,  t_ed)  +
            self.dice_loss(p_ncr, t_ncr)
        ) / 3

        l_ce = (
            self.ce(p_et,  t_et)  +
            self.ce(p_ed,  t_ed)  +
            self.ce(p_ncr, t_ncr)
        ) / 3

        total = self.lam_dice * l_dice + self.lam_ce * l_ce

        # epistemic loss — only computed when preds_stack is provided
        l_epi = torch.tensor(0.0, device=seg.device)
        if preds_stack is not None:
            l_epi = self.epistemic_loss(preds_stack, targets)
            total = total + self.lam_epi * l_epi

        return total, {
            'dice': l_dice.item(),
            'ce'  : l_ce.item(),
            'epi' : l_epi.item(),
        }

