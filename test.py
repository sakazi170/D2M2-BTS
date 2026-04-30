import os
import csv
import time
import torch
import argparse
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from utils.metric import calculate_metrics_with_debug
from utils.test_data_loader import BraTSDataset, post_process_prediction

from bts import BrainTumorSegNet
from ablation_models import (AblationV1, AblationV2, AblationV3, AblationV4, AblationV4_1,
                             AblationV4_2, AblationV5, AblationV5_1, AblationV5_1, AblationV6, AblationV5_2)


# ── Model registry ────────────────────────────────────────────────────────────
MODEL_DICT = {
    'bts' : BrainTumorSegNet,
    'v1': AblationV1,
    'v2': AblationV2,
    'v3': AblationV3,
    'v4': AblationV4,
    'v4_1': AblationV4_1,
    'v4_2': AblationV4_2,
    'v5': AblationV5,
    'v5_1': AblationV5_1,
    'v5_2': AblationV5_2,
    'v6': AblationV6,
}


# ── Test Time Augmentation ────────────────────────────────────────────────────
class TestTimeAugmentation:
    def __init__(self, device):
        self.device = device

    def augment(self, t1, t1ce, t2, flair):
        flip_combinations = [
            [], [2], [3], [4],
            [2, 3], [2, 4], [3, 4], [2, 3, 4]
        ]
        results = []
        for dims in flip_combinations:
            if dims:
                results.append((
                    torch.flip(t1,    dims=dims),
                    torch.flip(t1ce,  dims=dims),
                    torch.flip(t2,    dims=dims),
                    torch.flip(flair, dims=dims)))
            else:
                results.append((t1, t1ce, t2, flair))
        return results, flip_combinations

    def reverse_augment(self, seg_list, flip_combinations):
        reversed_preds = []
        for seg, dims in zip(seg_list, flip_combinations):
            reversed_preds.append(torch.flip(seg, dims=dims) if dims else seg)
        return reversed_preds


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_path_from_checkpoint(checkpoint_path):
    parts = checkpoint_path.split('/')
    try:
        idx = parts.index('checkpoints')
        return '/'.join(parts[idx + 1:-1])
    except ValueError:
        return 'default'


def extract_model_name_from_checkpoint(checkpoint_path):
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def get_seg_from_outputs(outputs, thr_et, thr_ed, thr_ncr, already_sigmoid=False):
    """
    Threshold each region independently.
    already_sigmoid=True  → mc_forward output (probabilities)
    already_sigmoid=False → single forward output (logits)
    """
    if already_sigmoid:
        et  = (outputs['mu_et']  > thr_et ).float()
        ed  = (outputs['mu_ed']  > thr_ed ).float()
        ncr = (outputs['mu_ncr'] > thr_ncr).float()
    else:
        et  = (torch.sigmoid(outputs['mu_et'])  > thr_et ).float()
        ed  = (torch.sigmoid(outputs['mu_ed'])  > thr_ed ).float()
        ncr = (torch.sigmoid(outputs['mu_ncr']) > thr_ncr).float()
    return torch.cat([et, ed, ncr], dim=1)   # (B, 3, H, W, D)


def binary_to_brats_label(seg):
    et  = seg[0].astype(bool)
    ed  = seg[1].astype(bool)
    ncr = seg[2].astype(bool)
    label_map = np.zeros(et.shape, dtype=np.uint8)
    label_map[ncr] = 1
    label_map[ed]  = 2
    label_map[et]  = 3   # ET highest priority
    return label_map


def derive_brats_regions(pred_label):
    et = (pred_label == 3).astype(np.uint8)
    tc = ((pred_label == 1) | (pred_label == 3)).astype(np.uint8)
    wt = ((pred_label == 1) | (pred_label == 2) | (pred_label == 3)).astype(np.uint8)
    return et, tc, wt


def run_inference(model, t1, t1ce, t2, flair, args, use_tta):
    """
    Handles all inference combinations: MC Dropout ON/OFF, TTA ON/OFF.
    Works identically for BrainTumorSegNet and all AblationV1–V6 models
    because every model implements both forward() and mc_forward().
    Returns outputs dict and already_sigmoid flag.
    """
    if args.mc_drop:
        if use_tta:
            aug_inputs, flip_combos = use_tta.augment(t1, t1ce, t2, flair)
            seg_list = []
            for aug_t1, aug_t1ce, aug_t2, aug_flair in aug_inputs:
                out = model.mc_forward(aug_t1, aug_t1ce, aug_t2, aug_flair,
                                       n_passes=args.n_passes)
                seg_list.append(out['seg'])
            seg_list = use_tta.reverse_augment(seg_list, flip_combos)
            mean_seg = torch.stack(seg_list).mean(dim=0)
            outputs  = {
                'mu_et' : mean_seg[:, 0:1],
                'mu_ed' : mean_seg[:, 1:2],
                'mu_ncr': mean_seg[:, 2:3],
                'seg'   : mean_seg,
                'var_et' : None,   # variance not available when TTA+MC combined
                'var_ed' : None,
                'var_ncr': None,
            }
        else:
            outputs = model.mc_forward(t1, t1ce, t2, flair,
                                       n_passes=args.n_passes)
        already_sigmoid = True

    else:
        if use_tta:
            aug_inputs, flip_combos = use_tta.augment(t1, t1ce, t2, flair)
            seg_list = []
            for aug_t1, aug_t1ce, aug_t2, aug_flair in aug_inputs:
                out = model(aug_t1, aug_t1ce, aug_t2, aug_flair)
                seg_list.append(torch.sigmoid(out['seg']))
            seg_list = use_tta.reverse_augment(seg_list, flip_combos)
            mean_seg = torch.stack(seg_list).mean(dim=0)
            outputs  = {
                'mu_et' : mean_seg[:, 0:1],
                'mu_ed' : mean_seg[:, 1:2],
                'mu_ncr': mean_seg[:, 2:3],
                'seg'   : mean_seg,
                'var_et' : None,
                'var_ed' : None,
                'var_ncr': None,
            }
            already_sigmoid = True
        else:
            outputs = model(t1, t1ce, t2, flair)
            already_sigmoid = False

    return outputs, already_sigmoid


def save_uncertainty_map(out, crop_coords, filename, unc_dir, depth=155):
    """Save mean variance across ET/ED/NCR as a NIfTI file."""
    affine_array = np.array([[-1,0,0,0],[0,-1,0,239],[0,0,1,0],[0,0,0,1]])

    if out['var_et'] is None:
        print('  [!] Uncertainty maps not available when TTA+MC combined — skipping.')
        return

    unc_map = (out['var_et'] + out['var_ed'] + out['var_ncr']) / 3
    unc_np  = unc_map[0, 0].cpu().numpy()

    xs = crop_coords[0].item()
    ys = crop_coords[1].item()
    zs = crop_coords[2].item()
    ze = min(zs + 128, depth)

    full_unc = np.zeros((240, 240, depth), dtype=np.float32)
    full_unc[xs:xs+128, ys:ys+128, zs:ze] = unc_np[:, :, :ze-zs]

    save_path = os.path.join(unc_dir, f'{filename}_uncertainty.nii.gz')
    nib.save(nib.Nifti1Image(full_unc, affine_array), save_path)
    print(f'  Uncertainty map saved → {save_path}')


def fmt_hd(v):
    """Format HD95 value — nan shown as 'nan' in CSV."""
    return 'nan' if (v is None or np.isnan(v)) else f'{v:.3f}'


# ── Args ──────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/data/qazisami/models/my_model_7/checkpoints/brats2020/btsNet/btsNet_best.pkl"


def parse_args():
    parser = argparse.ArgumentParser()

    # model / data
    parser.add_argument('--model',   type=str, default='bts')
    parser.add_argument('--dataset', type=str, default='brats2020')
    parser.add_argument('--cp',      type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--gpu',     type=str, default='2')

    parser.add_argument('--labels',       action='store_true')
    parser.add_argument('--pp',           action='store_true')
    parser.add_argument('--tta',          action='store_true')
    parser.add_argument('--custom_crop',  type=str, default=None)
    parser.add_argument('--save_pred',    action='store_true')

    parser.add_argument('--t_et',  type=float, default=0.3)
    parser.add_argument('--t_ed',  type=float, default=0.4)
    parser.add_argument('--t_ncr', type=float, default=0.35)

    parser.add_argument('--mc_drop',   action='store_true')
    parser.add_argument('--n_passes',  type=int,   default=10)
    parser.add_argument('--mc_drop_p', type=float, default=0.1)

    parser.add_argument('--save_uncer', action='store_true')
    parser.add_argument('--uncer_dir',  type=str, default=None)

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    suppress_thr = 300
    conn_comp    = 10
    affine_array = np.array([[-1,0,0,0],[0,-1,0,239],[0,0,1,0],[0,0,0,1]])

    args   = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.save_uncer and not args.mc_drop:
        print('[!] WARNING: --save_uncertainty has no effect without --mc_drop.')

    dataset_paths = {
        'brats2019': '/data/qazisami/dataset/BraTS2019/test',
        'brats2020': '/data/qazisami/dataset/BraTS2020/testing20',
        'brats2021': '/data/qazisami/dataset/BraTS2023-GLI/testing20',
        'brats2023': '/data/qazisami/dataset/BraTS2023-MEN/BraTS_MEN_Train/test',
    }
    assert args.dataset in dataset_paths, f'Unknown dataset: {args.dataset}'
    data_root = dataset_paths[args.dataset]

    # ── directories ───────────────────────────────────────────────────────────
    if args.save_pred:
        pred_dir = f'./predictions/{args.dataset}/{args.model}'
        os.makedirs(pred_dir, exist_ok=True)

    if args.save_uncer and args.mc_drop:
        unc_dir = args.uncer_dir or \
                  f'./uncertainty/{args.dataset}/{args.model}'
        os.makedirs(unc_dir, exist_ok=True)
    else:
        unc_dir = None

    # ── print config ──────────────────────────────────────────────────────────
    print('-' * 100)
    print(f'Dataset : {args.dataset}')
    print(f'Model : {args.model}')
    print(f'Checkpoint : {args.cp}')
    print(f'Labels : {args.labels}   '
          f'Post-process : {args.pp}   '
          f'TTA : {args.tta}   '
          f'Save pred : {args.save_pred}')
    print(f'Thresholds : ET={args.t_et}  ED={args.t_ed}  NCR={args.t_ncr}')
    print(f'MC Dropout : {args.mc_drop}  '
          f'(passes={args.n_passes}  p={args.mc_drop_p})')
    print(f'Save uncert : {args.save_uncer}  (dir={unc_dir})')
    print('-' * 100)

    # ── model ─────────────────────────────────────────────────────────────────
    assert args.model in MODEL_DICT, f'Unknown model: {args.model}'
    model = MODEL_DICT[args.model](
        mc_dropout_p=args.mc_drop_p,
        n_passes=args.n_passes
    ).to(device)

    checkpoint = torch.load(args.cp, map_location=device)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()
    print(f'Model loaded from {args.cp}\n')

    use_tta = TestTimeAugmentation(device) if args.tta else None

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = BraTSDataset(
        data_root,
        train_dataset=args.dataset,
        labels=args.labels,
        smart_crop=not args.labels,
        custom_crops_file=args.custom_crop)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ── metrics accumulators ──────────────────────────────────────────────────
    metrics = {
        r: {'dice': [], 'hd95': [], 'sensitivity': [], 'specificity': [],
            'valid_hd': 0, 'excluded_hd': 0, 'total_cases': 0}
        for r in ['et', 'tc', 'wt']
    }
    case_results = []

    print(f'Evaluating {len(dataset)} cases...')
    start_time = time.time()

    with torch.no_grad():

        # ── WITH labels ───────────────────────────────────────────────────────
        if args.labels:
            for batch_idx, (t1, t1ce, t2, flair, true_mask, filename, crop_coords) \
                    in enumerate(dataloader):

                print(f'[{batch_idx+1}/{len(dataset)}]  {filename[0]}:')
                t1        = t1.to(device)
                t1ce      = t1ce.to(device)
                t2        = t2.to(device)
                flair     = flair.to(device)
                true_mask = true_mask.to(device)

                outputs, already_sigmoid = run_inference(
                    model, t1, t1ce, t2, flair, args, use_tta)

                pred_binary = get_seg_from_outputs(
                    outputs, args.t_et, args.t_ed, args.t_ncr,
                    already_sigmoid=already_sigmoid)

                pred_np    = pred_binary[0].cpu().numpy()
                pred_label = binary_to_brats_label(pred_np)
                true_np    = true_mask[0].cpu().numpy()

                if args.pp:
                    pred_label = post_process_prediction(
                        pred_label,
                        et_threshold=suppress_thr,
                        min_component_size=conn_comp,
                        apply_connected_components=True)

                et_pred, tc_pred, wt_pred = derive_brats_regions(pred_label)
                et_true = (true_np == 3).astype(np.uint8)
                tc_true = ((true_np == 1) | (true_np == 3)).astype(np.uint8)
                wt_true = ((true_np == 1) | (true_np == 2) | (true_np == 3)).astype(np.uint8)

                et_m = calculate_metrics_with_debug(et_pred, et_true, 'ET')
                tc_m = calculate_metrics_with_debug(tc_pred, tc_true, 'TC')
                wt_m = calculate_metrics_with_debug(wt_pred, wt_true, 'WT')

                for region, m in zip(['et', 'tc', 'wt'], [et_m, tc_m, wt_m]):
                    metrics[region]['dice'].append(m[0])
                    metrics[region]['sensitivity'].append(m[2])
                    metrics[region]['specificity'].append(m[3])
                    metrics[region]['total_cases'] += 1
                    if m[1] is not None and not np.isnan(m[1]):
                        metrics[region]['hd95'].append(m[1])
                        metrics[region]['valid_hd'] += 1
                    else:
                        metrics[region]['excluded_hd'] += 1

                avg_dice = (et_m[0] + tc_m[0] + wt_m[0]) / 3
                print(f'  Dice  ET:{et_m[0]:.4f}  TC:{tc_m[0]:.4f}  '
                      f'WT:{wt_m[0]:.4f}  AVG:{avg_dice:.4f}')
                print(f'  HD95  ET:{fmt_hd(et_m[1])}  '
                      f'TC:{fmt_hd(tc_m[1])}  '
                      f'WT:{fmt_hd(wt_m[1])}')

                hd_vals  = [et_m[1], tc_m[1], wt_m[1]]
                valid_hd = [v for v in hd_vals if v is not None and not np.isnan(v)]
                hd95_avg = np.mean(valid_hd) if valid_hd else np.nan

                case_results.append({
                    'case'    : filename[0],
                    'dice_et' : et_m[0], 'dice_tc': tc_m[0], 'dice_wt': wt_m[0],
                    'dice_avg': avg_dice,
                    'hd95_et' : et_m[1], 'hd95_tc': tc_m[1], 'hd95_wt': wt_m[1],
                    'hd95_avg': hd95_avg,
                    'sens_avg': (et_m[2] + tc_m[2] + wt_m[2]) / 3,
                    'spec_avg': (et_m[3] + tc_m[3] + wt_m[3]) / 3,
                })

                if args.save_pred:
                    xs = crop_coords[0].item()
                    ys = crop_coords[1].item()
                    zs = crop_coords[2].item()
                    ze = min(zs + 128, 160)
                    save_label = pred_label.copy()
                    save_label[save_label == 3] = 4
                    full_pred  = np.zeros((240, 240, 160), dtype=np.float32)
                    full_pred[xs:xs+128, ys:ys+128, zs:ze] = save_label[:, :, :ze-zs]
                    nib.save(nib.Nifti1Image(full_pred, affine_array),
                             os.path.join(pred_dir, f'{filename[0]}.nii.gz'))
                    print(f'  Prediction saved  {full_pred.shape}')

                if args.save_uncer and args.mc_drop:
                    save_uncertainty_map(
                        outputs, crop_coords, filename[0], unc_dir, depth=160)

        # ── WITHOUT labels ────────────────────────────────────────────────────
        else:
            for batch_idx, (t1, t1ce, t2, flair, filename, crop_coords) \
                    in enumerate(dataloader):

                print(f'\n[{batch_idx+1}/{len(dataset)}]  {filename[0]}:')
                t1    = t1.to(device)
                t1ce  = t1ce.to(device)
                t2    = t2.to(device)
                flair = flair.to(device)

                outputs, already_sigmoid = run_inference(
                    model, t1, t1ce, t2, flair, args, use_tta)

                pred_binary = get_seg_from_outputs(
                    outputs, args.t_et, args.t_ed, args.t_ncr,
                    already_sigmoid=already_sigmoid)

                pred_np    = pred_binary[0].cpu().numpy()
                pred_label = binary_to_brats_label(pred_np)

                if args.pp:
                    pred_label = post_process_prediction(
                        pred_label,
                        et_threshold=suppress_thr,
                        min_component_size=conn_comp,
                        apply_connected_components=True)

                if args.save_pred:
                    xs = crop_coords[0].item()
                    ys = crop_coords[1].item()
                    zs = crop_coords[2].item()
                    ze = min(zs + 128, 155)
                    save_label = pred_label.copy()
                    save_label[save_label == 3] = 4
                    full_pred  = np.zeros((240, 240, 155), dtype=np.float32)
                    full_pred[xs:xs+128, ys:ys+128, zs:ze] = save_label[:, :, :ze-zs]
                    nib.save(nib.Nifti1Image(full_pred, affine_array),
                             os.path.join(pred_dir, f'{filename[0]}.nii.gz'))
                    print(f'  Prediction saved  {full_pred.shape}')

                if args.save_uncer and args.mc_drop:
                    save_uncertainty_map(
                        outputs, crop_coords, filename[0], unc_dir, depth=155)

    # ── save per-case CSV ─────────────────────────────────────────────────────
    if args.labels and case_results:
        results_dir = f'./results/{extract_path_from_checkpoint(args.cp)}'
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(
            results_dir,
            f'{extract_model_name_from_checkpoint(args.cp)}.csv')

        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Case',
                        'DSC_ET', 'DSC_TC', 'DSC_WT', 'DSC_AVG',
                        'HD95_ET', 'HD95_TC', 'HD95_WT', 'HD95_AVG',
                        'SENS_AVG', 'SPEC_AVG'])
            for r in case_results:
                w.writerow([
                    r['case'],
                    f"{r['dice_et']:.4f}", f"{r['dice_tc']:.4f}",
                    f"{r['dice_wt']:.4f}", f"{r['dice_avg']:.4f}",
                    fmt_hd(r['hd95_et']), fmt_hd(r['hd95_tc']),
                    fmt_hd(r['hd95_wt']), fmt_hd(r['hd95_avg']),
                    f"{r['sens_avg']:.4f}", f"{r['spec_avg']:.4f}",
                ])
        print(f'\nCase metrics saved → {csv_path}')

    # ── final summary ─────────────────────────────────────────────────────────
    if args.labels:
        print('\n' + '-' * 70)
        all_dice = []
        all_hd   = []

        for region in ['et', 'tc', 'wt']:
            m    = metrics[region]
            dice = np.nanmean(m['dice'])        if m['dice']        else 0.0
            hd   = np.nanmean(m['hd95'])        if m['hd95']        else 0.0
            sens = np.nanmean(m['sensitivity']) if m['sensitivity'] else 0.0
            spec = np.nanmean(m['specificity']) if m['specificity'] else 0.0

            all_dice.append(dice)
            all_hd.append(hd)

            print(f'{region.upper()}  '
                  f'Dice:{dice:.4f}  '
                  f'HD95:{hd:.3f}  '
                  f'Sens:{sens:.4f}  '
                  f'Spec:{spec:.4f}  '
                  f'(valid_hd:{m["valid_hd"]}  '
                  f'excluded:{m["excluded_hd"]}  '
                  f'total:{m["total_cases"]})')

        print('-' * 70)
        print(f'AVG  Dice:{np.mean(all_dice):.4f}  HD95:{np.mean(all_hd):.3f}')
        print('-' * 70)

    total_time = (time.time() - start_time) / 60
    print(f'\nTesting complete. Total time: {total_time:.2f} minutes')
    print(f'Checkpoint: {args.cp}')


if __name__ == '__main__':
    main()