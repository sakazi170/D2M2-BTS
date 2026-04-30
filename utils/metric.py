from medpy.metric.binary import dc, hd95, sensitivity, specificity
import numpy as np


def calculate_metrics_with_debug(pred_mask, true_mask, region_name):

    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()

    # both empty → perfect scores, HD95=0
    if pred_sum == 0 and true_sum == 0:
        print(f"  Debug - {region_name}: both empty → perfect scores")
        return 1.0, 0.0, 1.0, 1.0

    # one empty → Dice=0, HD95=nan (excluded from average like SOTA)
    elif pred_sum == 0 and true_sum > 0:
        print(f"  Debug - {region_name}: pred empty, GT={true_sum} → HD95 excluded")
        return 0.0, np.nan, 0.0, 1.0

    elif true_sum == 0 and pred_sum > 0:
        print(f"  Debug - {region_name}: GT empty, pred={pred_sum} → HD95 excluded")
        return 0.0, np.nan, 1.0, 0.0

    # both non-empty → compute normally
    dice = dc(pred_mask, true_mask)
    sens = sensitivity(pred_mask, true_mask)
    spec = specificity(pred_mask, true_mask)

    try:
        hausdorff = hd95(pred_mask, true_mask)
        if hausdorff == float('inf'):
            hausdorff = np.nan
            print(f"  Debug - {region_name}: HD95=inf → excluded")
    except Exception as e:
        hausdorff = np.nan
        print(f"  Debug - {region_name}: HD95 error ({e}) → excluded")

    return dice, hausdorff, sens, spec