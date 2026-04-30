import os
import tqdm
import argparse
import setproctitle
import time
import csv
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

from utils.train_data_loader import SubjectReader
from utils.iterator import set_random_seed, CosineAnnealingWithWarmUp
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from bts import BrainTumorSegNet, BrainTumorLoss
from ablation_models import (AblationV1, AblationV2, AblationV3,
                              AblationV4, AblationV4_1, AblationV4_2, AblationV5, AblationV5_1, AblationV5_2, AblationV6,
                              AblationLoss)


# ── Model and loss registries ─────────────────────────────────────────────────
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

# Full model uses BrainTumorLoss (supports epistemic loss).
# All ablation models use AblationLoss (Dice + CE only, epi always 0).
LOSS_DICT = {
    'bts' : BrainTumorLoss,
    'v1': AblationLoss,
    'v2': AblationLoss,
    'v3': AblationLoss,
    'v4': AblationLoss,
    'v4_1': AblationLoss,
    'v4_2': AblationLoss,
    'v5': AblationLoss,
    'v5_1': AblationLoss,
    'v5_2': AblationLoss,
    'v6': AblationLoss,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        type=str,   default='bts')
    parser.add_argument('--dataset',      type=str,   default='brats2020')
    parser.add_argument('--gpu',          type=str,   default='2')
    parser.add_argument('--ngpu',         type=int,   default=1)
    parser.add_argument('--ncpu',         type=int,   default=4)
    parser.add_argument('--bsize',        type=int,   default=1)
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--interval',     type=int,   default=1)
    parser.add_argument('--trainset',     action='store_true')
    parser.add_argument('--mixed',        action='store_true')
    parser.add_argument('--benchmark',    action='store_true', default=False)
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--cp',           type=str,   default=None)

    # loss weights
    parser.add_argument('--lam_dice',     type=float, default=1.0)
    parser.add_argument('--lam_ce',       type=float, default=0.5)
    parser.add_argument('--lam_epi',      type=float, default=0.1,
                        help='Weight for epistemic loss (bts only)')

    # MC Dropout
    parser.add_argument('--mc_dropout_p', type=float, default=0.1)
    parser.add_argument('--n_passes',     type=int,   default=5)
    parser.add_argument('--mc_train_T',   type=int,   default=3,
                        help='MC passes per training step (bts only)')
    parser.add_argument('--no_epi_loss',  action='store_true',
                        help='Disable epistemic loss (bts only)')

    return parser.parse_args()


def main():
    start_time = time.time()
    args       = parse_args()
    torch.backends.cudnn.benchmark = args.benchmark

    assert args.model in MODEL_DICT, \
        f'Unknown model: {args.model}. Choose from: {list(MODEL_DICT.keys())}'

    # ── dataset paths ─────────────────────────────────────────────────────────
    dataset_paths = {
        'brats2019': '/data/qazisami/dataset/BraTS2019/train',
        'brats2020': '/data/qazisami/dataset/BraTS2020/training80',
        'brats2021': '/data/qazisami/dataset/BraTS2023-GLI/BraTS2023-GLI-TrainingData',
        'brats2023': '/data/qazisami/dataset/BraTS2023-MEN/BraTS_MEN_Train/train',
    }
    assert args.dataset in dataset_paths, f'Unknown dataset: {args.dataset}'
    data_root = dataset_paths[args.dataset]

    # ── whether this run uses the epistemic loss ───────────────────────────────
    # Ablation models never use it regardless of flags — their forward_mc_train()
    # always returns preds_stack=None and AblationLoss ignores preds_stack.
    is_full_model = args.model == 'bts'
    use_epi_loss  = is_full_model and not args.no_epi_loss

    # ── save dirs ─────────────────────────────────────────────────────────────
    if args.no_epi_loss:
        run_tag      = f'{args.model}_d{args.lam_dice}_ce{args.lam_ce}'
    else:
        run_tag = f'{args.model}_d{args.lam_dice}_ce{args.lam_ce}_epi{args.lam_epi}'
    save_dir     = f'./checkpoints/{args.dataset}/{run_tag}'
    save_dir_log = f'./logs/{args.dataset}/{run_tag}'
    os.makedirs(save_dir,     exist_ok=True)
    os.makedirs(save_dir_log, exist_ok=True)

    # ── hyperparameters ───────────────────────────────────────────────────────
    seed           = 42
    lr             = 1e-3
    decay          = 1e-5
    warm_up_epochs = 5
    max_lr_epochs  = 50
    patch_size     = 128

    print('-' * 100)
    print(f'Dataset      : {args.dataset}')
    print(f'Model        : {args.model}')
    print(f'Loss weights — Dice:{args.lam_dice}  CE:{args.lam_ce}  '
          f'Epi:{args.lam_epi}  (enabled:{use_epi_loss})')
    if is_full_model:
        print(f'MC train     — T={args.mc_train_T} passes per step')
    else:
        print(f'MC train     — disabled (ablation model, epi loss not used)')
    print('-' * 100)

    set_random_seed(seed=seed, benchmark=args.benchmark)

    # ── dataloader ────────────────────────────────────────────────────────────
    subject_reader = SubjectReader(
        train_dir=data_root,
        train_dataset=args.dataset,
        training_size=patch_size)

    assert args.trainset, 'Pass --trainset flag to use training data'
    trainset = subject_reader.get_trainset()

    train_loader = DataLoader(
        trainset,
        batch_size=args.bsize * args.ngpu,
        shuffle=True,
        num_workers=args.ncpu,
        multiprocessing_context='spawn')

    # ── device ────────────────────────────────────────────────────────────────
    gpu_ids = [int(g) for g in args.gpu.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── model ─────────────────────────────────────────────────────────────────
    model = MODEL_DICT[args.model](
        mc_dropout_p=args.mc_dropout_p,
        n_passes=args.n_passes
    ).to(device)

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = LOSS_DICT[args.model](
        lam_dice=args.lam_dice,
        lam_ce=args.lam_ce,
        lam_epi=args.lam_epi)

    # ── optimizer and scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=decay)

    scheduler = CosineAnnealingWithWarmUp(
        optimizer,
        cycle_steps=args.epochs * len(train_loader),
        max_lr_steps=max_lr_epochs * len(train_loader),
        max_lr=lr,
        min_lr=lr / 1000,
        warmup_steps=warm_up_epochs * len(train_loader))

    # ── checkpoint resume ─────────────────────────────────────────────────────
    start_epoch = 0
    if args.cp:
        ckpt = torch.load(args.cp, map_location=device)
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f'Resumed from epoch {start_epoch}')

    if args.ngpu > 1:
        print('Multi-GPU not supported.')
        sys.exit()

    scaler = torch.amp.GradScaler('cuda')

    # ── CSV logger ────────────────────────────────────────────────────────────
    date_str = datetime.now().strftime('%Y%m%d')
    csv_path = os.path.join(save_dir_log, f'{run_tag}_{date_str}.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'avg_loss', 'dice', 'ce', 'epi'])

    # ── training loop ─────────────────────────────────────────────────────────
    best_loss          = 2.0
    previous_best_name = None
    total_steps        = len(train_loader)

    print(f'Training started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        setproctitle.setproctitle(f'{args.model}: {epoch}/{args.epochs}')
        print(f'\nEpoch {epoch}/{args.epochs}')

        model.train()
        epoch_loss  = 0.0
        comp_totals = {'dice': 0.0, 'ce': 0.0, 'epi': 0.0}

        loader = tqdm.tqdm(train_loader) if args.verbose else train_loader

        for step, batch in enumerate(loader):
            t1      = batch['t1'].to(device)
            t1ce    = batch['t1ce'].to(device)
            t2      = batch['t2'].to(device)
            flair   = batch['flair'].to(device)
            targets = batch['label'].to(device)

            optimizer.zero_grad()

            if args.mixed:
                with torch.amp.autocast('cuda'):
                    if use_epi_loss:
                        outputs, preds_stack = model.forward_mc_train(
                            t1, t1ce, t2, flair, T=args.mc_train_T)
                    else:
                        outputs = model.forward(t1, t1ce, t2, flair)
                        preds_stack = None

                    loss, components = criterion(outputs, targets, preds_stack)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if use_epi_loss:
                    outputs, preds_stack = model.forward_mc_train(
                        t1, t1ce, t2, flair, T=args.mc_train_T)
                else:
                    outputs = model.forward(t1, t1ce, t2, flair)
                    preds_stack = None

                loss, components = criterion(outputs, targets, preds_stack)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            for k in comp_totals:
                comp_totals[k] += components[k]

            if args.verbose:
                loader.set_postfix_str(
                    f'lr:{optimizer.param_groups[0]["lr"]:.2e} '
                    f'loss:{loss.item():.4f} '
                    f'dice:{components["dice"]:.4f} '
                    f'ce:{components["ce"]:.4f} '
                    f'epi:{components["epi"]:.4f}')

        # ── epoch summary ─────────────────────────────────────────────────────
        avg_loss = epoch_loss / total_steps
        avg_comp = {k: v / total_steps for k, v in comp_totals.items()}
        duration = (time.time() - epoch_start) / 60

        print(f'Avg loss:{avg_loss:.4f}  '
              f'dice:{avg_comp["dice"]:.4f}  '
              f'ce:{avg_comp["ce"]:.4f}  '
              f'epi:{avg_comp["epi"]:.4f}  '
              f'time:{duration:.2f}m')

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                f'{avg_loss:.5f}',
                f'{avg_comp["dice"]:.5f}',
                f'{avg_comp["ce"]:.5f}',
                f'{avg_comp["epi"]:.5f}',
            ])

        # ── save best ─────────────────────────────────────────────────────────
        if avg_loss < best_loss:
            old_name  = previous_best_name
            best_loss = avg_loss
            ckpt = {
                'net'      : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch'    : epoch,
                'loss'     : best_loss,
            }
            new_name = f'{args.model}_best_{epoch}_{best_loss:.4f}.pkl'
            try:
                torch.save(ckpt, os.path.join(save_dir, new_name))
                if old_name and os.path.exists(os.path.join(save_dir, old_name)):
                    os.remove(os.path.join(save_dir, old_name))
                previous_best_name = new_name
                print(f'Best model saved')
            except Exception as e:
                print(f'  [!] Save error: {e}')

        # ── periodic checkpoint ───────────────────────────────────────────────
        if epoch % 25 == 0:
            ckpt = {
                'net'      : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch'    : epoch,
                'loss'     : avg_loss,
            }
            torch.save(ckpt, os.path.join(
                save_dir, f'{args.model}_{epoch}_{avg_loss:.4f}.pkl'))
            print(f'Checkpoint saved at epoch {epoch}')

    total_time = (time.time() - start_time) / 60
    print(f'\nTraining complete. Total time: {total_time:.2f} minutes')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()