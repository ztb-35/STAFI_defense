"""
CrossFire defense evaluation on pre-flipped Supercombo models using recurrent dataset.
Computes delta between defense-repaired and clean models for all metrics.
"""

import sys
import gc
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from openpilot_torch import OpenPilotModel
from data import Comma2k19SequenceRecurrentDataset
from torch.utils.data import DataLoader
from crossfire_adapt import crossfire_setup, crossfire_detect_repair


distance_func = nn.CosineSimilarity(dim=2)
cls_loss_fn   = nn.CrossEntropyLoss()
reg_loss_fn   = nn.SmoothL1Loss(reduction='none')


def make_supercombo_inputs(batch, device):
    seq_imgs   = batch['seq_input_img'].to(device, non_blocking=True)
    seq_labels = batch['seq_future_poses'].to(device, non_blocking=True)
    B, T, C, H, W = seq_imgs.shape
    if C == 6:
        seq_imgs = torch.cat([seq_imgs, seq_imgs], dim=2)
    imgs12   = seq_imgs
    desire   = torch.zeros((B, 8), device=device)
    traffic  = torch.tensor([[1., 0.]], device=device).repeat(B, 1)
    h0       = torch.zeros((B, 512), device=device)
    traj_gt  = seq_labels
    return imgs12, desire, traffic, h0, traj_gt


def build_loader(txt_path, data_root, data_length, frame_stream_length, mode, batch_size, device):
    ds = Comma2k19SequenceRecurrentDataset(
        txt_path, data_root, data_length, frame_stream_length,
        mode, use_memcache=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                      pin_memory=(device.type == 'cuda'))


@torch.no_grad()
def run_inference(model, inputs):
    model.eval()
    return model(*inputs[:4]).detach()


def load_model(state_dict, device=None):
    m = OpenPilotModel()
    m.load_state_dict(state_dict)
    m.eval()
    if device is not None:
        m.to(device)
    return m


# ── Trajectory parsing & metric helpers ─────────────────────────────────────────

def parse_output(y: torch.Tensor) -> dict:
    B = y.shape[0]
    pl       = y[:, :5 * 991].view(B, 5, 991)
    pf       = pl[:, :, :-1]
    lead_p   = y[:, 6064]
    traj     = pf.view(B, 5, 2, 33, 15)[:, :, 0, :, :3]
    velocity = pf.view(B, 5, 2, 33, 15)[:, :, 1, :, 3:6]
    return {'traj': traj, 'velocity': velocity, 'lead_p': lead_p}


def select_best(parsed: dict, traj_gt: torch.Tensor) -> dict:
    traj  = parsed['traj']
    pend  = traj[:, :, 17, 0]
    gend  = traj_gt[:, 17, 0].unsqueeze(1).expand(-1, 5)
    d     = (pend - gend) / (traj_gt[:, 17, 0].abs().unsqueeze(1) + 1)
    idx   = d.argmin(dim=1)
    rows  = torch.arange(len(idx), device=idx.device)
    return {
        'best_traj':     traj[rows, idx],
        'best_velocity': parsed['velocity'][rows, idx],
        'lead_p':        parsed['lead_p'],
    }


def compute_delta(best_model: dict, best_base: dict, metric: str) -> float:
    fa  = best_model['best_traj']
    fb  = best_base['best_traj']
    va  = best_model['best_velocity']
    vb  = best_base['best_velocity']
    lpa = best_model['lead_p']
    lpb = best_base['lead_p']

    if metric == 'speed up':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) - 2 * (fa[:, 10, 1] - fb[:, 10, 1]).abs()
    elif metric == 'speed up 2':
        delta = lpb - lpa
    elif metric == 'velocity':
        delta = vb[:, 10, 0] - va[:, 10, 0]
    elif metric == 'slowing down':
        delta = fb[:, 10, 0] - fa[:, 10, 0]
    elif metric == 'left steering':
        delta = fa[:, 10, 1] - fb[:, 10, 1]
    elif metric == 'right steering':
        delta = fb[:, 10, 1] - fa[:, 10, 1]
    elif metric == 'speed up & left steering':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) / 9.94 + (fa[:, 10, 1] - fb[:, 10, 1])
    elif metric == 'speed up & right steering':
        delta = (fa[:, 10, 0] - fb[:, 10, 0]) / 9.94 + (fb[:, 10, 1] - fa[:, 10, 1])
    else:
        delta = fa[:, 10, 0] - fb[:, 10, 0]

    mask = torch.isfinite(delta)
    if not mask.any():
        return float('nan')
    return float(delta[mask].mean().item())


ALL_METRICS = [
    'speed up', 'speed up 2', 'velocity', 'slowing down',
    'left steering', 'right steering', 'speed up & left steering', 'speed up & right steering'
]


def collect_test_inputs(val_loader, device):
    test_inputs = []
    for batch in val_loader:
        imgs12, desire, traffic, h0, traj_gt = make_supercombo_inputs(batch, device)
        test_inputs.append((imgs12, desire, traffic, h0, traj_gt))
    return test_inputs


def process_sequence(model, test_inputs, device):
    all_bests = []
    for imgs12, desire, traffic, h0, traj_gt in test_inputs:
        B, T, C, H, W = imgs12.shape
        h0_current = h0.clone()
        batch_bests = []
        for t in range(T):
            imgs_t = imgs12[:, t, :, :, :]
            gt_t   = traj_gt[:, t, :, :]
            with torch.no_grad():
                out_t  = run_inference(model, (imgs_t, desire, traffic, h0_current))
                parsed = parse_output(out_t)
                best   = select_best(parsed, gt_t)
            batch_bests.append(best)
            h0_current = out_t[:, -512:]
        all_bests.append(batch_bests)
    return all_bests


def compute_metric_deltas(model_bests, clean_bests, metric):
    delta_vals = []
    for batch_idx in range(len(clean_bests)):
        for t in range(len(clean_bests[batch_idx])):
            delta = compute_delta(model_bests[batch_idx][t], clean_bests[batch_idx][t], metric)
            if not np.isnan(delta):
                delta_vals.append(delta)
    return float(np.nanmean(delta_vals)) if delta_vals else float('nan')


def evaluate(args):
    base      = Path(__file__).parent
    clean_path = base / 'openpilot_model' / 'supercombo_torch_weights.pth'
    model_dir  = base / 'openpilot_model'
    out_dir    = base / 'crossfire_results'
    out_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 1: Load clean model')
    print('=' * 65)
    clean_sd    = torch.load(clean_path, map_location='cpu')
    clean_model = load_model(clean_sd, device)
    print(f'  {len(clean_sd)} parameter tensors loaded')

    print('\nSTEP 2: Build recurrent val loader')
    val_loader = build_loader(args.val_txt, args.data_root, args.data_length,
                              args.frame_stream_length, 'val', args.batch_size, device)
    print(f'  val loader samples: {len(val_loader.dataset)}')

    print('\nSTEP 3: Collect full recurrent test inputs')
    test_inputs = collect_test_inputs(val_loader, device)
    print(f'  collected {len(test_inputs)} video sample(s)')
    if not test_inputs:
        print('  [WARN] val loader is empty; aborting.')
        return

    print('\nSTEP 4: CrossFire setup (on clean model)')
    protected_sd, metadata = crossfire_setup(
        clean_sd      = clean_sd,
        neuropots_pct = args.npc,
        gamma         = args.npg,
        depth_lambda  = args.lambda_d,
        pruning_ratio = args.prune,
        model         = None,
        data_samples  = None,
    )
    print(f'  neuropots_pct={args.npc}  gamma={args.npg}  '
          f'depth_lambda={args.lambda_d}  prune_keep={args.prune}')

    print('\nSTEP 5: Process clean baselines')
    clean_bests = process_sequence(clean_model, test_inputs, device)
    print(f'  Processed {len(test_inputs)} sample(s), {len(clean_bests[0])} frames each')

    print('\nSTEP 6: Evaluate attacked models')
    model_files = sorted([f for f in model_dir.glob('*.pth')
                          if f.name != 'supercombo_torch_weights.pth'])
    rows = []

    for model_path in model_files:
        model_name = model_path.stem
        print('\n' + '─' * 65)
        print(f'Attacked Model : {model_name}')

        attacked_sd    = torch.load(model_path, map_location='cpu')
        attacked_model = load_model(attacked_sd, device)

        atk_bests     = process_sequence(attacked_model, test_inputs, device)
        attack_deltas = {m: compute_metric_deltas(atk_bests, clean_bests, m)
                         for m in ALL_METRICS}
        del attacked_model, atk_bests
        torch.cuda.empty_cache(); gc.collect()

        # ── CrossFire repair (no oracle: neuropot restore + OOD bit correction) ──
        repaired_sd_noorac, report_noorac = crossfire_detect_repair(
            attacked_sd = attacked_sd,
            metadata    = metadata,
            clean_sd    = None,
        )
        detected = len(report_noorac['attacked_params']) > 0
        m_noorac = load_model(repaired_sd_noorac, device)
        noorac_bests  = process_sequence(m_noorac, test_inputs, device)
        noorac_deltas = {m: compute_metric_deltas(noorac_bests, clean_bests, m)
                         for m in ALL_METRICS}
        del m_noorac, repaired_sd_noorac, noorac_bests
        torch.cuda.empty_cache(); gc.collect()

        # ── CrossFire oracle repair (clean model available) ──
        repaired_sd_orac, report_orac = crossfire_detect_repair(
            attacked_sd = attacked_sd,
            metadata    = metadata,
            clean_sd    = clean_sd,
        )
        m_orac = load_model(repaired_sd_orac, device)
        orac_bests    = process_sequence(m_orac, test_inputs, device)
        oracle_deltas = {m: compute_metric_deltas(orac_bests, clean_bests, m)
                         for m in ALL_METRICS}
        del m_orac, repaired_sd_orac, orac_bests, attacked_sd
        torch.cuda.empty_cache(); gc.collect()

        print(f'  Detection : {"YES ✓" if detected else "NO ✗"}  '
              f'({len(report_noorac["attacked_params"])} layers attacked)')
        print(f'  Flips detected/repaired : '
              f'{report_noorac["total_detected"]}/{report_noorac["total_repaired"]}')
        for m in ALL_METRICS[:3]:
            print(f'    {m:30s}: {attack_deltas[m]:+.4f}')

        rows.append({
            'model_name':         model_name,
            'detected':           detected,
            'attacked_params':    report_noorac['attacked_params'],
            'fully_repaired':     report_noorac['fully_repaired'],
            'attack_deltas':      attack_deltas,
            'noorac_repair_deltas': noorac_deltas,
            'oracle_repair_deltas': oracle_deltas,
        })

    print('\n' + '=' * 65)
    print('SUMMARY')
    print('=' * 65)
    if rows:
        det_rate = np.mean([r['detected'] for r in rows]) * 100
        print(f'  Detection rate: {det_rate:.0f}%')
        for m in ALL_METRICS:
            avg_attack = np.nanmean([r['attack_deltas'][m] for r in rows])
            avg_noorac = np.nanmean([r['noorac_repair_deltas'][m] for r in rows])
            avg_oracle = np.nanmean([r['oracle_repair_deltas'][m] for r in rows])
            print(f'  {m:30s}: Attack {avg_attack:+.4f} | '
                  f'NoOrac {avg_noorac:+.4f} | Oracle {avg_oracle:+.4f}')

    out_path = out_dir / 'crossfire_eval_all_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nFull results → {out_path}')


def main():
    base = Path(__file__).parent
    p = argparse.ArgumentParser(description='CrossFire defense evaluation (recurrent)')
    p.add_argument('--batch_size',         type=int,   default=1)
    p.add_argument('--data_length',        type=int,   default=100)
    p.add_argument('--frame_stream_length',type=int,   default=8)
    p.add_argument('--npc',                type=float, default=0.10,
                   help='Neuropot percentage (default 0.10)')
    p.add_argument('--npg',                type=float, default=1.30,
                   help='Base gamma for neuropot scaling (default 1.30)')
    p.add_argument('--lambda_d',           type=float, default=1.10,
                   help='Depth lambda for neuropot scaling (default 1.10)')
    p.add_argument('--prune',              type=float, default=0.75,
                   help='L1 prune keep-ratio (default 0.75)')
    p.add_argument('--val_txt',  type=str,
                   default=str(base / 'data' / 'comma2k19_val.txt'))
    p.add_argument('--data_root', type=str,
                   default=str(base / 'data' / 'comma2k19') + '/')
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
