#!/usr/bin/env python
"""Replay nanotron's LR scheduler over the real 1.5B and 3B segment chains; tabulate and plot it (analysis only).

    $PRETRAIN_PY deploy/skipjack_3b_600m/lr_schedule_report.py replay --out <dir>   # torch + nanotron config (CPU)
    $DATA_PY     deploy/skipjack_3b_600m/lr_schedule_report.py plot   --out <dir>   # matplotlib

replay mirrors the trainer exactly, per optimizer update:
  * build: lr_scheduler_builder(optimizer, cfg LR args, train_steps) - source extracted from src/nanotron/helpers.py
    (that module imports flash-attn, which needs a GPU at import time);
  * resume (load_lr_scheduler: true): load_state_dict(saved) then _initial_step() (serialize/optimizer.py:369-370);
  * loop it = last_step+1 .. train_steps: the update uses the current group lr; optimizer.step(); lr_scheduler.step()
    (trainer.py:726, 731); the checkpoint after `it` stores the scheduler state at that point.
Two semantics are produced from the same configs:
  implemented    exactly the above (PyTorch's _initial_step increments last_epoch, so each resume shifts the index by
                 one; the shift accumulates over chained trunk segments; requeues would add more);
  uninterrupted  the resume restores last_epoch without the extra step - identical to one continuous scheduler, i.e.
                 update n uses lambda(n-1). This is what the paper's description implies and what a resume fix gives.
1.5B configs are read (never written) from the active baseline checkout; 3B from this worktree.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
T = 2_097_152
B15 = '/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys/configs/1.5B-baseline-seed42/filled/raw_diversity_oriented_seed42_{}.yaml'
B3 = str(WT / 'configs/3B-rewrite-600m/seed42/3b_diversity_oriented_seed42_{}.yaml')
GRIDS = {
    '1.5B': dict(cfg=B15, trunk_ends=[4292, 8583, 12875], bps=[4292, 8583, 12875], hs=[4768, 9537, 14305],
                 nominal=[10e9, 20e9, 30e9], W=500, peak=5e-4),
    '3B': dict(cfg=B3, trunk_ends=[4292, 8583, 12875, 17166, 21458, 25749], bps=[8583, 17166, 25749], hs=[9537, 19073, 28610],
               nominal=[20e9, 40e9, 60e9], W=1000, peak=4e-4),
}


def _builder():
    import ast, functools, logging, math, typing
    import torch
    sys.path.insert(0, str(WT / 'src'))
    from nanotron.config import LRSchedulerArgs
    tree = ast.parse((WT / 'src/nanotron/helpers.py').read_text())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lr_scheduler_builder'][0]
    ns = {'LambdaLR': torch.optim.lr_scheduler.LambdaLR, 'math': math, 'log_rank': lambda *a, **k: None,
          'logger': logging.getLogger(), 'logging': logging, 'Optimizer': torch.optim.Optimizer,
          'LRSchedulerArgs': LRSchedulerArgs, 'partial': functools.partial}
    ns.update({k: getattr(typing, k) for k in ('Optional', 'List', 'Dict', 'Union', 'Any', 'Tuple', 'Callable')})
    exec(compile(ast.Module([fn], []), 'nanotron.helpers.lr_scheduler_builder', 'exec'), ns)
    return ns['lr_scheduler_builder'], LRSchedulerArgs


def replay(out: Path):
    import warnings
    import numpy as np
    import torch
    import yaml
    warnings.filterwarnings('ignore')
    build, LRArgs = _builder()

    class Opt(torch.optim.SGD):
        def get_base_optimizer(self):
            return self

    def segment(path, state, first, fixed):
        c = yaml.safe_load(open(path))
        a, steps = LRArgs(**c['optimizer']['learning_rate_scheduler']), c['tokens']['train_steps']
        opt = Opt([torch.nn.Parameter(torch.zeros(1))], lr=a.learning_rate)
        sch = build(opt, a, total_training_steps=steps)
        if c['checkpoints']['load_lr_scheduler']:
            sch.load_state_dict(state)
            if fixed:
                sch.last_epoch -= 1
            sch._initial_step()
        used = []
        for _ in range(first, steps + 1):
            used.append(opt.param_groups[0]['lr'])
            opt.step()
            sch.step()
        return np.array(used), sch.state_dict(), sch.last_epoch, c

    arrays, meta = {}, {}
    for g, G in GRIDS.items():
        for sem, fixed in (('implemented', False), ('uninterrupted', True)):
            trunk, state, saved, states, prev = [], None, {}, {}, 0
            for i, end in enumerate(G['trunk_ends'], 1):
                used, state, le, _ = segment(G['cfg'].format(f'trunk{i}'), state, prev + 1, fixed)
                trunk.append(used)
                saved[end], states[end], prev = le, state, end
            arrays[f'{g}|{sem}|trunk'] = np.concatenate(trunk)      # index 0 = update 1
            for k, b in enumerate(G['bps'], 1):
                used, _, _, c = segment(G['cfg'].format(f'ep{k}'), states[b], b + 1, fixed)
                arrays[f'{g}|{sem}|ep{k}'] = used                   # index 0 = update b+1
                assert c['checkpoints']['resume_checkpoint_path'].endswith(f'/{b}') and c['checkpoints']['load_optimizer']
            meta[f'{g}|{sem}'] = {'scheduler_last_epoch_at_trunk_saves': saved}
    # the uninterrupted semantics must equal one continuous scheduler per branch config: update n uses lambda(n-1)
    for g, G in GRIDS.items():
        for k, (b, h) in enumerate(zip(G['bps'], G['hs']), 1):
            c = yaml.safe_load(open(G['cfg'].format(f'ep{k}')))
            a = LRArgs(**c['optimizer']['learning_rate_scheduler'])
            opt = Opt([torch.nn.Parameter(torch.zeros(1))], lr=a.learning_rate)
            lam = build(opt, a, total_training_steps=h).lr_lambdas[0]
            cont = np.array([lam(n - 1) * a.learning_rate for n in range(1, h + 1)])
            path = np.concatenate([arrays[f'{g}|uninterrupted|trunk'][:b], arrays[f'{g}|uninterrupted|ep{k}']])
            assert np.allclose(cont, path, rtol=0, atol=1e-15), (g, k)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'lr_per_update.npz', **{k.replace('|', '__'): v for k, v in arrays.items()})
    table, samples = tabulate(arrays, meta)
    (out / 'lr_samples.json').write_text(json.dumps({'meta': meta, 'samples': samples}, indent=1))
    with open(out / 'lr_schedule_table.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(json.dumps(table, indent=1))


def tabulate(arrays, meta):
    rows, samples = [], {}
    for g, G in GRIDS.items():
        W, peak, hs, bps = G['W'], G['peak'], G['hs'], G['bps']
        for k, (b, h) in enumerate(zip(bps, hs), 1):
            prev_h = hs[k - 2] if k > 1 else 0
            r = {'model': g, 'horizon': f'ep{k}', 'nominal_tokens': int(G['nominal'][k - 1]), 'final_step': h,
                 'processed_tokens': h * T, 'peak_lr': peak, 'warmup_steps': W, 'warmup_tokens': W * T,
                 'warmup_pct_of_horizon': round(100 * W / h, 3), 'branch_source_ckpt': b,
                 'decay_updates_after_ckpt': h - b, 'decay_tokens': (h - b) * T,
                 'decay_pct_of_horizon': round(100 * (h - b) / h, 3),
                 'nominal_epoch_interval_steps': h - prev_h,
                 'decay_pct_of_last_nominal_epoch': round(100 * (h - b) / (h - prev_h), 3),
                 'decay_start_fraction_of_horizon': round(b / h, 5),
                 'decay_start_fraction_of_last_nominal_epoch': round((b - prev_h) / (h - prev_h), 5)}
            for sem in ('uninterrupted', 'implemented'):
                lr = list(arrays[f'{g}|{sem}|trunk'][:b]) + list(arrays[f'{g}|{sem}|ep{k}'])   # lr[n-1] = update n
                n_peak = [n for n in range(1, h + 1) if abs(lr[n - 1] - peak) < 1e-15]
                first_low = min(n for n in range(b + 1, h + 1) if lr[n - 1] < peak - 1e-15)
                tag = 'unint' if sem == 'uninterrupted' else 'impl'
                r.update({f'{tag}_first_update_at_peak': n_peak[0], f'{tag}_last_update_at_peak': n_peak[-1],
                          f'{tag}_updates_at_peak': len(n_peak), f'{tag}_first_update_below_peak': first_low,
                          f'{tag}_lr_last_update': lr[h - 1], f'{tag}_updates_at_lr0_after_warmup': sum(1 for n in range(W + 1, h + 1) if lr[n - 1] == 0.0),
                          f'{tag}_trunk_updates_below_peak_before_ckpt': sum(1 for n in range(W + 1, b + 1) if lr[n - 1] < peak - 1e-15)})
                pts = sorted({1, 2, W, W + 1, W + 2, b - 1, b, b + 1, b + 2, b + 3, h - 3, h - 2, h - 1, h})
                samples[f'{g}|{sem}|ep{k}'] = {n: lr[n - 1] for n in pts}
            rows.append(r)
    return rows, samples


def plot(out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    d = np.load(out / 'lr_per_update.npz')
    A = {k.replace('__', '|'): d[k] for k in d.files}
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    TRUNK = '#52514e'
    BR = {'ep1': '#2a78d6', 'ep2': '#eb6834', 'ep3': '#1baf7a'}   # reference palette slots 1-3, fixed order
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})

    def curves(ax, g, sem, xs='tokens', norm=False):
        G = GRIDS[g]
        peak, last_b = G['peak'], G['bps'][-1]
        sc = (lambda n: n * T / 1e9) if xs == 'tokens' else (lambda n: n / G['hs'][-1])
        ys = (lambda v: v / peak) if norm else (lambda v: v * 1e4)
        tr = A[f'{g}|{sem}|trunk'][:last_b]
        n = np.arange(1, len(tr) + 1)
        ax.plot(sc(n), ys(tr), color=TRUNK, lw=2, label='shared trunk (constant after warmup)', zorder=3)
        for k, (b, h) in enumerate(zip(G['bps'], G['hs']), 1):
            e = A[f'{g}|{sem}|ep{k}']
            nb = np.arange(b, h + 1)
            yb = np.concatenate([[tr[b - 1]], e])
            ax.plot(sc(nb), ys(yb), color=BR[f'ep{k}'], lw=2, label=f'ep{k} branch: ckpt {b} → step {h}', zorder=4)
            ax.plot([sc(b)], [ys(tr[b - 1])], marker='v', ms=8, color=BR[f'ep{k}'], mec=SURF, mew=2, zorder=5)
            ax.plot([sc(h)], [ys(e[-1])], marker='o', ms=8, color=BR[f'ep{k}'], mec=SURF, mew=2, zorder=5)
        ax.axvline(sc(G['W']), color=INK2, lw=0.8, ls=':', zorder=2)
        ax.annotate(f'warmup ends\n(update {G["W"] + 1} = first at peak)', (sc(G['W']), ys(peak) * 0.55), xytext=(6, 0),
                    textcoords='offset points', fontsize=8, color=INK2)

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 7.2), sharey=False)
    for ax, g in zip(axs, GRIDS):
        G = GRIDS[g]
        curves(ax, g, 'uninterrupted')
        ax.set_title(f'{g}: peak {G["peak"]:.0e}, warmup {G["W"]} updates, global batch 2,097,152 tokens/update',
                     loc='left', fontsize=10, color=INK)
        ax.set_ylabel('learning rate (×1e-4)')
        ax.set_xlabel('processed tokens (billions) = optimizer step × 2,097,152')
        ax.set_ylim(-0.2, G['peak'] * 1e4 * 1.12)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle('Trunk + branch annealing (scheduler replay, uninterrupted semantics). ▼ branch source checkpoint, ● endpoint',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'lr_schedule_absolute.png', dpi=160)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={'width_ratios': [2.2, 1, 1]})
    ax = axs[0]
    for g, ls in (('1.5B', '-'), ('3B', (0, (4, 3)))):
        G = GRIDS[g]
        n = np.arange(1, G['bps'][-1] + 1)
        ax.plot(n / G['hs'][-1], A[f'{g}|uninterrupted|trunk'][:len(n)] / G['peak'], color=TRUNK, lw=2, ls=ls)
        for k, (b, h) in enumerate(zip(G['bps'], G['hs']), 1):
            nb = np.arange(b + 1, h + 1)
            ax.plot(nb / G['hs'][-1], A[f'{g}|uninterrupted|ep{k}'] / G['peak'], color=BR[f'ep{k}'], lw=2, ls=ls)
    ax.plot([], [], color=INK2, lw=2, ls='-', label='1.5B (solid)')
    ax.plot([], [], color=INK2, lw=2, ls=(0, (4, 3)), label='3B (dashed)')
    for k in (1, 2, 3):
        ax.plot([], [], color=BR[f'ep{k}'], lw=2, label=f'ep{k}')
    ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.79, 0.04), labelcolor=INK2)
    ax.set_title('Normalized: the 1.5B and 3B schedules coincide', loc='left', fontsize=10, color=INK)
    ax.set_xlabel('optimizer step / final step of the 3rd horizon')
    ax.set_ylabel('LR / peak LR')
    for ax, g, k, title in ((axs[1], '3B', 3, '3B ep3: last 12 updates (H = 28610)'), (axs[2], '1.5B', 3, '1.5B ep3: last 12 updates (H = 14305)')):
        G = GRIDS[g]
        h = G['hs'][k - 1]
        for sem, col, mk in (('uninterrupted', BR['ep3'], 'o'), ('implemented', INK, 's')):
            e = A[f'{g}|{sem}|ep{k}'][-12:]
            ax.plot(np.arange(-11, 1), e * 1e7, color=col, lw=2 if sem == 'uninterrupted' else 1,
                    marker=mk, ms=5, label='uninterrupted' if sem == 'uninterrupted' else 'as implemented')
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.set_xlabel('update index relative to H')
        ax.set_ylabel('LR applied (×1e-7)')
        ax.legend(frameon=False, fontsize=7, loc='upper right', labelcolor=INK2)
        ax.set_xticks(np.arange(-10, 1, 2))
    fig.tight_layout()
    fig.savefig(out / 'lr_schedule_normalized_and_endpoints.png', dpi=160)
    print('wrote', out / 'lr_schedule_absolute.png', out / 'lr_schedule_normalized_and_endpoints.png')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['replay', 'plot'])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    replay(a.out) if a.command == 'replay' else plot(a.out)


if __name__ == '__main__':
    main()
