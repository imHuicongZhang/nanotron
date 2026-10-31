"""nanotron's own LR scheduler, importable on a CPU host.

lr_scheduler_builder lives in src/nanotron/helpers.py, which imports flash-attn (GPU-only at import). Its source is
extracted from this checkout and executed with the names it uses, so every LR check (startup validation, LR-resume
regression tests, the Skipjack replay report) evaluates exactly the scheduler the trainer runs.
"""
from __future__ import annotations

import ast
import functools
import logging
import math
import sys
import typing
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=None)
def builder(code_root: str = str(CODE_ROOT)):
    """(lr_scheduler_builder, LRSchedulerArgs) of the nanotron source under code_root."""
    import torch
    src = Path(code_root) / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from nanotron.config import LRSchedulerArgs
    tree = ast.parse((src / 'nanotron/helpers.py').read_text())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lr_scheduler_builder'][0]
    ns = {'LambdaLR': torch.optim.lr_scheduler.LambdaLR, 'math': math, 'log_rank': lambda *a, **k: None,
          'logger': logging.getLogger(), 'logging': logging, 'Optimizer': torch.optim.Optimizer,
          'LRSchedulerArgs': LRSchedulerArgs, 'partial': functools.partial}
    ns.update({k: getattr(typing, k) for k in ('Optional', 'List', 'Dict', 'Union', 'Any', 'Tuple', 'Callable')})
    exec(compile(ast.Module([fn], []), 'nanotron.helpers.lr_scheduler_builder', 'exec'), ns)
    return ns['lr_scheduler_builder'], LRSchedulerArgs


def lr_lambda(cfg: dict):
    """(peak lr, lambda) of a config's scheduler: update n of that segment uses peak * lambda(n - 1)."""
    import torch
    build, LRArgs = builder()
    a = LRArgs(**cfg['optimizer']['learning_rate_scheduler'])
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=a.learning_rate)
    opt.get_base_optimizer = lambda: opt
    return a.learning_rate, build(opt, a, cfg['tokens']['train_steps']).lr_lambdas[0]
