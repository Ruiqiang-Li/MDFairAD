import argparse
import sys
import os
import warnings

import torch
import numpy as np

warnings.filterwarnings('ignore')

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from models.helpers import ParserSingleton, WandbSingleton
from models.datasets import CreditDataset, RedditDataset, TwitterDataset
from models.variants.non_linear_end_to_end_ad import NonLinearEndToEndADModel
from constants import CREDIT, REDDIT, TWITTER

_DATASET_MAP = {
    CREDIT:  CreditDataset,
    REDDIT:  RedditDataset,
    TWITTER: TwitterDataset,
}


def _parse_ad_args():
    """Parse anomaly-detection-specific CLI args (add_help=False to coexist with ParserSingleton)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--ad_model', type=str, default='dominant',
                   choices=['dominant', 'vgod', 'conad', 'cola'])
    p.add_argument('--ad_hid',   type=int, default=64)
    p.add_argument('--ad_epoch', type=int, default=100)
    p.add_argument('--ad_batch', type=int, default=0)
    p.add_argument('--seed_num', type=int, default=5)
    return p.parse_known_args()[0]


def _format_metric_name(metric_name):
    return metric_name.replace('_', ' ')


def _print_warmup_banner():
    print("\n[MDFP] Warmup stage in progress, please be patient.")
    print("       If compute is limited, cache the warmup result on the first run:")
    print("           save cache : add `--e2e_cache_warmup` to the command")
    print("           load cache : add `--e2e_load_warmup`  to skip warmup next time")


def _print_results(results):
    print(f"\n{'='*60}")
    print("  Evaluation Results")
    print(f"{'='*60}")
    for metric, value in results.items():
        print(f"  {_format_metric_name(metric):<24s}: {value:.4f}")
    print(f"{'='*60}\n")


def _report_seed_summary(all_results):
    if not all_results:
        return
    metric_names = list(all_results[0].keys())
    print(f"\n{'='*60}")
    print("  Multi-seed Summary (mean ± std)")
    print(f"{'='*60}")
    for metric_name in metric_names:
        values = np.array([float(result[metric_name]) for result in all_results], dtype=np.float64)
        print(f"  {_format_metric_name(metric_name):<24s}: {values.mean():.4f} ± {values.std():.4f}")
    print(f"{'='*60}\n")


def run_once(args, ad_args):
    WandbSingleton()

    print(f"\n{'='*60}")
    print(f"  Dataset : {getattr(args, 'dataset', REDDIT)}")
    print(f"  Device  : {args.device}")
    print(f"{'='*60}")

    from utils import seed_everything
    seed_everything(args.seed)

    dataset_name  = getattr(args, 'dataset', REDDIT)
    dataset_class = _DATASET_MAP.get(dataset_name)
    if dataset_class is None:
        print(f"[ERROR] Unsupported dataset '{dataset_name}'. Available: {list(_DATASET_MAP.keys())}")
        sys.exit(1)

    dataset = dataset_class(dataset_name)
    data    = dataset.data

    _print_warmup_banner()
    e2e_model = NonLinearEndToEndADModel(args, data)
    results = e2e_model.run()
    _print_results(results)
    return results


def main():
    args = ParserSingleton().args
    ad_args = _parse_ad_args()

    seed_num = max(1, int(getattr(ad_args, 'seed_num', 1)))
    all_results = []
    for seed in range(seed_num):
        args.seed = seed
        print(f"\n{'#'*60}")
        print(f"[Multi-Seed] Running seed {seed + 1}/{seed_num} (seed={seed})")
        print(f"{'#'*60}")
        result = run_once(args, ad_args)
        if isinstance(result, dict):
            all_results.append(result)

    if all_results:
        _report_seed_summary(all_results)


if __name__ == '__main__':
    main()
