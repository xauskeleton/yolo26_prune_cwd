"""
Extract DMS learned pruning ratios from training checkpoint → YAML.

Usage:
    python extract_dms_ratios.py --ckpt runs/detect/train/weights/lasta.pt --output dms_ratios.yaml

Then use with prune.py:
    python prune.py --weights last.pt --layer-ratio dms_ratios.yaml
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from dms.dms_utils import extract_ratios_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Extract DMS ratios from checkpoint')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to DMS training checkpoint (.pt)')
    parser.add_argument('--output', type=str, default='dms_ratios.yaml',
                        help='Output YAML path for prune.py --layer-ratio')
    args = parser.parse_args()

    ratios = extract_ratios_from_checkpoint(args.ckpt, args.output)

    print(f"\nPer-layer pruning ratios:")
    for name, ratio in sorted(ratios.items()):
        print(f"  {name:<40} {ratio:.4f}")

    print(f"\nNext step:")
    print(f"  python prune.py --weights <model.pt> --layer-ratio {args.output}")


if __name__ == "__main__":
    main()