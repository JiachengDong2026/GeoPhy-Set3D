#!/usr/bin/env python3
"""Independent scheme A measurement-only research entry point."""
import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geophy_set3d.training.research_selfsup import load_research_config, train_research


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true', help='Resume only a matching completed epoch boundary')
    parser.add_argument('--device', help='Override the device before the resolved config is frozen')
    parser.add_argument('--seed', type=int, choices=(20261011, 20261012, 20261013),
                        help='Preregistered seed; automatically gives this run a distinct seed suffix')
    parser.add_argument('--run-name', help='Create a distinct, immutable output directory')
    parser.add_argument('--output-root', help='Override the output parent directory')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    config = load_research_config(args.config)
    if args.device:
        config['training']['device'] = args.device
    if args.seed is not None:
        config['training']['seed'] = args.seed
        name = config['experiment']['name']
        config['experiment']['name'] = re.sub(r'_seed\d+$', '', name) + f'_seed{args.seed}'
    if args.run_name:
        config['experiment']['name'] = args.run_name
    if args.output_root:
        config['experiment']['output_root'] = args.output_root
    if args.validate_only:
        print('Valid executable research config:', args.config)
        return
    print(train_research(config, resume=args.resume))


if __name__ == '__main__':
    main()
