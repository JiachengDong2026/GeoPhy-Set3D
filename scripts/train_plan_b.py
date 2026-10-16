#!/usr/bin/env python3
"""Independent continuous scheme B trainer; formal training requires passed gates."""
import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from research.plan_b.training import validate_config, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device')
    parser.add_argument('--run-name')
    parser.add_argument('--output-root')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stop-after-epochs', type=int, help='Controlled completed-epoch interruption; never marks an unfinished run complete')
    parser.add_argument('--promote-formal', action='store_true', help='Enable a formal design config only after same-source real smoke/gate evidence exists; all evidence checks still run')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.promote_formal:
        if config.get('stage') != 'formal' or config.get('status') != 'design_template_not_executable':
            parser.error('--promote-formal applies only to formal design templates')
        config['status'] = 'executable'
    if args.device:
        config['training']['device'] = args.device
    if args.run_name:
        config['experiment']['name'] = args.run_name
    if args.output_root:
        config['experiment']['output_root'] = args.output_root
    if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
        parser.error('--stop-after-epochs must be positive')
    validate_config(config)
    if args.validate_only:
        print('Valid executable scheme B config (formal evidence is checked again before training):', args.config)
        return
    print(train(config, resume=args.resume, stop_after_epochs=args.stop_after_epochs))


if __name__ == '__main__':
    main()
