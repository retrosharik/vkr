from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_map_generator.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate or reuse ML maps, run full simulations with agent, and build/evaluate datasets without retraining.')
    parser.add_argument('--project-root', default='.', help='Project root. Default: current directory.')
    parser.add_argument('--output-dir', default='ml_map', help='Directory for generated maps relative to project root.')
    parser.add_argument('--total-maps', type=int, default=50, help='How many maps should be available and used in total.')
    parser.add_argument('--civilian-count', type=int, default=12, help='Base number of civilians per scenario.')
    parser.add_argument('--seed', type=int, default=None, help='Global random seed. Omit this flag to get a new random generation seed each run.')
    parser.add_argument('--startup-timeout-seconds', type=int, default=90, help='How long to wait for the server prompt before marking the run as failed.')
    parser.add_argument('--agent-command', default=f'{sys.executable} main.py', help='Command used to start the agent after the server is ready. The command is executed with cwd=BaseRescueAgent.')
    parser.add_argument('--skip-run', action='store_true', help='Only ensure maps exist, then build/evaluate datasets without launching simulations.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_pipeline(
        project_root=Path(args.project_root).resolve(),
        output_dir_name=args.output_dir,
        total_maps=args.total_maps,
        civilian_count=args.civilian_count,
        seed=args.seed,
        startup_timeout_seconds=args.startup_timeout_seconds,
        skip_run=args.skip_run,
        agent_command=args.agent_command,
    )


if __name__ == '__main__':
    raise SystemExit(main())
