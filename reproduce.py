"""Run the packaged reproduction stages. Full simulations require an explicit command."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['figures', 'inputs', 'summaries', 'quick', 'full'])
    parser.add_argument('--procs', type=int, default=2)
    args = parser.parse_args()
    if args.procs < 1:
        parser.error('--procs must be positive')
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = '1'
    stages = {
        'figures': [['figs2.py']],
        'inputs': [['build_nets.py'], ['human.py']],
        'summaries': [['report2.py'], ['extra_thresholds.py']],
        'quick': [['run2.py', 'E1', 'E3', '--quick', '--procs', str(args.procs)]],
        'full': [['run2.py', 'E1', 'E1alt', 'E1net', 'E1null', 'E2', 'E4b', 'E5', 'E6', 'E8', 'EX', 'E3', 'E7', '--procs', str(args.procs)],
                 ['report2.py'], ['extra_thresholds.py'], ['figs2.py']],
    }
    for command in stages[args.stage]:
        subprocess.run([sys.executable] + command, cwd=ROOT, env=env, check=True)

if __name__ == '__main__':
    main()
