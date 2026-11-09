"""OpenWAM training entry: badminton hybrid supervision + paraphrase prompts."""
import runpy
import sys
from pathlib import Path

ROOT = Path('/datapool/lianjh/openwam')
sys.path[:0] = [str(ROOT / 'code/source'), str(ROOT / 'code/badminton')]
import openwam_badminton  # noqa: F401  registers 'badminton'
import openwam_badminton_hybrid  # noqa: F401  registers 'badminton_hybrid'

if __name__ == '__main__':
    sys.argv = [
        str(ROOT / 'code/source/scripts/train.py'),
        '--config-path', str(ROOT / 'configs'),
        '--config-name', 'top_state_5ep',
        *sys.argv[1:],
    ]
    runpy.run_path(str(ROOT / 'code/source/scripts/train.py'), run_name='__main__')
