"""Rebuild native Ne rates; thread limits must be set before import."""
import json
import time
from pathlib import Path

from atomx.xfel import build_atomic_rate_table

HERE = Path(__file__).resolve().parent
BASE = {(1, -1): 2, (2, -1): 2, (2, 1): 2, (2, -2): 4}


def main():
    start = time.perf_counter()
    def progress(phase, done, total):
        if done == 1 or done % 10 == 0 or done == total:
            print(phase, done, '/', total, 'elapsed_s', round(time.perf_counter()-start, 2), flush=True)
    table = build_atomic_rate_table(10, BASE, 1000, checkpoint_dir=HERE / 'checkpoints',
                                    workers=8, progress=progress)
    table['provenance']['build_elapsed_s'] = time.perf_counter() - start
    (HERE / 'ne_1000eV.json').write_text(json.dumps(table, indent=2, allow_nan=False)+'\n')
    print('COMPLETE', len(table['configs']), flush=True)


if __name__ == '__main__':
    main()
