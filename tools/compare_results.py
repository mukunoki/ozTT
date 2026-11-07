#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""compare_results.py BASELINE.csv NEW.csv

Compares two results.csv files of the evaluator case by case (key: suite, n,
scheme, format, phi, passes).  For every case of NEW the selected parameters
(slices, integer_bits, moduli, passes, partial_products, bw), the output hash
and the effective bits must equal those of BASELINE; timings are not compared.
Prints one line per mismatch and a summary; the exit status is 0 only when
NEW has at least one case and every case of NEW has a matching baseline.
A duplicate case key in either file or a missing file is an error (exit 2).
Use it to show that a host or kernel change left the computed results
bit-identical.
"""
import csv
import sys

COLS = ('slices', 'integer_bits', 'moduli', 'passes', 'partial_products', 'bw', 'output_hash', 'effective_bits')


def fail(msg):   # usage or input error
    print(msg, file=sys.stderr)
    sys.exit(2)


def load(path):
    d = {}
    try:
        f = open(path, newline='')
    except OSError as e:
        fail(e)
    with f:
        for r in csv.DictReader(f):
            k = (r['suite'], r['n'], r['scheme'], r['format'], float(r['phi']), r['passes'])
            if k in d:
                fail(f'{path}: duplicate case {k}')
            d[k] = r
    return d


def main():
    if sys.argv[1:] == ['--help']:
        print(__doc__.strip())
        sys.exit(0)
    if len(sys.argv) != 3:
        fail(__doc__.strip().splitlines()[0])
    old, new = load(sys.argv[1]), load(sys.argv[2])
    bad = 0
    for k, r in new.items():
        if k not in old:
            print('no baseline for', k)
            bad += 1
            continue
        diff = [c for c in COLS if r[c] != old[k][c]]
        if diff:
            bad += 1
            print('DIFF', k, [(c, old[k][c], r[c]) for c in diff])
    missing = [k for k in old if k not in new]
    print(f'new {len(new)} cases, baseline {len(old)}, mismatches {bad}, baseline cases not in new {len(missing)}')
    if not new:
        print('FAIL: no case in NEW')
    sys.exit(0 if bad == 0 and new else 1)


main()
