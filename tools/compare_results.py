#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""compare_results.py BASELINE.csv NEW.csv

Compares two results.csv files of the evaluator case by case (key: suite, n,
scheme, format, phi, passes).  For every case of NEW the selected parameters
(slices, integer_bits, moduli, passes, partial_products, bw), the output hash
and the effective bits must equal those of BASELINE; timings are not compared.
Prints one line per mismatch and a summary; the exit status is 0 only when
every case of NEW has a matching baseline.  Use it to show that a host or
kernel change left the computed results bit-identical.
"""
import csv
import sys

COLS = ('slices', 'integer_bits', 'moduli', 'passes', 'partial_products', 'bw', 'output_hash', 'effective_bits')


def load(path):
    d = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            d[(r['suite'], r['n'], r['scheme'], r['format'], float(r['phi']), r['passes'])] = r
    return d


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        sys.exit(2)
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
    sys.exit(0 if bad == 0 else 1)


main()
