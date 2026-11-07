#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""compare_reference.py [--reference REF.csv] RESULTS.csv...

Compares evaluator results with the paper's measurements, case by case
(key: suite, n, scheme, format, phi, passes).  Default reference:
evaluation/paper_reference.csv next to this tools directory.  For every case
it prints the whole-call time and the phase sum with their deviation from the
reference in percent, the effective bits of both, the fp64_equivalent flag,
and PARAMS DIFFER when the selected parameters (slices, integer_bits, moduli,
passes, partial_products) differ.  The summary gives the largest deviations
and the number of cases whose bits or parameters differ.  Cases without a
reference row are listed and skipped.  The exit status is 0 only when every
compared case has the reference's bits and parameters; timing deviations are
informative (performance varies with the board and its clock) and never fail.
"""
import csv
import os
import sys

PARAMS = ('slices', 'integer_bits', 'moduli', 'passes', 'partial_products')


def key(r):
    return (r['suite'], r['n'], r['scheme'], r['format'], float(r['phi']), r['passes'])


def main():
    args = sys.argv[1:]
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation', 'paper_reference.csv')
    if args[:1] == ['--reference']:
        ref_path = args[1]
        args = args[2:]
    if not args:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        sys.exit(2)
    with open(ref_path, newline='') as f:
        ref = {key(r): r for r in csv.DictReader(f)}
    rows = []
    for path in args:
        with open(path, newline='') as f:
            rows += list(csv.DictReader(f))
    print('%-5s %6s %-3s %-7s %4s %1s | %9s %9s %6s | %9s %9s %6s | %8s %8s | %s' % (
        'suite', 'n', 'sch', 'fmt', 'phi', 'p', 'total', 'ref', 'd%', 'phsum', 'ref', 'd%', 'bits', 'refbits', 'eq'))
    mx = mxp = 0.0
    bad = skipped = 0
    for r in rows:
        q = ref.get(key(r))
        if q is None:
            print('no reference for', key(r))
            skipped += 1
            continue
        t, tr = float(r['total_ms']), float(q['total_ms'])
        p, pr = float(r['phase_sum_ms']), float(q['phase_sum_ms'])
        b, br = float(r['effective_bits']), float(q['effective_bits'])
        same = all(r[c] == q[c] for c in PARAMS)
        d, dp = 100 * (t / tr - 1), 100 * (p / pr - 1)
        mx, mxp = max(mx, abs(d)), max(mxp, abs(dp))
        bits_differ = not (abs(b - br) <= 1e-6 or (b != b and br != br))   # nan == nan counts as equal
        bad += bits_differ or not same
        print('%-5s %6s %-3s %-7s %4s %1s | %9.2f %9.2f %+6.1f | %9.2f %9.2f %+6.1f | %8.4f %8.4f | %s %s' % (
            r['suite'], r['n'], r['scheme'], r['format'], r['phi'], r['passes'], t, tr, d, p, pr, dp, b, br,
            r.get('fp64_equivalent', ''), '' if same else 'PARAMS DIFFER'))
    print('cases %d (skipped %d), max |total diff| %.1f%%, max |phase-sum diff| %.1f%%, bits/params mismatches %d'
          % (len(rows) - skipped, skipped, mx, mxp, bad))
    sys.exit(0 if bad == 0 else 1)


main()
