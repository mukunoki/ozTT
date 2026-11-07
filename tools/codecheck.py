#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""codecheck.py BASELINE NEW [--allow-nonascii]

Checks that the C/C++ source NEW differs from BASELINE only in comments and
whitespace, and in string literals that contained non-ASCII text in BASELINE
(such messages may be reworded).  NEW must also be pure ASCII unless
--allow-nonascii is given.  Prints "OK NEW" or "FAIL NEW: reason";
the exit status is 0 only for OK.

Both files are tokenized the same way: comments become whitespace, whitespace
is dropped, string literals (including raw and adjacent literals) are kept as
single tokens, and character literals stay with the code.  The two token
streams must then be identical.  No preprocessor is run: a changed macro is a
changed token.
"""
import re
import sys


def tokens(text):
    text = text.replace('\\\n', '')   # line continuations
    out = []          # list of ('code', str) / ('str', str)
    i, n = 0, len(text)
    code = []

    def flush():
        s = re.sub(r'\s+', '', ''.join(code))
        code.clear()
        if s:
            out.append(('code', s))

    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            # line continuation inside a // comment
            while j != -1 and text[j - 1] == '\\':
                j = text.find('\n', j + 1)
            i = n if j == -1 else j
            code.append(' ')
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            if j == -1:
                raise SystemExit('unterminated /* comment')
            i = j + 2
            code.append(' ')
        elif c == '"':
            m = re.search(r'(?:u8|u|U|L)?R$', ''.join(code)[-3:]) if code else None
            if m:  # raw string
                k = text.index('(', i)
                delim = text[i + 1:k]
                j = text.index(')' + delim + '"', k)
                lit = text[i:j + len(delim) + 2]
                i = j + len(delim) + 2
            else:
                j = i + 1
                while text[j] != '"':
                    j += 2 if text[j] == '\\' else 1
                lit = text[i:j + 1]
                i = j + 1
            flush()
            if out and out[-1][0] == 'str':
                out[-1] = ('str', out[-1][1][:-1] + lit[1:])   # adjacent literals concatenate
            else:
                out.append(('str', lit))
        elif c == "'" and not (code and re.search(r'[0-9A-Za-z_]$', ''.join(code)[-1:])):
            j = i + 1
            while text[j] != "'":
                j += 2 if text[j] == '\\' else 1
            code.append(text[i:j + 1])
            i = j + 1
        else:
            code.append(c)
            i += 1
    flush()
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        sys.exit(2)
    base, new = args
    tb = tokens(open(base, encoding='utf-8').read())
    newtext = open(new, encoding='utf-8').read()
    tn = tokens(newtext)
    bad = [(k, ch) for k, ch in enumerate(newtext) if ord(ch) > 127]
    ok = True
    if bad and '--allow-nonascii' not in sys.argv:
        line = newtext.count('\n', 0, bad[0][0]) + 1
        print(f'FAIL {new}: {len(bad)} non-ASCII characters remain (first at line {line}: {bad[0][1]!r})')
        ok = False
    if len(tb) != len(tn):
        print(f'FAIL {new}: token count differs ({len(tb)} baseline vs {len(tn)} new)')
        ok = False
    for k, (a, b) in enumerate(zip(tb, tn)):
        if a == b:
            continue
        if a[0] == 'str' and b[0] == 'str' and any(ord(ch) > 127 for ch in a[1]):
            continue   # a non-ASCII message was reworded: allowed
        print(f'FAIL {new}: code differs at token {k}:\n  baseline: {a[1][:300]}\n  new:      {b[1][:300]}')
        ok = False
        break
    if ok:
        print(f'OK {new}')
    sys.exit(0 if ok else 1)


main()
