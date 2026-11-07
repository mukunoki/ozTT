#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""codecheck.py BASELINE NEW [--allow-nonascii]   |   codecheck.py --selftest

Checks that the C/C++ source NEW differs from BASELINE only in comments,
whitespace and digit separators.  NEW must also be pure ASCII unless
--allow-nonascii is given.  Prints "OK NEW" or "FAIL NEW: reason"; the exit
status is 0 only for OK.

Both files are split into C++ preprocessing tokens the same way: line
continuations are joined, comments become whitespace, whitespace separates
tokens and is otherwise dropped, and the end of a preprocessor directive line
is a token of its own (so two directives on one line differ from two lines).
Identifiers, numbers (without digit separators: 1'000 equals 1000), character
literals and punctuators (longest match) are single tokens; string literals,
including raw strings, are kept whole and adjacent literals are concatenated.
The two token streams must then be identical: a changed string or character
literal is a difference, and no preprocessor is run, so a changed macro is a
changed token.  A // comment ending in a backslash continues on the next line
(as in C++), on both sides alike.  --selftest checks these rules on built-in
snippets.
"""
import re
import sys

TOKEN = re.compile(r'''
    (?P<ws> \s+ | //[^\n]* | /\*.*?\*/ )
  | (?P<str> (?:u8|u|U|L)?R"(?P<d>[^()\\\s]*)\(.*?\)(?P=d)" | (?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*" )
  | (?P<chr> (?:u8|u|U|L)?'(?:[^'\\\n]|\\.)*' )
  | (?P<num> \.?[0-9](?:[eEpP][+-]|'?[0-9A-Za-z_]|\.)* )
  | (?P<id> [A-Za-z_][A-Za-z0-9_]* )
  | (?P<op> <<=|>>=|\.\.\.|->\*|<=>|->|::|\#\#|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\.\*
           | [-+*/%^&|=<>!~?:;,.(){}\[\]\#] )
''', re.X | re.S)


def tokens(text):
    """Preprocessing tokens of a C++ source as (kind, text) pairs."""
    text = text.replace('\\\n', '')   # line continuations
    out, i, bol, directive = [], 0, True, False
    while i < len(text):
        m = TOKEN.match(text, i)
        if not m:
            line = text.count('\n', 0, i) + 1
            raise ValueError(f'cannot tokenize line {line}: {text[i:i + 20]!r}')
        i, kind, tok = m.end(), m.lastgroup, m.group()
        if kind == 'ws':
            if '\n' in tok:
                if directive:
                    out.append(('nl', '\n'))
                bol, directive = True, False
            continue
        if kind == 'num':
            tok = tok.replace("'", '')
        if kind == 'str' and out and out[-1][0] == 'str':
            out[-1] = ('str', out[-1][1][:-1] + tok[tok.index('"') + 1:])   # adjacent literals
        else:
            out.append((kind, tok))
        directive = directive or (tok == '#' and bol)
        bol = False
    if directive:
        out.append(('nl', '\n'))
    return out


def compare(base, new):
    """None when the two sources have the same tokens, else a message."""
    tb, tn = tokens(base), tokens(new)
    for k, (a, b) in enumerate(zip(tb, tn)):
        if a != b:
            return f'code differs at token {k}: baseline {a[1][:200]!r}, new {b[1][:200]!r}'
    if len(tb) != len(tn):
        return f'token count differs ({len(tb)} baseline vs {len(tn)} new)'
    return None


def selftest():
    same = [   # (baseline, new) that must be OK
        ('int a; // one\n', 'int a; /* two */\n'),
        ('int a=b+c;', 'int a = b\n + c ;\n'),
        ('int a; /* x \\ */ int b;', 'int a; int b;'),
        ('// a \\\nint x;\nint y;', '// b \\\nint x;\nint y;'),   # the comment continues on both sides
        ('#define A \\\n  1\nint a\\\n= 2;', '#define A 1\nint a = 2;'),
        ('int n = 1\'000;', 'int n = 1000;'),
        ('const char *s = "a" "b";', 'const char *s = "ab";'),
        ('/* c */ #define A 1\n', '#define A 1 // c\n'),
    ]
    differ = [   # (baseline, new) that must FAIL
        ('a + +b;', 'a++ + b;'),
        ('a <<= 1;', 'a << = 1;'),
        ('#define A 1\n#define B 2\n', '#define A 1 #define B 2\n'),
        ('const char *s = "abc";', 'const char *s = "abd";'),
        ("char c = 'a';", "char c = 'b';"),
        ("int c = u'a';", "int c = u'b';"),
        ('// a \\\nint x;\nint y;', '// a\nint x;\nint y;'),   # the backslash swallows "int x;"
        ('int n = 10;', 'int n = 1 0;'),
    ]
    bad = [(a, b) for a, b in same if compare(a, b)] + [(a, b) for a, b in differ if not compare(a, b)]
    for a, b in bad:
        print(f'selftest FAILED: {a!r} vs {b!r}')
    print(f'selftest: {len(same) + len(differ) - len(bad)} of {len(same) + len(differ)} cases OK')
    sys.exit(1 if bad else 0)


def read(path):
    with open(path, encoding='utf-8', errors='surrogateescape') as f:
        return f.read()


def main():
    opts = [a for a in sys.argv[1:] if a.startswith('--')]
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if opts == ['--selftest'] and not args:
        selftest()
    if len(args) != 2 or set(opts) - {'--allow-nonascii'}:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        sys.exit(2)
    base, new = args
    newtext = read(new)
    msgs = []
    bad = [k for k, ch in enumerate(newtext) if ord(ch) > 127]
    if bad and '--allow-nonascii' not in opts:
        line = newtext.count('\n', 0, bad[0]) + 1
        msgs.append(f'{len(bad)} non-ASCII characters remain (first at line {line}: {newtext[bad[0]]!r})')
    try:
        d = compare(read(base), newtext)
    except ValueError as e:
        d = str(e)
    if d:
        msgs.append(d)
    for m in msgs:
        print(f'FAIL {new}: {m}')
    if not msgs:
        print(f'OK {new}')
    sys.exit(1 if msgs else 0)


main()
