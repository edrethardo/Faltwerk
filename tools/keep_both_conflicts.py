#!/usr/bin/env python3
"""Resolve 'both sides added' conflicts: keep HEAD block, then theirs."""
import re, sys
p = sys.argv[1]
s = open(p).read()
pat = re.compile(r'^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n', re.S | re.M)
n = len(pat.findall(s))
s = pat.sub(lambda m: m.group(1).rstrip('\n') + '\n\n' + m.group(2), s)
open(p, 'w').write(s)
print('resolved', n)
