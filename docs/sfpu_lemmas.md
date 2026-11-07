# Lemmas for the triple-float procedures on the SFPU

This note gives the proofs behind two contracts stated in
`src/backend/common/kernels/sfpu_tf_proc.h`:

* Part A: the final `FastTwoSum` of `tfMulR` and `tfMulP2` is exact for every triple
  that reaches these procedures in the library (Lemmas P and R), under the standing
  assumption that the FP32 arithmetic on the way neither overflows nor produces a
  subnormal number. The section "The arithmetic stays in the normal range" shows
  that this assumption holds for every element that the FP64 loader delivers exactly
  (normalized magnitude at least `2^-72`), which is the loader's own scope.
* Part B: why the second step of `tfAddSmall` uses `TwoSum` and not `FastTwoSum`.

The procedures themselves are in that header; the names of intermediate values used
below (`s3`, `e4`, `s5`, `e6`, `P0`, `F0`, `s_a`, `e_a`, `s_b`, `e_b`, `e_c`, `G`, ...)
are the ones written next to each line of `tfMulP2` and `tfMulR` there. The checks
mentioned below were made with the research implementation from which the library
derives; the check programs are not part of this distribution.

## Part A: the precondition of the final FastTwoSum of tfMulR and tfMulP2

Both procedures normalize at the end with the three operations

    (a, b) -> (s, t) = (fl(a+b), b - (fl(a+b) - a))

(`tfMulR` with `(a, b) = (e_c, G)`, `tfMulP2` with `(e6, e4)`). This returns
`s + t = a + b` exactly when "a = 0, or b = 0, or the exponent of a >= the exponent of
b" (Dekker); otherwise about half an ulp of b is lost. The condition fails when the
input triple overlaps: multiplying `(1, 2^50, 2^26)` by the radix 2 gives
`(2^51, 2^27, 0)` and loses the trailing +2 of the true value (in both procedures).
The following shows that the precondition holds for the triples that occur in actual
use, under the assumptions listed in the Notation, so that the six operations of
`TwoSum` are not needed (they would add three operations to each procedure).

### Notation

* `ulp(x)` is the weight of the last bit of the FP32 number x; `u` is a power of two;
  `c1 = 1 + 2^-24`.
* A triple `(x0, x1, x2)` satisfies `H_c(u)` when x0 is an integer multiple of u and
  `|x1| + |x2| <= c u`. x0 may be 0, and the two tail limbs may overlap each other.
  The bound is on the sum of the absolute values, not on the sum of the tail: with a
  bound on `|x1 + x2|` alone, pairs that cancel strongly (`x1 = 2^40 u`, `x2 ~ -x1`)
  would be admitted, and the estimate of `|s_a|` in Lemma R would not hold. (No such
  pair is known to break the final precondition, but the proof does not cover them.)
* `beta` is the radix: an integer with `2 <= beta <= 2^20` (at most 255 for `tfMulR`).
* FP32 arithmetic rounds to nearest even, and no overflow or subnormal number occurs
  on the way. This is a standing assumption of both lemmas (it is the value range
  stated at `tfMulP2` in the header); for the calls in the library it is discharged in
  the section "The arithmetic stays in the normal range" below.
* Shape of a FastTwoSum output: with `r1 = fl(t)` and `r2 = t - r1` (exact), if
  `|t| <= v` for a power of two v, then v is representable and rounding is monotone,
  so `|r1| <= v`, `|r2| <= ulp(r1)/2 <= 2^-24 v`, and `|r1| + |r2| <= c1 v`. This is
  how "tail <= v" below turns into the `c1 v` of H.

### Lemma P (tfMulP2; beta is a power of two, the condition under which tfMulRadix selects tfMulP2)

If the input satisfies `H_c(u)` and `beta c < 2^23`, then the precondition holds, the
multiplication by the radix is exact, and the output satisfies `H_{c1}(u')`, where
`u' = ulp(r0)`, or `ulp(s3)` (`<= u`) if `r0 = y0 + s3` is exact.

Proof. Since beta is a power of two, `y_i = beta x_i` is exact (this is what fails for
a beta that is not a power of two). `s3 = fl(y1 + y2)`, `|e4| <= ulp(s3)/2`, and
`|s3| <= beta c u (1 + 2^-23) < 2^24 u` (from `beta c < 2^23`), so `ulp(s3) <= u`, and
y0, a multiple of u, is a multiple of `ulp(s3)`. If `e6 != 0` (`y0 + s3` is not
exact), then by Sterbenz and by "if `|y0 + s3| < |s3|`, then `y0 + s3` is a multiple of
`ulp(s3)` with an exponent not above that of s3 and is therefore representable",
`|s5| >= |s3|`, and e6 is a nonzero multiple of `ulp(s3)`: `|e6| >= ulp(s3) > |e4|`.
The tail of the output, `r1 + r2 = e6 + e4`, is at most
`ulp(s5)/2 + ulp(s3)/2 <= ulp(r0)` if `e6 != 0`; if `e6 = 0`, `|e4| <= ulp(s3)/2` and
`r0 = y0 + s3` is a multiple of `ulp(s3)`. []

### Lemma R (tfMulR)

If the input satisfies `H_c(u)` and `beta c (1 + 2^-22) < 2^8`, then the precondition
holds, the only errors are the three approximations named below, and the output
satisfies `H_{c1}(u')`, where `u' = ulp(r0)`, or `ulp(s_a)` (`<= 2^-16 u`) if in case
(i) `r0 = P0 + s_a` is exact.

Proof. `P0 + F0 = beta x0` and `P1 + F1 = beta x1` are exact (a half of the Dekker
splitting has 12 bits, beta has 8). `P0 = fl(beta x0)`; if `|beta x0| < 2^24 u`, then
`beta x0` is representable and `F0 = 0`; otherwise `ulp(P0) >= 2u` and F0 is a multiple
of u. `P1 = fl(beta x1)` and `|F1| <= ulp(P1)/2`, so
`|P1| + |F1| <= beta |x1| (1 + 2^-23)`; together with `G1 = fl(F1 + fl(beta x2))`,

    |s_a| = |fl(P1 + G1)| <= beta (|x1| + |x2|) (1 + 2^-22) <= beta c u (1 + 2^-22) < 2^8 u

(this is where H must bound `|x1|` and `|x2|` separately), so `ulp(s_a) <= 2^-16 u`.

* (i) `F0 = 0`: `s_b = s_a`, `e_b = 0`, `G = e_a` (`|G| <= ulp(s_a)/2`). P0 is a
  multiple of u, so `P0 + s_a` is a multiple of `ulp(s_a)`. If `|P0 + s_a| < |s_a|`, it
  is exact for the same reason as in Lemma P (`e_c = 0`). Otherwise `|r0| >= |s_a|` and
  `e_c` is a multiple of `ulp(s_a)`: 0, or `|e_c| >= ulp(s_a) > |G|`.
* (ii) `F0 != 0`, `e_b = 0`: `s_b = F0 + s_a` is a multiple of `ulp(s_a)` and
  `|s_b| <= ulp(P0)/2 + 2^8 u << |P0|`, so `ulp(r0) >= ulp(P0)/2 >= u`, and `e_c` is a
  multiple of `ulp(s_a)`: 0, or `|e_c| >= ulp(s_a) > |G|`.
* (iii) `F0 != 0`, `e_b != 0`: `F0 + s_a` is not exact, so
  `|s_b| >= max(|F0|, |s_a|)/2`, `ulp(s_a) <= 2 ulp(s_b)`, and
  `|G| <= fl(ulp(s_a)/2 + ulp(s_b)/2) <= 1.5 ulp(s_b)`. Since `|s_b| << |P0|`, `e_c` is
  a multiple of `ulp(s_b)`: 0, or `|e_c| >= ulp(s_b)`, and from `|G| < 2 ulp(s_b)` the
  exponent of `e_c` is not smaller than that of G.

The tail is at most `ulp(r0)/2 + ulp(r0)/2` in (ii), in (iii) and in (i) with
`e_c != 0`; in (i) with `e_c = 0` it is `|e_a| <= ulp(s_a)/2` (and `r0 = P0 + s_a` is a
multiple of `ulp(s_a)`). []

### The calls in the library satisfy H

**Slicers** (Ozaki scheme I: `ozaki1/kernels/slice_tf_*`; Ozaki scheme II:
`ozaki2/kernels/digits.h`).

Grid u at the entry. `split_f64` (`common/kernels/f64_tf.h`) splits the normalized
value `x = S 2^P` (S the 53-bit integer significand) into `hi = a u`,
`mid = b 2^(P+7)`, `lo = c 2^P`, where `u = 2^(P+30)`, a is the upper 23 bits of S
rounded to nearest (`2^22 <= a <= 2^23`), and the remainder of the lower 30 bits,
`r = S - a 2^30 = b 2^7 + c`, satisfies `-2^29 <= r < 2^29`, `-2^22 <= b < 2^22`,
`0 <= c <= 127`. This u is not the FP32 `ulp(hi)`: for `a < 2^23` the last significand
bit of hi is always 0 and `ulp(hi) = 2^(P+29) = u/2` (only for `a = 2^23` is
`ulp(hi) = u`). The sum of the absolute values of the tail is `r 2^P < u/2` for
`r >= 0`, and `|b| 2^(P+7) + c 2^P <= 2^(P+29) + 127 * 2^P < u` for `r < 0`, so the
entry satisfies `H_1(u)`. Taking `ulp(hi)` as the grid would fail:
`x = 2^-2 + 2^-25 + 127 * 2^-54` (`S = 2^52 + 2^29 + 127`, `P = -54`) splits into
`(hi, mid, lo) = (2^-2 + 2^-24, -2^-25, 127 * 2^-54)`, whose tail sum
`2^-25 + 127 * 2^-54` exceeds `ulp(hi) = 2^-25` but is below `u = 2^-24`. (The
soft-float reader, a diagnostic path that the library does not use, forms
`hi = fl(x)`, `mid = fl(x - hi)`, `lo = fl(x - hi - mid)` with `|mid| <= ulp(hi)/2` and
`|lo| <= ulp(mid)/2`, which satisfies `H_1(ulp(hi))`.) After normalization the entry
has `|x| <= 1` in scheme I and `|x| <= 2^20` in scheme II, so from `S >= 2^52` follows
`P <= -32` and `u <= 2^-2 < 1`.

The induction starts from this `H_1(u)`. The output of a multiplication by the radix
(the `beta c` of the lemmas holds for `c = 1` as well as for `c = c1`, see below)
satisfies `H_{c1}(u')`, where u' is one of two kinds: `ulp(s3)` or `ulp(s_a)`, which
are not above the grid u of the input (`ulp(s3) <= u` in Lemma P, `ulp(s_a) <= 2^-16 u`
in Lemma R), or `ulp(r0)`, which can exceed u (the leading limb has been multiplied by
the radix) and is bounded separately below. Subtracting a digit, `t0 -= d` (d an
integer, also after clamping), keeps x0 a multiple of the grid as long as the grid is at
most 1 and does not change the tail, so it preserves `H_{c1}`. (Scheme II has folded the
first multiplication by the radix into the exponent of the normalization and subtracts
a digit directly from the entry `H_1(u)`; since `u < 1`, the same holds.) After a
multiplication by the radix `|t0| < 2^23`, so `ulp(r0) <= 1`; `ulp(s3)` and `ulp(s_a)`
are not above the grid of the input, which is at most 1 by the induction hypothesis;
hence "grid <= 1" holds inductively for both kinds of u'.

`beta c`: for the `2^20` of scheme II with `c = c1`, `2^20 c1 < 2^23`; for the `tfMulR`
of scheme I (radices 14 and 254), `254 c1 (1 + 2^-22) < 2^8`.

**Horner recurrence** (Ozaki scheme I, `ozaki1/kernels/recon_tf_*`):
`acc = acc beta + S_d`, where the integer `S_d` with `|S_d| < 2^31` is added by one or
two calls of `tfAddS`. With two calls the value is split into an upper and a lower
part: for a two's complement input (the `S_d` of the deferred reconstruction)
`v & 0xFFFFFF00` and `v & 0xFF`; for a sign-magnitude input (the `P_d` of the INT8
path; `SIGNMAG=1` in `recon_tf_i32in.cpp`) `|v| & 0x7FFFFF00` and `|v| & 0xFF`, each
multiplied by the sign of v. In both cases the upper part is at most `2^31` and the
lower part at most 255 in magnitude, and only these magnitudes are used below.

All limbs are integers. By the lemmas the output of a multiplication by the radix
satisfies `H_{c1}(u')`, and for the two kinds of input that the multiplication receives
in the Horner recurrence (`H_4(ulp(s))` and `H_{2^11}(1)`, shown below) it is in
`H_{c1}(ulp(r0))` in both cases: u' differs from `ulp(r0)` only when the sum is exact
(`u' = ulp(s3)` or `ulp(s_a)`); for `H_4(ulp(s))` (`x0 = s`, an input without
cancellation) `|beta s| >= beta 2^23 ulp(s)` is far larger than `|s3|`, `|s_a|` (below
`2^10 ulp(s)`), so `ulp(s3), ulp(s_a) < ulp(r0)`; for `H_{2^11}(1)` (an input after
cancellation) `s3` and `s_a` are integers below `2^24` and exact (`e4 = e_a = 0`), so
the tail is 0.

`tfAddS` replaces x0 by `s = fl(x0 + v)` and adds the residual `e1`
(`|e1| <= ulp(s)/2`) to x1 with TwoSum, passing the remainder down (only the lowest
limb rounds). The sum of the absolute values of the tail is therefore at most
`(|x1| + |x2| + |e1|)(1 + 2^-22)`, and with two additions (intermediate sum `s1`, first
residual `|e1'| <= ulp(s1)/2`) at most `(|x1| + |x2| + |e1'| + |e1|)(1 + 2^-22)^2`.

* Without cancellation (`|s| >= |r0|/2` and `|s| >= |s1|/2`):
  `ulp(r0), ulp(s1) <= 2 ulp(s)`, so the tail is at most
  `(2 c1 + 1 + 1/2)(1 + 2^-21) ulp(s) < 4 ulp(s)`, i.e. `H_4(ulp(s))` (with a single
  addition `(2 c1 + 1/2)(1 + 2^-22) < 3`, i.e. `H_3`).
* Cancellation can occur only for `|r0| < 2^33`, because `|v| < 2^31`. Then `ulp(r0)`,
  `ulp(s1)/2` and `ulp(s)/2` are all at most `2^9`, the tail is below
  `3 * 2^9 (1 + 2^-21) < 2^11`, and `x0 = s` is an integer: `H_{2^11}(1)`. Lemma P
  holds because `beta c <= 64 * 2^11 < 2^23`.

The proof of Lemma R used `beta c (1 + 2^-22) < 2^8` only for "`ulp(s_a)` is
sufficiently smaller than u" and "`|s_a| << |P0|` when `F0 != 0`". Both remain true
for `H_4` (`beta = 254` gives `beta c (1 + 2^-22) < 2^10`, so `|s_a| < 2^10 u` and
`ulp(s_a) <= 2^-14 u`). For `H_{2^11}(1)`, x1 and x2 are integers with
`|x1| + |x2| <= 2^11`, so `P1 = beta x1`, `fl(beta x2) = beta x2`, `G1` and `s_a` are
integers below `2^20` and exact (`F1 = 0`, `e_a = 0`). In (i) and (ii) `G = 0` and the
precondition is trivial; in (iii) `|s_b| <= ulp(P0)/2 + 2^19 << |P0|` (`F0 != 0`
implies `|P0| >= 2^24`) gives the same argument.

### The arithmetic stays in the normal range

Both lemmas assume that no FP32 operation on the way overflows or produces a
subnormal number (Notation). For the calls in the library this follows from the
bounds on the limbs as follows; where it does not follow, the lemmas are not claimed.

*Upper side.* In the slicers the leading limb is at most `R <= 1` before every
multiplication in scheme I (the invariant of the residual, `slice_tf_body.h`) and
below 1 after a digit has been subtracted in scheme II (the digits are bounded by
`2^19 + 2^16 + 1`, which is `2^20` times the residual; `digits.h`; at the entry
`|x| <= 2^20`), so after the multiplication it is below `2^8` and `2^20`
respectively, and the tail is at most `c1 u <= c1` (the grid is at most 1, shown
above). The largest intermediate value, `4097` times a limb in the Dekker splitting
of `tfMulR`, is therefore below `2^20`.
In the Horner recurrence the limbs are integers whose magnitude the host check
`oz_horner_fits` (`ozaki1/format.hpp`) keeps small enough for `4097` times the limb
to stay below `2^128`, as explained at that function.

*Lower side.* In the Horner recurrence every limb and every intermediate value is an
integer (the limbs, the radix, the digits and `4097` are integers, and a rounded sum,
difference or product of integers is an integer), so a nonzero value is at least 1.
In the slicers let `x = S 2^P` be the normalized value of the element, with `S` its
53-bit integer significand, so that `P` is the exponent of its last bit (`elsb` in
`f64_tf.h`). The entry limbs `hi = a 2^(P+30)`, `mid = b 2^(P+7)`, `lo = c 2^P` are
integer multiples of `2^P` (the loader `tf_load_f64` in `slice_tf_core.h` forms `mid`
and `lo` as exact products of the same multiples), and the radix, the digits and
`4097` are integers. Let `P >= -124`, i.e. `|x| >= 2^-72` (as `2^52 <= S < 2^53`).
Then, by induction over the operations, every value that occurs is a multiple of
`2^P`: the exact sum, difference or product of multiples of `2^P` is `k 2^P` for an
integer k, which is either 0 or at least `2^-124 > 2^-126` in magnitude, so it is
rounded as a normalized FP32 number, either exactly (`|k| < 2^24`) or on a grid
`2^(P+j)` with `j >= 1`, and the result is again a multiple of `2^P`. Hence no
subnormal number is produced. This range is exactly the one in which the FP64 loader
is verified to deliver the triple exactly (`split_f64` in `f64_tf.h`, "Scope of
application": below `2^-72` the loader may lose `lo`, then `mid`, depending on
whether the SFPU flushes subnormals). For an element below `2^-72` the lemmas are
therefore not claimed; with the default settings such an element lies below the width
that the later stages retain anyway (54 to 64 bits below the row or column maximum;
"Precision and input range" in `README.md`, and the binade counts relative to the
maximum in `f64_tf.h`), and for larger slice counts or widths the loader's comment
states what is lost first.

### Checks performed on the research implementation (not reproducible from this distribution)

The lemmas above do not depend on the following check, whose programs are not
distributed. The procedures were instantiated on a bit-level model of single SFPU
operations and compared with 512-bit fixed-point arithmetic for the slicers of scheme I,
the digit generation of scheme II (including that the entry triple satisfies `H_1(u)`
with `u = 2^(P+30)`), the Horner recurrence of scheme I (two's complement and
sign-magnitude input, values at the bound, cancellation of the leading part), the shapes
directly after a digit subtraction, families of triples for the lemmas (`tfMulR` with
cancelling shapes and with overlapping tails, for `beta c < 2^8` and for the
`beta c < 2^10` of the Horner `H_4`) and entries whose tail exceeds `ulp(hi)`; no
violation of the precondition and no mismatch was found. The check, and not a lemma,
also established the proposition W for `tfMulP2`: if the input satisfies "`x0 = 0` or
`|x1 + x2| <= |x0|`", the precondition holds, the multiplication by the radix is exact,
and the output satisfies W again. The model was confirmed to agree bit for bit with the
hardware for the tile-operation wrappers, `Rnd`, `tfMulRadix` (radices 2, 32, 64, 14,
254), `tfMulR`, `tfAddS` and the FP64 loader. `tfAddSmall` and `ModRound` (which the
device executes in every call of scheme II) were not part of this model-versus-device
comparison, but they instantiate the same procedure templates on the same wrapper
operations, so the agreement of the single operations carries over to them.

## Part B: the second step of tfAddSmall, and the error of the CRT fraction

### Why FastTwoSum is not used

With the three operations of Dekker's FastTwoSum in the second step, `|a1| >= |v|`
would be required, and this cannot be stated without a lower bound on `a0`. A step
whose precondition cannot be proved is not kept on the strength of a property that
holds somewhere else, least of all to save three operations. The checks below show,
in addition, that an end-to-end accuracy test would not reveal such an error: only a
direct measurement of the absolute error of the CRT fraction can.

### Checks performed on the research implementation (not reproducible from this distribution)

With the research implementation (programs not distributed) the worst absolute error of
the CRT fraction `frac` was measured directly in a simulation of the two-pass
configuration of Ozaki scheme II with INT8, for the second step written as TwoSum and
as FastTwoSum, and for the decision on the number of small terms made relative to the
largest term (the implementation) and with a fixed absolute threshold (a variant the
library does not have). FastTwoSum with the relative decision came out unharmed only
because the second pass (the accumulation of the residual), where the error would
appear, has no small terms and the step is not taken; in the first pass the step is
taken for the last two terms, and that its worst error equalled that of TwoSum in the
simulated trials does not mean that the precondition `|a1| >= |v|` is guaranteed. With
the fixed threshold FastTwoSum lost many bits of frac. The error of frac does not show
in the final accuracy: a sweep of the final accuracy over the exponent range of the
input gave identical results for that worst variant and for the implementation, because
the final result is determined by the width b and the CRT window. An end-to-end
accuracy test therefore cannot detect this kind of error; only a direct measurement of
the absolute error of frac can.
