# Complete finite eigensystem refinement

Let B be the saved complete116-column original-P complement and h,S the unchanged exactly symmetric native float arrays. A=BᵀhB and M=BᵀSB are recontracted in longdouble. Their transpose averages remove contraction roundoff, while final physical residuals are evaluated against the original h,S and B.

The saved full generalized coefficient matrix C supplies a complete initial basis. Its symmetric Gram K=CᵀMC=I+D must satisfy ‖D‖∞≤1e−10. This bound gives positive K without discarding any direction. W=I−D/2+3D²/8−5D³/16+35D⁴/128 is the fourth-order inverse-square-root series; truncation starts at D⁵. Recomputing actual Gram checks the finite implementation. The complete transformed basis CW stays within the same B span.

For each pair of the complete symmetric reduced matrix R, with a=Rii, d=Rjj, b=Rij, δ=(d−a)/2, use t=sign(δ)b/(|δ|+hypot(δ,b)); at δ=0 use t=sign(b). Set c=1/sqrt(1+t²), s=tc. The orthogonal two-column transformation has columns(c,−s) and(s,c), and its updated diagonals are a−tb,d+tb. The transformed offdiagonal vanishes algebraically. These formulas avoid dividing by a tiny b. All i<j pairs are visited lexicographically; only an exactly zero b skips a rotation. The rotations preserve the complete116-dimensional span.

No particular old or failed root is selected for refinement. At each complete sweep boundary every mapped physical vector is normalized in original S and every original constrained residual is measured. Convergence requires both the stricter internal offdiagonal criterion and all original full-spectrum gates. The eventual sorting permutes the complete arrays together. The isolated lowest physical root must retain its original signed-span overlap, minimum gap, minimum energy, original-P leakage, and frozen-orbital Gram controls.

This is a numerical representation of the same finite auxiliary problem. A pass cannot qualify otherκ, screened CI response, MCDHF stationarity, radial convergence, or experimental accuracy. A failure retains all completed arrays and counters and cannot trigger another driver, subset, precision, or iteration scan within this contract.
