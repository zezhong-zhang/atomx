# Independent inner Coulomb quadrature review

This is a source and mathematics review of the archived 500ac ordinary and
64ac private candidate epochs. Both consume identical `convol.py`, SHA
`270ac7a453588d0bacc746fb90f062ade4ea5af58e6e690f491db2e5b8ac72fe`.
No kernels, arrays, eigensolvers, H, CI, gradients, or optimizers were executed
in this review. The original failed qualification remains failed.

## The measured common error

The [actual anchor failure](../correlated29_radial_reuse_qualification_preparation/actual/anchors/failure.json)
is `analytic_6_pppp`: absolute error `3.6586666666948986e-5` Ha, exceeding
the original limit `3.585493333333332e-14` Ha. The
[frozen driver](../correlated29_radial_reuse_qualification_preparation/run.py)
uses a single Bernstein cell, `ns = ks = 4`, radius `R = 0.7`, outer
`n_gauss = 16`, and four signed spinors with P = Q = scale times r,
whose scales are 1, -2, 3, and 4.

For one component, the density product on each electron is proportional
to r squared, with total scale -24. For rank k, the exact radial integral is

    Rk = -24 * 2 * R^5 / [5 * (k + 3)].

At k = 6, the inner integrand is r^8. `_triang_sym` uses only four
Gauss-Legendre nodes, which integrate polynomials through degree 7.
On [0, 1], the exact error of the four-node rule applied to x^8 is

    integral(x^8) - GL4(x^8)
      = 1 / [(2*4 + 1) * binomial(8,4)^2]
      = 1 / 44100.

Since integral(x^8) = 1/9, the relative inner error is 1/4900. Mapping
to [0, u] multiplies both exact integral and error by u^9. The outer
integrand for this particular anchor then has degree 4, so its 16-node
rule resolves that polynomial. The signed quartet becomes less negative
by

    | -48 * 0.7^5 / 45 | / 4900
      = 3.658666666666...e-5 Ha.

This agrees with the archived observed error to floating-point rounding.
Dense and semiseparable contractions share the same triangle builder;
their agreement cannot detect this common integration error.

## Minimum inner order

On each knot interval, a B-spline of order s has degree at most s-1.
The inner integrand B_i(r) B_j(r) r^k therefore has degree at most
2s-2+k for nonnegative integer k. A rule with n nodes requires

    2n - 1 >= 2s - 2 + k,
    n >= ceil((2s - 1 + k)/2) = s + floor(k/2).

Thus PP and QQ inner integrations use their respective component order
plus `k // 2`. `_triang_pqpq` computes both Q-inner/P-outer and
P-inner/Q-outer triangles. One shared node count

    max(ksp, ksq) + floor(k/2)

covers both. The reversed mixed component has the same requirement.
Applying this formula only to one mixed direction would leave the other
underintegrated. No blocking algebra issue was found in this proposed
inner-order formula within the stated domain.

## Limits of that correction

The formula proves polynomial coverage of the rising inner integral.
It does not prove exact integration of the entire Coulomb kernel.
On intervals with nonzero left endpoint, the outer expression and the
falling moments contain negative powers of r; they are generally rational
functions. Outer quadrature still needs independent accuracy controls.

On the origin interval, an exactly integrated polynomial inner density
times r^k produces powers that cancel the outer r^(-k-1). The resulting
outer polynomial has degree at most d_left+d_right. For arbitrary
same-order spline pair densities this can reach 4s-4, requiring at least
2s-1 outer Gauss nodes for worst-case polynomial exactness. The default
`n_gauss = ks` in the archived basis preparation documents overlap
coverage; it is not a proof of complete two-electron exactness.

Negative k is outside this inner-order derivation: the inner integrand
may cease to be polynomial. In particular, the negative exponent
`-(k+1)` used for falling moments must not inherit the rising-moment
degree formula. The archived Gauss-node helper documents numerical
adequacy for n <= 20; this review does not certify n > 20.

## Radial kernel and physical states

The toy quartet is a legal finite radial-kernel input with no angular
quantum-number labels. Its k = 6 anchor cannot be waived by assigning
it an s-s angular coefficient of zero. In real scalar-Coulomb assembly,
the separate angular layer imposes parity and j-triangle rules. Rank 6
can occur in f channels; a valid angular coefficient does not lower the
polynomial degree required by the radial builder.

For a smooth finite central potential, the regular exact Dirac solution
has leading P proportional to r^(l+1), and Q proportional to
r^(l_small+1). This is a statement about the physical regular solution,
not a constraint that can be assumed for every individual B-spline
quartet or variational test direction. Generic origin-cell trial functions
must retain the full polynomial coverage above. The physical leading-power
statement is not extrapolated to a singular point-nucleus potential here.

The failed analytic anchor ran before the actual 29-orbital/1046-CSF
ordinary-versus-candidate H/CI/gradient comparison. It therefore provides
no measured Yb energy, transition, stationarity, or experimental-gap
correction. It cannot be used to explain the approximately 2000 cm^-1
experimental discrepancy or to qualify a production optimization step.

