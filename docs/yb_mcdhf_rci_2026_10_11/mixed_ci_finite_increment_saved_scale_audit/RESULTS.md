# Saved finite-scale causal results

The original finite qualification remains failed. This source/saved-array audit made zero new H/CI/gradient/optimizer/finite-method calls. Both1e−4 central comparisons fail the original derivative gate; both1e−6 and1e−8 saved comparisons pass it. The coarse-to-medium trends support central truncation; they do not qualify unexecuted consumer tests.

| Seed | Signed h | Explicit radial Ha | Reconstruction Ha | Rayleigh CI response Ha | Rayleigh total Ha | Reported total Ha |
|---|---:|---:|---:|---:|---:|---:|
| 413711 | +1e-04 | 4.949339944998352e-05 | 5.922053511086924e-17 | 1.172496456361065e-09 | 4.94945719464991e-05 | 4.949457194621441e-05 |
| 413711 | −1e-04 | 5.123756027459439e-05 | -2.27614956714419e-17 | 1.138842213714347e-09 | 5.123869911678534e-05 | 5.123869911575066e-05 |
| 413711 | +1e-06 | -3.68369103182688e-09 | -3.66796836730941e-17 | 1.356670253081055e-15 | -3.683689711836311e-09 | -3.683689051883786e-09 |
| 413711 | −1e-06 | 1.375724391146034e-08 | -1.852885695618777e-17 | 1.322921384900368e-15 | 1.375724521585287e-08 | 1.37572458391893e-08 |
| 413711 | +1e-08 | -8.670098364818925e-11 | -7.89602554206274e-17 | 4.088381666966571e-20 | -8.670106256756085e-11 | -8.670070094771146e-11 |
| 413711 | −1e-08 | 8.770836510992797e-11 | 7.848493989217224e-17 | 4.085146771541662e-20 | 8.770844363571933e-11 | 8.770635178863112e-11 |
| 413712 | +1e-04 | 5.322110122411015e-05 | 1.130733193427407e-17 | 7.526197206842799e-10 | 5.322185384384214e-05 | 5.322185384500837e-05 |
| 413712 | −1e-04 | 5.063256593148745e-05 | -7.037110955083333e-17 | 1.175573199580551e-09 | 5.063374150461666e-05 | 5.063374150648917e-05 |
| 413712 | +1e-06 | 1.81313172062001e-08 | 8.110783101822105e-17 | 3.224710687338208e-15 | 1.813132051201862e-08 | 1.813132081657758e-08 |
| 413712 | −1e-06 | -7.745578731245022e-09 | 5.344751225397799e-17 | 3.647441205362328e-15 | -7.745575030356305e-09 | -7.745573374269455e-09 |
| 413712 | +1e-08 | 1.2990377329371e-10 | 3.84460402196442e-17 | 2.504988080677694e-19 | 1.299038119902491e-10 | 1.299046911750169e-10 |
| 413712 | −1e-08 | -1.288651776269858e-10 | -4.141660871876173e-17 | 2.509180539299511e-19 | -1.288652187926765e-10 | -1.288669999982487e-10 |

| Seed | h | Central explicit radial | Central CI response | Central reconstruction | Total D(h) | Error to gradient | Original gate pass |
|---|---:|---:|---:|---:|---:|---:|---|
| 413711 | 1e-04 | -0.008720804123054334 | 1.682712132335914e-07 | 4.099101539115557e-13 | -0.008720635851431197 | -1.684135142853566e-07 | False |
| 413711 | 1e-06 | -0.00872046747164361 | 1.687443409034389e-11 | -9.075413358453163e-12 | -0.00872046746384459 | -2.592767840703392e-11 | True |
| 413711 | 1e-08 | -0.00872046743790586 | 1.617447712454506e-15 | -7.87225976563998e-09 | -0.008720475310164009 | -7.872247097251428e-09 | True |
| 413712 | 1e-04 | 0.01294267646311348 | -2.114767394481357e-06 | 4.083922074255369e-13 | 0.01294056169612738 | 2.114150132567078e-06 | False |
| 413712 | 1e-06 | 0.01293844796872256 | -2.1136525901206e-10 | 1.383015938212153e-11 | 0.01293844777118746 | 2.251926475155264e-10 | True |
| 413712 | 1e-08 | 0.01293844754603479 | -2.096229310908244e-14 | 3.993132446920296e-09 | 0.01293845153914628 | 3.993151462267086e-09 | True |

The measured orders from1e−4→1e−6 are1.9063 and1.9863, with error reductions6495.51 and9388.18. Existing-gradient consistency at1e−6 and the h² extrapolation errors (−9.09e−12/+1.38e−11) support the coarse-step truncation explanation. At1e−8 the errors rise to−7.87e−9/+3.99e−9: the smallest-step noise regime is observed, but its distinct bank/endpoint/eigenpair/summation contributions have not been isolated.

The full13 saved H spectra retain the original complete column-residual and Gram checks. Full target bindings/finite norms/response components, lifted-gradient comparison, coordinate tangency and input checks are in [result.json](result.json) and [verification.json](verification.json). The source/test/original contracts remain unchanged; an authorized distinct successor is required to exercise the17 consumer tests skipped by the original `-x` failure.
