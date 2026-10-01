import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx.properties.g_factor import g_factor_dirac_hydrogenic, g_factor_qed
from atomx.properties.zeeman.lande import lande_gJ_bound_single_electron, lande_gJ_hydrogenic

# H 1s bound g: CODATA 2.0022838453 (bound electron in H)
g = lande_gJ_bound_single_electron(-1, 1)
print(f"zeeman.lande bound H 1s: {g:.9f}  (CODATA 2.002283845)")
# consistency of the two modules' Dirac values
print(f"g_factor_dirac_hydrogenic(1,1,-1) = {g_factor_dirac_hydrogenic(1,1,-1):.9f}")
print(f"lande_gJ_hydrogenic(-1,1)         = {lande_gJ_hydrogenic(-1,1):.9f}")
# g_factor_qed ignores kappa:
print("g_factor_qed kappa-dependence:", g_factor_qed(1,-1), g_factor_qed(1,-2), g_factor_qed(92,-1))
