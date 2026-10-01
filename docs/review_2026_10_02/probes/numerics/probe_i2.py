from atomx import Atom
a = Atom(1, config='6s1', engine='bspline', nuclear='point', corrections='none').solve()
print("bspline box r_max actually used:", a.result._bsp_data['t'][-1])
print("Basis docstring claims auto box: max(75, 5*n^2/Z) =", max(75, 5*36/1))
