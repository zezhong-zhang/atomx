import sys
sys.path.insert(0, "/data/code/atomx/src")
from atomx import Atom

h = Atom(1, config="1s1", corrections="none")   # shooting default fine here? need bspline for field shift
h2 = Atom(1, config="1s1", engine="bspline", corrections="none")
h2.solve()
res = h2._result
fs = res.field_shift_parameter()
print("H 1s F:", fs.F_MHz_per_fm2, "Ha/Bohr2:", fs.F_Ha_per_Bohr2, "(expect 2/3 Ha/Bohr2 = 1.5666 MHz/fm2)")
iso = res.isotope_shift((1, 1), (1, 2))
print(f"H-D 1s level shift: NMS={iso.E_NMS_MHz:.1f} MHz  FS={iso.E_FS_MHz:.3f} MHz  total={iso.E_total_MHz:.1f} MHz")
# anchors: NMS level shift = +0.5*(1/M_H - 1/M_D) Ha = +1.3609e-4 Ha = +8.955e5 MHz
# FS = F*delta<r^2> = 1.5666 * (2.1421^2 - 0.8783^2) = 1.5666*3.8172 = 5.98 MHz
print("anchors: NMS=+8.955e5 MHz (level, lighter H above D), FS=+5.98 MHz")
