"""NEW: average-atom sanity -- electron budget, T/ne trends, docstring claims."""
import numpy as np

from atomx.plasma.average_atom import average_atom_thomas_fermi


def main():
    print("== Fe, ne = 1e20, scan T ==")
    print(f"{'T(eV)':>9} {'Z_bar':>8} {'sum occ':>9} {'saturated':>10}")
    for T in (0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 2.0e4):
        r = average_atom_thomas_fermi(26, T, 1e20)
        tot = sum(r.occupations.values())
        print(f"{T:9.2f} {r.Z_bar:8.3f} {tot:9.4f} {str(r.budget_saturated):>10}")

    print("== Fe, T = 100 eV, scan ne (pressure ionization trend) ==")
    for ne in (1e16, 1e18, 1e20, 1e22, 1e24):
        r = average_atom_thomas_fermi(26, 100.0, ne)
        print(f"ne={ne:8.1e}  Z_bar={r.Z_bar:8.3f}  mu={r.mu_eV:10.3f} eV")

    # Docstring claim check: 'ne = Z* . n_ion' is never enforced and n_ion
    # is not an input.  Demonstrate: same ne at two different 'n_ion'
    # interpretations returns identical Z_bar (no n_ion argument exists).
    import inspect
    from atomx.plasma import average_atom
    sig = inspect.signature(average_atom_thomas_fermi)
    print("signature params:", list(sig.parameters))
    src = inspect.getsource(average_atom_thomas_fermi)
    print("mentions n_ion in code body:", "n_ion" in src)

    # budget closure in saturated case
    r = average_atom_thomas_fermi(26, 0.01, 1e20)
    print(f"saturated: sum occ = {sum(r.occupations.values()):.12f}, "
          f"Z_bar = {r.Z_bar:.2e}")


if __name__ == "__main__":
    main()
