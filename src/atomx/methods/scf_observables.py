"""Derived-observable methods for :class:`SCFResult` (hyperfine, isotope
shift, Zeeman), extracted from the ``dirac_fock`` god-file.

These are thin wrappers that defer into ``atomx.properties.*`` to compute
observables from a converged SCF result.  Kept as a mixin so ``SCFResult``'s
public API is unchanged (``result.hyperfine_constants(...)`` etc. still work
via inheritance) while ``dirac_fock.py`` shrinks by ~520 lines.
"""

from __future__ import annotations

from atomx.data.hyperfine_presets import TABLE as _HYPERFINE_PRESET_TABLE


def _hyperfine_preset(name: str):
    """Return ``(a_e_on, delta_qed_hf, delta_recoil, delta_nuc)`` for ``name``."""
    try:
        p = _HYPERFINE_PRESET_TABLE[name]
    except KeyError:
        raise KeyError(
            f"Unknown hyperfine preset {name!r}. "
            f"Known presets: {sorted(_HYPERFINE_PRESET_TABLE)}. "
            f"To add a new preset, update atomx.data.hyperfine_presets.TABLE."
        ) from None
    return (p.a_e_on, p.delta_qed_hf, p.delta_recoil, p.delta_nuc)


class SCFObservablesMixin:
    """Mixin providing derived-observable methods for :class:`SCFResult`.

    Assumes the host class supplies the SCF-result attributes these methods
    reference (``orbitals_raw``, ``orbital_energies``, ``_bsp_data``, ``Z``,
    ``subshells``, ``N_electrons``, ...).
    """

    def hyperfine_constants(
        self,
        isotope=None,
        orbitals=None,
        *,
        include_anomalous_moment: bool = False,
        include_finite_nuclear_mass: bool = False,
        qed_hf_factor=None,
        recoil_factor=None,
        nuclear_structure_factor=None,
        correlation_factor=None,
        preset: str | None = None,
    ):
        """Compute magnetic-dipole *and* electric-quadrupole hyperfine
        constants on the ladder.

        Walks a ladder of multiplicative corrections on top of the
        single-configuration Dirac–Fock baseline (infinite-nuclear-mass
        Dirac equation):

            dhf
              → finite_nuclear_mass   (M/(M+m_e))^3  — reduced-mass
                                                       correction
              → ae                    (1 + a_e)      — electron anomaly
              → qed_hf                (1 + δ_QEDhf)  — higher-order QED +
                                                       Breit on HF operator
              → recoil                (1 + δ_recoil) — recoil QED
              → nuclear_structure     (1 + δ_nuc)    — Zemach / proton
                                                       structure /
                                                       Bohr–Weisskopf
              → correlation           (1 + δ_corr)   — electron correlation

        Each rung is **optional** and defaults to disabled.  This is
        deliberate: the individual QED corrections nearly cancel each
        other for ground-state s-shells, so applying only a subset can
        *worsen* the prediction (see
        :mod:`atomx.properties.hyperfine.dipole` module docstring).

        The electric-quadrupole constant ``B_MHz`` is always computed
        (when the nucleus has Q ≠ 0 and the orbital has j > 1/2) from
        the same DHF radial integrals, without ladder corrections —
        the B ladder is future work.

        Preset coefficients
        -------------------
        Pass ``preset="h1s"`` / ``"d1s"`` / ``"he3_1s"`` to inject
        published Karshenboim 2005 values for ``a_e``, ``δ_QEDhf``,
        ``δ_recoil`` — the sum reproduces the 1 ppm experimental H 1s
        splitting.  Any explicit ``_factor`` kwarg overrides the
        preset.

        Parameters
        ----------
        isotope : tuple(int, int) or None
            ``(Z, A)`` isotope specifier.
        orbitals : list of str or None
            Orbital labels to include; default = all occupied orbitals.
        include_anomalous_moment : bool, keyword-only
            Apply ``(1 + a_e)``, ``a_e = 1.15965218e-3``.  Default ``False``.
        include_finite_nuclear_mass : bool, keyword-only
            Apply the reduced-mass factor ``(M/(M+m_e))³`` where M is
            the nuclear mass (``A × m_p`` with tabulated A).  This is
            the dominant "recoil" correction for light atoms — for
            hydrogen it lowers A by ~0.16 %.  DHF assumes infinite M;
            turning this on shifts the baseline to the proper finite-M
            Dirac result.  Default ``False``.
        qed_hf_factor : float, dict or None, keyword-only
            ``δ_QEDhf`` — higher-order QED + Breit correction to the
            hyperfine operator itself.  Accepts a scalar or per-orbital
            dict (missing labels → 0).  For H 1s, Karshenboim 2005 gives
            δ_QEDhf ≈ −1.268 × 10⁻³ (≈ −1.80 MHz), which nearly cancels
            the Schwinger +1.65 MHz.  For general atoms no closed-form
            exists; users plug in values from the literature or a
            dedicated QED code.
        recoil_factor : float, dict or None, keyword-only
            ``δ_recoil`` — recoil-QED correction (≠ the classical
            reduced-mass factor which has its own rung).  For H 1s,
            Karshenboim 2005 gives the residual recoil QED as ≈ −7 ppm
            on top of the (M/(M+m_e))³ classical piece.
        nuclear_structure_factor : float, dict or None, keyword-only
            ``δ_nuc`` — nuclear-structure correction: Zemach (finite
            charge & magnetization distributions), proton polarizability,
            Bohr-Weisskopf.  For H 1s, Karshenboim 2005 gives
            δ_nuc ≈ −40 ppm (Zemach dominant).
        correlation_factor : float, dict or None, keyword-only
            ``δ_corr`` — electron correlation / core polarization.  A
            placeholder slot until atomx gains a dedicated RPA /
            Sternheimer module; published alkali values for valence
            s-shells are ``δ_corr ≈ 0.30`` (Dzuba–Flambaum–Sushkov 1984).

        Returns
        -------
        HyperfineResult
        """
        from atomx.constants import PROTON_ELECTRON_MASS_RATIO
        from atomx.properties.hyperfine import (
            ELECTRON_ANOMALOUS_MOMENT,
            hyperfine_A_MHz,
            hyperfine_B_MHz,
            nuclear_moment,
        )
        from atomx.results import HyperfineResult

        ELECTRON_ANOMALOUS_MOMENT_CODATA = ELECTRON_ANOMALOUS_MOMENT

        if self._bsp_data is None or not self.orbitals_raw:
            raise RuntimeError(
                "Hyperfine calculation requires retained orbitals "
                "(orbitals_raw + _bsp_data); re-run solve().")

        if isotope is not None:
            z, a = isotope
            mom = nuclear_moment(z, a)
        elif self.Z is not None:
            mom = nuclear_moment(self.Z)
        else:
            raise ValueError(
                "Cannot infer isotope: SCFResult.Z is None — pass an "
                "explicit (Z, A) via the `isotope` argument.")

        # --- Apply preset, if any -----------------------------------
        if preset is not None:
            ae_p, qed_p, rec_p, nuc_p = _hyperfine_preset(preset)
            if not include_anomalous_moment:
                include_anomalous_moment = True
            if not include_finite_nuclear_mass:
                include_finite_nuclear_mass = True
            if qed_hf_factor is None:
                qed_hf_factor = qed_p
            if recoil_factor is None:
                recoil_factor = rec_p
            if nuclear_structure_factor is None:
                nuclear_structure_factor = nuc_p
            # preset leaves correlation_factor under user control.

        # --- Build orbital label → (P, Q, kappa) map ---------------
        orb_map = {}
        for sh in self.subshells:
            orb_list = self.orbitals_raw.get(sh.kappa, [])
            idx = sh.n - 1 - sh.l
            if idx < len(orb_list):
                P, Q, _occ = orb_list[idx]
                orb_map[sh.label] = (P, Q, sh.kappa)

        labels = orbitals or list(orb_map.keys())

        # Finite-nuclear-mass factor: nuclear mass in m_e units, then (M/(M+1))^3
        # (We use the tabulated mass number A and the proton-electron mass ratio
        # as a proxy for the actual nuclear mass in m_e units; this is exact for
        # H-1 and ≤ 0.05 % accurate for other light isotopes.)
        if include_finite_nuclear_mass:
            M_over_me = mom.A * PROTON_ELECTRON_MASS_RATIO
            fnm_factor = (M_over_me / (M_over_me + 1.0)) ** 3
        else:
            fnm_factor = 1.0

        ladder = ["dhf"]
        if include_finite_nuclear_mass:
            ladder.append("finite_nuclear_mass")
        if include_anomalous_moment:
            ladder.append("ae")
        if qed_hf_factor is not None:
            ladder.append("qed_hf")
        if recoil_factor is not None:
            ladder.append("recoil")
        if nuclear_structure_factor is not None:
            ladder.append("nuclear_structure")
        if correlation_factor is not None:
            ladder.append("correlation")

        def _lookup_factor(f, label):
            if f is None:
                return 0.0
            if isinstance(f, dict):
                return float(f.get(label, 0.0))
            return float(f)

        A_MHz_ladder = {}
        A_MHz_final = {}
        B_MHz_final = {}
        for lbl in labels:
            if lbl not in orb_map:
                continue
            P, Q, kappa = orb_map[lbl]

            # Rung 0: pure DHF (no anomaly, no anything)
            A_dhf = hyperfine_A_MHz(
                P, Q, self._bsp_data, kappa=kappa, g_I=mom.g_I,
                include_anomalous_moment=False,
            )
            rung = {"dhf": A_dhf}
            A_val = A_dhf

            # Rung 1: × (M/(M+m_e))^3 finite-nuclear-mass factor
            if include_finite_nuclear_mass:
                A_val = A_val * fnm_factor
                rung["finite_nuclear_mass"] = A_val

            # Rung 2: + anomalous electron moment
            if include_anomalous_moment:
                A_val = A_val * (1.0 + ELECTRON_ANOMALOUS_MOMENT_CODATA)
                rung["ae"] = A_val

            # Rung 3: + higher-order QED + Breit correction to HF operator
            if qed_hf_factor is not None:
                A_val = A_val * (1.0 + _lookup_factor(qed_hf_factor, lbl))
                rung["qed_hf"] = A_val

            # Rung 4: + recoil-QED (on top of the classical reduced-mass)
            if recoil_factor is not None:
                A_val = A_val * (1.0 + _lookup_factor(recoil_factor, lbl))
                rung["recoil"] = A_val

            # Rung 5: + nuclear structure (Zemach, polarizability, BW)
            if nuclear_structure_factor is not None:
                A_val = A_val * (1.0 + _lookup_factor(nuclear_structure_factor, lbl))
                rung["nuclear_structure"] = A_val

            # Rung 6: + correlation / core polarization
            if correlation_factor is not None:
                A_val = A_val * (1.0 + _lookup_factor(correlation_factor, lbl))
                rung["correlation"] = A_val

            A_MHz_ladder[lbl] = rung
            A_MHz_final[lbl] = A_val

            # Electric quadrupole B (no ladder, bare DHF) — nonzero only
            # if nucleus has a quadrupole (I > 1/2) and orbital has j > 1/2.
            if mom.Q != 0.0 and abs(kappa) >= 2:
                B_MHz_final[lbl] = hyperfine_B_MHz(
                    P, Q, self._bsp_data, kappa=kappa, Q_nuc_barns=mom.Q,
                )

        return HyperfineResult(
            isotope=mom.name,
            Z=mom.Z,
            A=mom.A,
            I=mom.I,
            mu_I=mom.mu_I,
            Q_nuc=mom.Q,
            A_MHz=A_MHz_final,
            A_MHz_ladder=A_MHz_ladder,
            ladder=ladder,
            B_MHz=B_MHz_final,
        )

    # ------------------------------------------------------------------
    # Isotope shifts (NMS + field shift)
    # ------------------------------------------------------------------

    def mass_shift_NMS(self, isotope=None, *, kinetic_energy=None):
        """Normal mass shift of the *total energy* for a given isotope.

        Parameters
        ----------
        isotope : tuple(int, int) or None
            ``(Z, A)`` isotope specifier; defaults to the element's
            standard isotope.
        kinetic_energy : float, optional
            Explicit ⟨T_e⟩ in Hartree to use instead of the virial
            theorem ⟨T⟩ = −E_total.  For Dirac-Fock systems where
            the relativistic correction matters this can be obtained
            from one-electron energies:
            ``⟨T⟩ = Σ_a n_a ε_a − ⟨V_nuc⟩ − 2⟨V_ee⟩``.

        Returns
        -------
        atomx.properties.isotope_shift.MassShift
        """
        from atomx.properties.hyperfine import nuclear_moment
        from atomx.properties.isotope_shift import normal_mass_shift

        if isotope is not None:
            z, a = isotope
            mom = nuclear_moment(z, a)
        elif self.Z is not None:
            mom = nuclear_moment(self.Z)
        else:
            raise ValueError(
                "Cannot infer isotope: pass (Z, A) via the `isotope` argument."
            )

        return normal_mass_shift(
            total_energy=self.total_energy,
            mass_me=mom.mass_me,
            isotope_label=mom.name,
            kinetic_energy=kinetic_energy,
        )

    def mass_shift_SMS(self, K_SMS_au: float, isotope=None):
        """Specific mass shift for the given isotope, from a user K_SMS.

        The SMS two-body operator ``−(1/M) Σ_{i<j} p_i · p_j`` is not
        evaluated by atomx from scratch; users supply the coefficient
        ``K_SMS_au`` (atomic units, ``Hartree · m_e``) from a
        companion calculation (MBPT, CI, or an Anisimova 2023 model-
        operator table) and this method wraps it for isotope-shift
        bookkeeping.

        Parameters
        ----------
        K_SMS_au : float
            Specific-mass-shift coefficient in atomic units.
        isotope : tuple(int, int) or None
            ``(Z, A)`` isotope specifier; default from ``self.Z``.

        Returns
        -------
        atomx.properties.isotope_shift.SpecificMassShift
        """
        from atomx.properties.hyperfine import nuclear_moment
        from atomx.properties.isotope_shift import specific_mass_shift

        if isotope is not None:
            z, a = isotope
            mom = nuclear_moment(z, a)
        elif self.Z is not None:
            mom = nuclear_moment(self.Z)
        else:
            raise ValueError(
                "Cannot infer isotope: pass (Z, A) via the `isotope` argument."
            )

        return specific_mass_shift(
            K_SMS_au=K_SMS_au,
            mass_me=mom.mass_me,
            isotope_label=mom.name,
        )

    def field_shift_parameter(self, orbitals=None):
        """Per-orbital Seltzer field-shift factor F_a.

        Parameters
        ----------
        orbitals : list of str, optional
            Restrict to these orbital labels.  Default = every orbital.

        Returns
        -------
        atomx.properties.isotope_shift.FieldShiftParameter
        """
        from atomx.properties.isotope_shift import field_shift_factor_per_orbital

        if self._bsp_data is None or not self.orbitals_raw:
            raise RuntimeError(
                "Field-shift calculation requires retained orbitals; "
                "re-run solve()."
            )
        if self.Z is None:
            raise ValueError("SCFResult.Z is None — cannot evaluate F.")

        orb_list = []
        for sh in self.subshells:
            obl = self.orbitals_raw.get(sh.kappa, [])
            idx = sh.n - 1 - sh.l
            if idx < len(obl):
                P, Q, _occ = obl[idx]
                if orbitals is None or sh.label in orbitals:
                    orb_list.append((P, Q, sh.occupation, sh.kappa, sh.label))

        return field_shift_factor_per_orbital(orb_list, self._bsp_data, self.Z)

    def isotope_shift(self, isotope_a, isotope_b, *,
                      K_SMS_au: float | None = None,
                      kinetic_energy: float | None = None):
        """Isotope shift between two isotopes for this atomic level.

        Parameters
        ----------
        isotope_a, isotope_b : tuple(int, int)
            ``(Z, A)`` identifiers; must share the same Z.
        K_SMS_au : float, optional
            User-supplied specific-mass-shift coefficient in atomic
            units.  If ``None`` (default), the SMS term is zero and
            the returned shift contains NMS + FS only.
        kinetic_energy : float, optional
            Explicit ⟨T_e⟩ in Hartree; passed through to the NMS
            calculation.  Default: virial theorem ⟨T⟩ = −E_total.

        Returns
        -------
        atomx.properties.isotope_shift.IsotopeShiftResult
            ``δν^{A,B} = ν^A − ν^B`` with NMS + SMS + FS pieces
            separated.  When ``K_SMS_au=None``, ``E_SMS_MHz`` is zero.
        """
        from atomx.properties.hyperfine import nuclear_moment
        from atomx.properties.isotope_shift import (
            IsotopeShiftResult,
            normal_mass_shift,
            specific_mass_shift,
        )

        za, aa = isotope_a
        zb, ab = isotope_b
        if za != zb:
            raise ValueError(
                f"Both isotopes must have the same Z; got {za} and {zb}."
            )
        mom_a = nuclear_moment(za, aa)
        mom_b = nuclear_moment(zb, ab)

        # --- normal mass shift ---------------------------------------
        ms_a = normal_mass_shift(
            self.total_energy, mom_a.mass_me,
            isotope_label=mom_a.name, kinetic_energy=kinetic_energy,
        )
        ms_b = normal_mass_shift(
            self.total_energy, mom_b.mass_me,
            isotope_label=mom_b.name, kinetic_energy=kinetic_energy,
        )
        E_NMS_MHz = ms_a.E_NMS_MHz - ms_b.E_NMS_MHz

        # --- specific mass shift (user-supplied K_SMS) ---------------
        E_SMS_MHz = 0.0
        if K_SMS_au is not None:
            sms_a = specific_mass_shift(K_SMS_au, mom_a.mass_me,
                                         isotope_label=mom_a.name)
            sms_b = specific_mass_shift(K_SMS_au, mom_b.mass_me,
                                         isotope_label=mom_b.name)
            E_SMS_MHz = sms_a.E_SMS_MHz - sms_b.E_SMS_MHz

        # --- field shift ---------------------------------------------
        # δ⟨r²⟩ from tabulated r_rms (zero if unknown — then FS = 0)
        r2_a = mom_a.r_rms_fm ** 2 if mom_a.r_rms_fm > 0 else 0.0
        r2_b = mom_b.r_rms_fm ** 2 if mom_b.r_rms_fm > 0 else 0.0
        delta_r2 = r2_a - r2_b

        fs_par = self.field_shift_parameter()
        E_FS_MHz = fs_par.total_F_MHz_per_fm2 * delta_r2
        orbital_FS = {
            lbl: F * delta_r2 for lbl, F in fs_par.F_MHz_per_fm2.items()
        }

        E_total_MHz = E_NMS_MHz + E_SMS_MHz + E_FS_MHz

        return IsotopeShiftResult(
            isotope_a=mom_a.name,
            isotope_b=mom_b.name,
            mass_me_a=mom_a.mass_me,
            mass_me_b=mom_b.mass_me,
            r_rms_fm_a=mom_a.r_rms_fm,
            r_rms_fm_b=mom_b.r_rms_fm,
            delta_r2_fm2=delta_r2,
            E_NMS_MHz=E_NMS_MHz,
            E_FS_MHz=E_FS_MHz,
            E_SMS_MHz=E_SMS_MHz,
            E_total_MHz=E_total_MHz,
            orbital_FS_MHz=orbital_FS,
        )

    # ------------------------------------------------------------------
    # Zeeman (Landé g-factors + weak-field shifts)
    # ------------------------------------------------------------------

    def zeeman_factors(
        self,
        orbitals=None,
        *,
        include_anomalous_moment: bool = True,
    ):
        """Single-electron Landé g-factors for each orbital.

        Parameters
        ----------
        orbitals : list of str, optional
            Restrict to these orbital labels; default = all occupied.
        include_anomalous_moment : bool, keyword-only
            Include ``(1 + a_e)`` in the ``.bound`` entry (default ``True``).

        Returns
        -------
        dict
            ``{label: LandeFactor}`` — each entry has ``.nr``, ``.dirac``,
            ``.bound``.  For heavy multi-electron atoms, ``.bound`` is
            only a single-electron estimate; true many-electron Zeeman
            is a future project.
        """
        from atomx.properties.zeeman.lande import lande_factor

        if self.Z is None:
            raise ValueError("SCFResult.Z is None — cannot evaluate g_J.")

        out = {}
        for sh in self.subshells:
            if orbitals is not None and sh.label not in orbitals:
                continue
            out[sh.label] = lande_factor(
                kappa=sh.kappa, Z=self.Z, label=sh.label,
                include_anomalous_moment=include_anomalous_moment,
            )
        return out

    def zeeman_shifts(
        self,
        B_Tesla: float,
        orbitals=None,
        *,
        g_source: str = "bound",
    ):
        """Weak-field Zeeman shifts ``ΔE = g_J μ_B B m_J`` for each orbital.

        Parameters
        ----------
        B_Tesla : float
            External field strength in tesla.
        orbitals : list of str, optional
            Restrict to these orbital labels.
        g_source : str
            Which g refinement to use: ``"nr"``, ``"dirac"``, or
            ``"bound"`` (default).

        Returns
        -------
        dict
            ``{label: ZeemanShift}`` — each entry exposes ``.shifts_MHz``
            (a ``{m_J: shift_MHz}`` dict) and ``.splitting_MHz``.
        """
        from atomx.properties.zeeman import zeeman_shift_table

        lf_dict = self.zeeman_factors(orbitals=orbitals)
        return {
            label: zeeman_shift_table(lf, B_Tesla, g_source=g_source)
            for label, lf in lf_dict.items()
        }
