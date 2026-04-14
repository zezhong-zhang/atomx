"""Manim Scene classes for atomx result objects.

Each class is a normal ``manim.Scene`` subclass that takes an atomx
result in its constructor and builds the animation in ``construct``.
They are intended for offline rendering via the manim CLI:

    manim -pqh my_scene.py  RadialOrbitalScene

Pattern
-------
Because the manim CLI instantiates a ``Scene`` without arguments, our
factory-style design uses a module-level constructor:

>>> from atomx.viz.manim import RadialOrbitalScene
>>> from atomx import Atom
>>> atom = Atom(Z=10).solve()
>>>
>>> class NeRadial(RadialOrbitalScene):
...     wavefunctions = atom.wavefunctions
...     orbital_label = "2p3/2"

Then run:  ``manim -pqh ne_radial.py NeRadial``.

Available scenes
----------------
* :class:`RadialOrbitalScene` — animate P(r), Q(r) building up along r.
* :class:`GrotrianScene`      — levels appearing one at a time, then
  transition arrows added.
* :class:`SpectrumScene`      — Voigt profiles accumulating to form a
  synthetic spectrum.
* :class:`QEDBreakdownScene`  — bar-chart of corrections appearing
  sequentially with running-total annotation.
* :class:`SCFConvergenceScene` — residual curve drawn iteration by
  iteration on a log axis.
* :class:`DRScene`            — free electron capture → doubly-excited
  state → Auger / radiative decay cartoon.

Import is guarded so that ``atomx.viz.manim`` can be loaded on systems
without manim (attribute access fails with a helpful error).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

try:
    import manim as _manim
    from manim import (
        AnimationGroup,
        Arrow,
        Axes,
        BarChart,
        Create,
        DashedVMobject,
        FadeIn,
        FadeOut,
        Line,
        MathTex,
        NumberLine,
        Scene,
        Text,
        VGroup,
        Write,
    )
    _MANIM_AVAILABLE = True
except ImportError as _ie:
    _manim = None
    _MANIM_AVAILABLE = False
    _MANIM_IMPORT_ERROR = _ie

    # Define a no-op stub so `class Foo(Scene)` at module import time
    # still works when manim is not installed — the user will only hit
    # the error when they actually try to use the class.
    class Scene:   # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise RuntimeError(
                "manim is required for atomx.viz.manim scene classes "
                "but is not installed: "
                f"{_MANIM_IMPORT_ERROR}\n"
                "Install with:  pip install manim-community"
            )


# ---------------------------------------------------------------------------
# Radial orbital
# ---------------------------------------------------------------------------

class RadialOrbitalScene(Scene):
    """Animate P(r) and Q(r) for a single orbital.

    Override class attributes in a subclass, or instantiate with
    keyword arguments.  Attributes:

    wavefunctions : :class:`atomx.results.Wavefunctions`
    orbital_label : str
    r_max_bohr : float or None   (default: 1.5 × orbital expectation)
    show_small_component : bool  (default True)
    """

    wavefunctions: Any = None
    orbital_label: Optional[str] = None
    r_max_bohr: Optional[float] = None
    show_small_component: bool = True

    def construct(self):
        if not _MANIM_AVAILABLE:  # pragma: no cover
            raise RuntimeError("manim not installed")
        if self.wavefunctions is None or self.orbital_label is None:
            raise ValueError(
                "Subclass RadialOrbitalScene and set .wavefunctions + "
                ".orbital_label class attributes."
            )
        orb = self.wavefunctions.orbitals[self.orbital_label]
        r = np.asarray(self.wavefunctions.grid)
        P = np.asarray(orb.P)
        Q = np.asarray(orb.Q) if orb.Q is not None else None

        r_max = self.r_max_bohr or float(min(r.max(), 10.0))
        mask = r <= r_max
        r = r[mask]
        P = P[mask]
        if Q is not None:
            Q = Q[mask]

        y_max = float(max(np.abs(P).max(), np.abs(Q).max() if Q is not None else 0.0))

        axes = Axes(
            x_range=[0, float(r_max), max(float(r_max) / 5, 0.5)],
            y_range=[-y_max * 1.1, y_max * 1.1, y_max / 2],
            x_length=8, y_length=4.5,
            tips=False,
        )
        x_label = axes.get_x_axis_label("r  (Bohr)")
        y_label = axes.get_y_axis_label("")
        title = Text(
            f"{self.orbital_label}    ε = {orb.energy:.6f} Ha",
            font_size=24,
        ).to_edge(_manim.UP)

        P_graph = axes.plot_line_graph(
            x_values=r.tolist(), y_values=P.tolist(),
            add_vertex_dots=False, line_color=_manim.BLUE,
        )
        Q_graph = None
        if self.show_small_component and Q is not None:
            Q_graph = axes.plot_line_graph(
                x_values=r.tolist(), y_values=Q.tolist(),
                add_vertex_dots=False, line_color=_manim.RED,
            )

        self.play(Create(axes), Write(x_label), Write(title))
        self.play(Create(P_graph), run_time=2)
        if Q_graph is not None:
            self.play(Create(Q_graph), run_time=1.5)
        self.wait(1)


# ---------------------------------------------------------------------------
# Grotrian
# ---------------------------------------------------------------------------

class GrotrianScene(Scene):
    """Build a Grotrian diagram: levels appear, then transitions.

    Attributes
    ----------
    levels : sequence
        Each entry has ``energy`` (numeric, axis units) and optionally
        ``label`` and ``L``.
    transitions : sequence, optional
        Entries with ``upper``, ``lower``, ``wavelength_nm``.
    energy_unit : str
    """

    levels: Any = None
    transitions: Any = None
    energy_unit: str = "eV"

    def construct(self):
        if not _MANIM_AVAILABLE:  # pragma: no cover
            raise RuntimeError("manim not installed")
        if self.levels is None:
            raise ValueError("Set GrotrianScene.levels class attribute.")

        def _get(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default

        energies = [float(_get(lv, "energy", 0.0)) for lv in self.levels]
        L_vals = [int(_get(lv, "L", 0)) for lv in self.levels]
        labels = [str(_get(lv, "label", "")) for lv in self.levels]
        unique_L = sorted(set(L_vals))
        L_to_x = {L: i for i, L in enumerate(unique_L)}

        axes = Axes(
            x_range=[-0.5, len(unique_L) - 0.5, 1],
            y_range=[min(energies) - 1, max(energies) + 1,
                     (max(energies) - min(energies)) / 4 or 1],
            x_length=7, y_length=5, tips=False,
        )
        y_label = axes.get_y_axis_label(f"E ({self.energy_unit})")
        self.play(Create(axes), Write(y_label))

        level_lines = []
        for E, L, lbl in zip(energies, L_vals, labels):
            x = L_to_x[L]
            start = axes.coords_to_point(x - 0.35, E)
            end = axes.coords_to_point(x + 0.35, E)
            seg = Line(start, end, color=_manim.BLUE, stroke_width=3)
            text = Text(lbl, font_size=14).next_to(seg, _manim.RIGHT, buff=0.1)
            self.play(Create(seg), Write(text), run_time=0.3)
            level_lines.append((seg, E, x))

        if self.transitions:
            for tr in self.transitions:
                u = int(_get(tr, "upper"))
                lo = int(_get(tr, "lower"))
                if not (0 <= u < len(self.levels)) or not (0 <= lo < len(self.levels)):
                    continue
                _, E_u, x_u = level_lines[u]
                _, E_l, x_l = level_lines[lo]
                start = axes.coords_to_point(x_u, E_u)
                end = axes.coords_to_point(x_l, E_l)
                arrow = Arrow(start=start, end=end,
                              color=_manim.RED, buff=0, stroke_width=2)
                self.play(Create(arrow), run_time=0.5)
                lam = _get(tr, "wavelength_nm", None)
                if lam is not None:
                    mid = axes.coords_to_point(
                        0.5 * (x_u + x_l), 0.5 * (E_u + E_l))
                    txt = Text(f"{float(lam):.1f} nm",
                               font_size=12, color=_manim.RED).move_to(mid)
                    self.play(Write(txt), run_time=0.3)
        self.wait(2)


# ---------------------------------------------------------------------------
# Spectrum build-up
# ---------------------------------------------------------------------------

class SpectrumScene(Scene):
    """Animate a synthetic spectrum being built line by line.

    Attributes
    ----------
    spectrum_factory : callable
        Zero-argument callable returning either:
        * a :class:`~atomx.spectrum.Spectrum` instance (final spectrum,
          rendered in one sweep), or
        * a sequence of ``(label, Spectrum)`` pairs for a step-by-step
          build-up.
    """

    spectrum_factory: Any = None

    def construct(self):
        if not _MANIM_AVAILABLE:  # pragma: no cover
            raise RuntimeError("manim not installed")
        if self.spectrum_factory is None:
            raise ValueError(
                "Set SpectrumScene.spectrum_factory class attribute "
                "(a callable returning a Spectrum or list of "
                "(label, Spectrum) pairs)."
            )
        built = self.spectrum_factory()
        frames = built if isinstance(built, list) else [("spectrum", built)]

        first_spec = frames[0][1]
        wl = np.asarray(first_spec.wavelength_nm)
        y_max = max(float(np.asarray(spec.values).max()) for _, spec in frames)

        axes = Axes(
            x_range=[float(wl.min()), float(wl.max()),
                     (float(wl.max()) - float(wl.min())) / 5],
            y_range=[0, y_max * 1.1, y_max / 4],
            x_length=10, y_length=5, tips=False,
        )
        x_label = axes.get_x_axis_label("λ (nm)")
        y_label = axes.get_y_axis_label("intensity")
        self.play(Create(axes), Write(x_label), Write(y_label))

        for lbl, spec in frames:
            y = np.asarray(spec.values)
            graph = axes.plot_line_graph(
                x_values=wl.tolist(), y_values=y.tolist(),
                add_vertex_dots=False, line_color=_manim.TEAL,
            )
            title = Text(lbl, font_size=18).to_edge(_manim.UP)
            self.play(FadeIn(title), Create(graph), run_time=1.0)
            self.wait(0.5)
            self.play(FadeOut(title, graph), run_time=0.5)

        # Final: show the last frame and hold
        final = frames[-1][1]
        graph = axes.plot_line_graph(
            x_values=wl.tolist(),
            y_values=np.asarray(final.values).tolist(),
            add_vertex_dots=False, line_color=_manim.TEAL,
        )
        self.play(Create(graph))
        self.wait(2)


# ---------------------------------------------------------------------------
# QED breakdown (bar chart)
# ---------------------------------------------------------------------------

class QEDBreakdownScene(Scene):
    """Stacked-bar animation of an energy-components dict.

    Attributes
    ----------
    components : dict
        ``{name: value_Hartree}``  (typically ``result.energy_components``).
    """

    components: Any = None

    def construct(self):
        if not _MANIM_AVAILABLE:  # pragma: no cover
            raise RuntimeError("manim not installed")
        if self.components is None:
            raise ValueError("Set QEDBreakdownScene.components.")

        items = [(k, v) for k, v in self.components.items()
                 if isinstance(v, (int, float))]
        names = [k for k, _ in items]
        values = [float(v) for _, v in items]

        chart = BarChart(
            values=values, bar_names=names,
            y_range=[min(min(values), 0), max(max(values), 0) * 1.1,
                     (max(values) - min(values)) / 5 or 1],
            x_length=10, y_length=5,
        )
        title = Text("QED / correction breakdown",
                     font_size=22).to_edge(_manim.UP)
        self.play(FadeIn(title), Create(chart), run_time=1.5)
        self.wait(2)


# ---------------------------------------------------------------------------
# SCF convergence
# ---------------------------------------------------------------------------

class SCFConvergenceScene(Scene):
    """Animate the residual curve of an SCF iteration history.

    Attributes
    ----------
    residuals : sequence of float
    energies : sequence of float, optional
    """

    residuals: Any = None
    energies: Any = None

    def construct(self):
        if not _MANIM_AVAILABLE:  # pragma: no cover
            raise RuntimeError("manim not installed")
        if self.residuals is None:
            raise ValueError("Set SCFConvergenceScene.residuals.")

        r = np.asarray(self.residuals, dtype=float)
        n = len(r)
        # Use log-residual on a linear axis for manim simplicity
        log_r = np.log10(np.maximum(r, 1e-16))

        axes = Axes(
            x_range=[0, n + 1, max(1, n // 10)],
            y_range=[float(log_r.min()) - 1, float(log_r.max()) + 1, 2],
            x_length=8, y_length=4.5, tips=False,
        )
        axes.add_coordinates()
        title = Text("SCF convergence (log₁₀ residual)",
                     font_size=22).to_edge(_manim.UP)
        self.play(Create(axes), Write(title))

        points = [axes.coords_to_point(i + 1, log_r[i]) for i in range(n)]
        lines = []
        for i in range(1, n):
            seg = Line(points[i - 1], points[i],
                       color=_manim.BLUE, stroke_width=3)
            lines.append(seg)
        for seg in lines:
            self.play(Create(seg), run_time=0.2)
        self.wait(2)


# ---------------------------------------------------------------------------
# Dielectronic recombination cartoon
# ---------------------------------------------------------------------------

class DRScene(Scene):
    """Cartoon of a dielectronic-recombination event.

    Attributes
    ----------
    ion_charge : int
        Residual ion charge (for visual label only).
    resonance_label : str
        Label of the doubly-excited intermediate.
    gamma_auger : float
        Auger width (s⁻¹) — determines line colour intensity.
    gamma_radiative : float
        Radiative width (s⁻¹) — competing decay.
    """

    ion_charge: int = 2
    resonance_label: str = "d*"
    gamma_auger: float = 1.0e13
    gamma_radiative: float = 1.0e13

    def construct(self):
        if not _MANIM_AVAILABLE:  # pragma: no cover
            raise RuntimeError("manim not installed")

        # Three horizontal levels (initial ion, resonance, recombined final)
        h_init = -2.0
        h_res = 0.5
        h_final = -3.0

        line_init = Line([-3, h_init, 0], [3, h_init, 0],
                         color=_manim.BLUE, stroke_width=3)
        line_res = Line([-3, h_res, 0], [3, h_res, 0],
                        color=_manim.YELLOW, stroke_width=3)
        line_final = Line([-3, h_final, 0], [3, h_final, 0],
                          color=_manim.GREEN, stroke_width=3)

        lbl_init = Text(f"ion X^{{{self.ion_charge}+}}",
                        font_size=18).next_to(line_init, _manim.RIGHT)
        lbl_res = Text(self.resonance_label,
                       font_size=18).next_to(line_res, _manim.RIGHT)
        lbl_final = Text(f"X^{{{self.ion_charge - 1}+}}*",
                         font_size=18).next_to(line_final, _manim.RIGHT)

        self.play(Create(line_init), Create(line_res), Create(line_final),
                  Write(lbl_init), Write(lbl_res), Write(lbl_final))

        # Step 1: free electron "falls in" from above
        e_start = [-4, h_res + 2.5, 0]
        e_landing = [-2, h_res, 0]
        electron = _manim.Dot(e_start, color=_manim.WHITE, radius=0.1)
        self.play(FadeIn(electron))
        self.play(electron.animate.move_to(e_landing), run_time=1.5)

        # Capture arrow from init to resonance
        capture = Arrow([-2, h_init, 0], [-2, h_res, 0],
                        color=_manim.YELLOW, buff=0.05, stroke_width=3)
        cap_txt = Text("e⁻ capture", font_size=14, color=_manim.YELLOW)
        cap_txt.next_to(capture, _manim.LEFT)
        self.play(Create(capture), Write(cap_txt))
        self.wait(0.5)

        # Step 2: radiative decay to the final singly-bound state
        stab = Arrow([1.5, h_res, 0], [1.5, h_final, 0],
                     color=_manim.GREEN, buff=0.05, stroke_width=3)
        photon = Text("hν", font_size=16, color=_manim.GREEN)
        photon.next_to(stab, _manim.RIGHT)
        self.play(Create(stab), Write(photon))
        self.wait(1.5)

        # Branching ratio annotation
        total = self.gamma_auger + self.gamma_radiative
        eta = self.gamma_radiative / total if total > 0 else 0
        br_txt = Text(
            f"fluorescence yield = {eta:.2f}",
            font_size=16, color=_manim.WHITE,
        ).to_edge(_manim.DOWN)
        self.play(Write(br_txt))
        self.wait(2)
