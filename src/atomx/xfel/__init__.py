"""XFEL-pulse physics: sequential-photoionization cascades and
transient-configuration observables (see docs/XFEL_GAP_ANALYSIS.md)."""

from atomx.xfel.atomic_rates import build_atomic_rate_table
from atomx.xfel.cascade import (
    MCResult,
    Pulse,
    RateProvider,
    TableRateProvider,
    build_rate_table,
    cascade,
    cascade_mc,
)
from atomx.xfel.coherent import (
    CoherentDynamics,
    CoherentGaussianPulse,
    CoherentSystem,
    NativeOneElectronCoherentModel,
)
from atomx.xfel.cross_sections import NativeSubshellCrossSection
from atomx.xfel.double_ionization import add_direct_double_ionization
from atomx.xfel.imaging import ImagingSignal, ScatteringFactorBank, native_form_factor
from atomx.xfel.kinetics import (
    GaussianPulse,
    KineticGraph,
    KineticTransition,
    PopulationDynamics,
    TabulatedPulse,
)
from atomx.xfel.mcdhf_coherent import MCDHFCoherentModel
from atomx.xfel.model import (
    NativeRateArchive,
    NativeXFELModel,
    configuration_resonances,
    configuration_weight,
    retabulate_native_pi,
)
from atomx.xfel.multielectron_coherent import MultiElectronCoherentModel
from atomx.xfel.native_channels import add_fluorescence, add_photo_shakeoff
from atomx.xfel.observables import XFELSimulation, build_native_scattering_bank
from atomx.xfel.resonance import ResonantLine, add_resonances
from atomx.xfel.table_mc import CompiledRateTable, TableMCResult
from atomx.xfel.two_electron_coherent import TwoElectronCoherentModel

__all__ = [
    "MCResult",
    "Pulse",
    "RateProvider",
    "TableRateProvider",
    "build_rate_table",
    "build_atomic_rate_table",
    "cascade",
    "cascade_mc",
    "CompiledRateTable",
    "TableMCResult",
    "NativeSubshellCrossSection",
    "CoherentDynamics",
    "CoherentGaussianPulse",
    "CoherentSystem",
    "NativeOneElectronCoherentModel",
    "MCDHFCoherentModel",
    "TwoElectronCoherentModel",
    "MultiElectronCoherentModel",
    "GaussianPulse",
    "TabulatedPulse",
    "KineticGraph",
    "KineticTransition",
    "PopulationDynamics",
    "ResonantLine",
    "add_resonances",
    "add_fluorescence",
    "add_photo_shakeoff",
    "add_direct_double_ionization",
    "ScatteringFactorBank",
    "ImagingSignal",
    "native_form_factor",
    "NativeRateArchive",
    "NativeXFELModel",
    "configuration_resonances",
    "configuration_weight",
    "retabulate_native_pi",
    "XFELSimulation",
    "build_native_scattering_bank",
]
