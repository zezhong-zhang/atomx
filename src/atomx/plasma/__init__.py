"""Plasma physics: screening models and collisional-radiative modeling.

Provides:
- Debye-Huckel screening
- Ion-sphere model
- Stewart-Pyatt interpolation
- Basic collisional-radiative model (CRM)
- Conservative dense/CSR rate generators and explicit-rate sparse propagation
- Fixed-charge LTE/non-LTE slab coupling and explicit multi-charge LTE EOS,
  layered radiation/wave deposition and smooth Lagrangian source/conduction
- Finite physical-resonance multi-charge nonLTE energy closure (nonlte_source);
  conventions and physical limits: docs/EUV_SOURCE_MODEL.md
"""
