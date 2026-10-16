Complete shared Numba CPU/CUDA contraction control

This direction reuses the existing `numba_density_control.py` CPU/CUDA owner
for all 2688 native tagged density nodes, all 25 ordered pairs of the five
retained physical radial orbitals and every rank 0..3. Its independent
reference is the archived public CellCoulombFeatures weighted action.

New private `pair_ladder_numba.py` uses one ordered scalar algorithm for the
complete source-owner PP/HH diagram contractions of direction 3: DC/static/
dynamic, all 10 scalar doubles and all 10 basis directions plus real/imag
parts of one signed amplitude vector. Source is retained by the same host
physical owner, not recomputed by the GPU kernel. The private control
explicitly rejects complex interaction matrices; production complex CPU
phase gates are owned by direction 3 and are not silently cast to real.

Both targets reuse existing scoped Numba `_workers` and `_add`/`_mul`
arithmetic. CPU has fastmath=False; CUDA uses explicit IEEE dadd_rn/dmul_rn.
Actual compiled PTX is audited for FMA/atomic/reduction instructions. Every
CPU/CUDA output coordinate and signed basis column must be bitwise equal,
and both must retain the original 1e-11 complete numerical/Ha gates.
No subset/norm-only comparison, coordinate removal or backend fallback is
allowed. Allocation rejects before any kernel/device allocation if the
complete matrix/batch exceeds its explicit payload cap.

Initial automatic001 retained the actual libcudart loader failure; it did
not prove a virtual-address or hardware failure. The probe now reuses the
existing scoped numba_cuda_runtime CUDA_PATH/library path via same-PID exec
startup, preserving every bridge environment/profile and inherited hard
limit. Automatic002/003 then ran real CUDA under the original hard 4 GiB
address cap. The separate prospective 128 GiB manual GPU contract was not
needed and was never executed for automatic admission.

Observed device-owned arrays: 1,361,840 bytes, below the declared 32 MiB cap.
Includes the density owner and all three live DC/static/dynamic PP/HH owners.
Driver/context/JIT address and RSS overhead are separately recorded and
constrained by the process 4 GiB limit. No device pool/global thread policy
is modified. A bounded device0 flock prevents another instance of this probe
running a CUDA workload concurrently. Actual loader module and mapped CUDA
library hashes are recorded; each PTX is preserved with its signature/hash.

Cold ladder compilation, device allocation, geometry upload, complete
amplitude upload and complete separated output download are measured
separately. Existing density constructor retains its combined geometry/
allocation/upload scope and is labelled accordingly. Five alternating warm
pairs compare full CPU and transfer-inclusive GPU apply_into calls; resident
CPU/GPU observations have explicit separate scopes. Device events bracket
20 kernels and intervening host launch gaps, not a pure single-kernel
duration. The 10x12 physical ladder is deliberately tiny and triggers low
occupancy warnings. These data cannot establish large native SD speed or
end-to-end acceleration.

Desktop GPU compute PIDs 241814 and 345497 remained present before/after.
The strict no-other-GPU-process timing guard therefore failed; speedup=null.
Correctness and payload/address admission passed independently. No controlled
GPU speedup or highest-level certification is claimed.

Replay: `.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_six_gpu001/probe.py --output <fresh-directory>`.
The automatic entry hard caps 4 GiB AS, 120 CPU seconds, one CPU/BLAS/Numba
worker and 32 MiB total owned GPU payload. It imports the complete fixed input
from direction 3, copies/archives it, archives loaded scientific sources,
writes complete CPU/CUDA output arrays, PTX and result.json. Tests are CPU-only
and reject complex/partial inputs, aliased separated outputs and resource
overflow before CUDA selection.

Remaining: big physical full-SD/FS-CCSDT contractions, live-device batching
and overlap at admitted complete dimensions, MPI/multi-GPU distribution,
full residual/solve/response and controlled exclusive-device timing. The
new control is not adopted as a production GPU SD backend.
