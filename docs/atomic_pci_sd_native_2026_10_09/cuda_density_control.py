"""Private complete weighted cell-Coulomb batch, explicit float64 CUDA only.

The physical geometry and rank weights come from the shared CPU feature
service. This is an arithmetic/control prototype, not a native SD backend.
"""

import numpy as np

KERNEL = r"""
extern "C" __global__ void moments(
  const double *x, const long long *outer, const double *rise,
  const double *fall, double *up, double *down,
  int nv, int ng, int batch) {
  long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=(long long)nv*batch) return;
  int cell=tid/batch, b=tid%batch;
  double u=0.0,d=0.0;
  // Match the shared CPU feature service's component-before-node order.
  for(int c=0;c<2;++c) for(int m=0;m<ng;++m) {
    long long i=outer[((long long)c*nv+cell)*ng+m];
    double value=x[i*batch+b];
    u=u+rise[cell*ng+m]*value;
    d=d+fall[cell*ng+m]*value;
  }
  up[tid]=u; down[tid]=d;
}
extern "C" __global__ void scans(
  const double *up,const double *down,double *prefix,double *suffix,
  int nv,int batch) {
  int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b>=batch) return;
  double u=0.0,d=0.0;
  for(int cell=0;cell<nv;++cell) {
    long long i=(long long)cell*batch+b;
    prefix[i]=u; u=u+up[i];
  }
  for(int cell=nv-1;cell>=0;--cell) {
    long long i=(long long)cell*batch+b;
    suffix[i]=d; d=d+down[i];
  }
}
extern "C" __global__ void scatter(
  const double *x,const long long *cell_map,const long long *node_map,
  const double *rise,const double *fall,const double *prefix,
  const double *suffix,const long long *rowptr,const long long *partner,
  const long long *edge,const double *weights,double *y,
  int nodes,int ng,int batch) {
  long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=(long long)nodes*batch) return;
  int i=tid/batch,b=tid%batch;
  long long cell=cell_map[i],m=node_map[i];
  double value=0.0;
  if(cell>=0) {
    value=value+fall[cell*ng+m]*prefix[cell*batch+b];
    value=value+rise[cell*ng+m]*suffix[cell*batch+b];
  }
  // Incident edges retain original global edge ordering, both diagonal
  // orientations included. No atomic races, screening or lost interference.
  for(long long e=rowptr[i];e<rowptr[i+1];++e)
    value=value+weights[edge[e]]*x[partner[e]*batch+b];
  y[tid]=value;
}
"""


class CUDADensityControl:
    """Resident shared geometry; cap checked before each complete device batch."""

    def __init__(self, feature, *, memory_bytes):
        import cupy as cp

        if (
            isinstance(memory_bytes, bool)
            or not isinstance(memory_bytes, int)
            or memory_bytes < 1
        ):
            raise ValueError("positive explicit complete device memory cap required")
        self.cp, self.feature, self.cap = cp, feature, memory_bytes
        self.nodes, self.nv, self.ng = (
            len(feature.radii),
            *feature.outer_indices.shape[1:],
        )
        left, right = feature.edge_left, feature.edge_right
        # Bound both orientations and all temporary host/device coordinates
        # before creating the CSR control; do not drop zero-weight edges.
        self.geometry_bound = 8 * (
            8 * self.nodes
            + 3 * feature.outer_indices.size
            + 12 * len(left)
            + 6 * self.nv * self.ng
        )
        if self.geometry_bound > self.cap:
            raise MemoryError("complete device geometry/CSR control exceeds cap")
        row = np.empty(2 * len(left), dtype=np.int64)
        partner = np.empty_like(row)
        edge = np.repeat(np.arange(len(left), dtype=np.int64), 2)
        row[0::2], row[1::2] = left, right
        partner[0::2], partner[1::2] = right, left
        order = np.argsort(row, kind="stable")
        rowptr = np.concatenate(
            ([0], np.cumsum(np.bincount(row, minlength=self.nodes)))
        )
        cell_map, node_map = (np.full(self.nodes, -1, dtype=np.int64) for _ in range(2))
        for c in range(2):
            for cell in range(self.nv):
                for m in range(self.ng):
                    i = feature.outer_indices[c, cell, m]
                    if cell_map[i] != -1:
                        raise ValueError(
                            "duplicate outer coordinate must be handled explicitly"
                        )
                    cell_map[i], node_map[i] = cell, m
        self.geometry = tuple(
            cp.asarray(a)
            for a in (
                feature.outer_indices,
                cell_map,
                node_map,
                rowptr,
                partner[order],
                edge[order],
            )
        )
        self.kernels = tuple(
            cp.RawKernel(KERNEL, name, options=("--fmad=false",))
            for name in ("moments", "scans", "scatter")
        )
        self.rank = None
        self.weights = None

    def preflight(self, batch):
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("positive complete batch required")
        bound = self.geometry_bound + 8 * batch * (2 * self.nodes + 4 * self.nv)
        free, _ = self.cp.cuda.runtime.memGetInfo()
        if bound > self.cap or bound > free:
            raise MemoryError(
                "complete resident density batch exceeds cap/free device memory"
            )
        return bound

    def apply(self, x, rank):
        cp = self.cp
        if (
            not isinstance(x, cp.ndarray)
            or x.dtype != cp.float64
            or x.ndim != 2
            or x.shape[0] != self.nodes
            or not x.flags.c_contiguous
        ):
            raise ValueError(
                "complete C-contiguous device float64 density batch required"
            )
        batch = x.shape[1]
        self.preflight(batch)
        if self.rank != rank:
            self.weights = tuple(cp.asarray(a) for a in self.feature.rank_arrays(rank))
            self.rank = rank
        rising, falling, weights = self.weights
        outer, cell, node, rowptr, partner, edge = self.geometry
        up, down, prefix, suffix = [
            cp.empty((self.nv, batch), dtype=cp.float64) for _ in range(4)
        ]
        y = cp.empty_like(x)
        integer = tuple(np.int32(n) for n in (self.nv, self.ng, batch))
        self.kernels[0](
            ((self.nv * batch + 127) // 128,),
            (128,),
            (x, outer, rising, falling, up, down, *integer),
        )
        self.kernels[1](
            ((batch + 127) // 128,),
            (128,),
            (up, down, prefix, suffix, np.int32(self.nv), np.int32(batch)),
        )
        self.kernels[2](
            ((self.nodes * batch + 127) // 128,),
            (128,),
            (
                x,
                cell,
                node,
                rising,
                falling,
                prefix,
                suffix,
                rowptr,
                partner,
                edge,
                weights,
                y,
                np.int32(self.nodes),
                np.int32(self.ng),
                np.int32(batch),
            ),
        )
        return y
