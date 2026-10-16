"""Exact integrated flow on a supplied finite acyclic positive-rate graph."""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _flow(order, indptr, destination, rates, initial):
    visits = initial.copy()
    residence = np.zeros(len(initial))
    terminal = np.zeros(len(initial))
    for position in range(len(order)-1, -1, -1):
        state = order[position]
        total = 0.
        for edge in range(indptr[state], indptr[state+1]):
            total += rates[edge]
        if total == 0:
            terminal[state] = visits[state]
        else:
            time = visits[state]/total
            residence[state] = time
            for edge in range(indptr[state], indptr[state+1]):
                visits[destination[edge]] += time*rates[edge]
    return visits, residence, terminal


def _prepare_graph(order, source, destination, rates, initial, max_edges, memory_bytes):
    for cap in (max_edges, memory_bytes):
        if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)) or cap < 1:
            raise ValueError('positive integer graph resource caps required')
    order, source, destination = map(np.asarray, (order, source, destination))
    rates, initial = np.asarray(rates, dtype=float), np.asarray(initial, dtype=float)
    if source.size == 0:
        source = source.astype(np.int64)
    if destination.size == 0:
        destination = destination.astype(np.int64)
    n = initial.size
    if (initial.ndim != 1 or n < 1 or np.any(~np.isfinite(initial)) or np.any(initial < 0)
            or order.shape != (n,) or order.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(order), np.arange(n))
            or source.ndim != 1 or source.shape != destination.shape or source.shape != rates.shape
            or source.dtype.kind not in 'iu' or destination.dtype.kind not in 'iu'
            or np.any(source < 0) or np.any(destination < 0) or np.any(source >= n) or np.any(destination >= n)
            or np.any(~np.isfinite(rates)) or np.any(rates < 0)):
        raise ValueError('finite nonnegative graph weights, in-range edges and complete order required')
    with np.errstate(over='ignore'):
        initial_total = float(initial.sum())
    if not np.isfinite(initial_total):
        raise ValueError('finite total initial weight required')
    if len(source) > max_edges or 80*len(source)+128*n > memory_bytes:
        raise MemoryError('complete acyclic graph exceeds numerical payload/edge cap')
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    if np.any(rank[destination] >= rank[source]):
        raise ValueError('every graph edge must strictly decrease the declared order')
    permutation = np.argsort(source, kind='stable')
    indptr = np.r_[0, np.cumsum(np.bincount(source.astype(np.int64), minlength=n))].astype(np.int64)
    totals = np.bincount(source.astype(np.int64), weights=rates, minlength=n)
    if np.any(~np.isfinite(totals)):
        raise ArithmeticError('outgoing rate sum overflow')
    return (np.asarray(order, dtype=np.int64), source, destination, rates,
            initial, initial_total, permutation, indptr, totals)


def integrated_acyclic_rates(order, source, destination, rates, initial, *,
                             max_edges=5000000, memory_bytes=512*1024**2):
    """Infinite-time visits/residences/counts with no normalization repair.

    ``order`` is a complete permutation with every destination preceding its
    source. Repeated positive edges remain separate physical contributions.
    Initial weights may be unnormalized and carry arbitrary consistent units.
    A terminal node has zero supplied outgoing rate; omitted processes are
    caller-owned physics, never inferred here. Upward/cyclic kinetics reject.
    """
    (order, source, destination, rates, initial, initial_total,
     permutation, indptr, totals) = _prepare_graph(
        order, source, destination, rates, initial, max_edges, memory_bytes)
    visits, residence, terminal = _flow(np.asarray(order, dtype=np.int64), indptr,
                 np.asarray(destination[permutation], dtype=np.int64), rates[permutation], initial)
    counts = residence[source]*rates
    defect = float(terminal.sum()-initial_total)
    if (not all(np.isfinite(x).all() for x in (visits, residence, terminal, counts))
            or not np.isfinite(defect) or abs(defect) > 1e-10*max(initial_total, 1e-300)):
        raise ArithmeticError('acyclic population flow failed finite/particle-conservation gate')
    return dict(visits=visits, residence_times=residence, terminal_weights=terminal,
                transition_counts=counts, outgoing_rates=totals, particle_defect=defect)


@nb.njit(cache=True)
def _finite_flow(order, indptr, destination, rates, totals, initial, time_edges):
    population = initial.copy()
    residence = np.zeros(len(initial))
    for step in range(len(time_edges)-1):
        dt = time_edges[step+1]-time_edges[step]
        incoming = population.copy()
        for position in range(len(order)-1, -1, -1):
            state = order[position]
            rate = totals[state]
            if rate == 0:
                factor = dt
                survival = 1.
            elif dt <= 1./rate:
                factor = dt/(1.+dt*rate)
                survival = 1./(1.+dt*rate)
            else:
                # The reciprocal form also covers finite rates with dt*rate
                # beyond the floating range; no fast channel is discarded.
                reciprocal = 1./rate
                factor = reciprocal/(1.+reciprocal/dt)
                survival = (reciprocal/dt)/(1.+reciprocal/dt)
            integral = factor*incoming[state]
            population[state] = survival*incoming[state]
            residence[state] += integral
            for edge in range(indptr[state], indptr[state+1]):
                incoming[destination[edge]] += integral*rates[edge]
    return population, residence


def finite_acyclic_rates(order, source, destination, rates, initial, time_edges, *,
                         max_edges=5000000, max_steps=4096,
                         memory_bytes=512*1024**2):
    """Positive finite-time backward-Euler populations and every edge count.

    Time/rate units must be reciprocal. The caller supplies a complete grid
    starting at zero and independently refines it for the observable of
    interest. Implicit stability and conservation provide no temporal error
    bound. There is no clipping, renormalization, constant source, upward
    kinetics or elimination of fast states. Physical graph coverage remains
    caller-owned, as for :func:`integrated_acyclic_rates`.
    """
    for cap in (max_steps, max_edges, memory_bytes):
        if (isinstance(cap, (bool, np.bool_))
                or not isinstance(cap, (int, np.integer)) or cap < 1):
            raise ValueError('positive integer time/graph resource caps required')
    if not np.isrealobj(time_edges):
        raise ValueError('real time edges required')
    edges = np.asarray(time_edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges))
            or edges[0] != 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError('finite increasing time edges starting at zero required')
    if len(edges)-1 > max_steps or edges.nbytes > memory_bytes:
        raise MemoryError('complete temporal grid exceeds numerical payload/step cap')
    (order, source, destination, rates, initial, initial_total,
     permutation, indptr, totals) = _prepare_graph(
        order, source, destination, rates, initial, max_edges, memory_bytes)
    if 80*len(source)+128*len(initial)+edges.nbytes > memory_bytes:
        raise MemoryError('complete graph and temporal grid exceed numerical payload cap')
    population, residence = _finite_flow(order, indptr,
        np.asarray(destination[permutation], dtype=np.int64), rates[permutation],
        totals, initial, edges)
    counts = residence[source]*rates
    defect = float(population.sum()-initial_total)
    if (not all(np.isfinite(x).all() for x in (population, residence, counts))
            or np.any(population < 0) or np.any(residence < 0) or np.any(counts < 0)
            or not np.isfinite(defect) or abs(defect) > 1e-10*max(initial_total, 1e-300)):
        raise ArithmeticError('finite acyclic flow failed finite/positive/particle gate')
    return dict(populations=population, residence_times=residence,
                transition_counts=counts, outgoing_rates=totals, particle_defect=defect,
                duration=float(edges[-1]), steps=len(edges)-1,
                method='positive backward Euler; independent time-grid refinement required')
