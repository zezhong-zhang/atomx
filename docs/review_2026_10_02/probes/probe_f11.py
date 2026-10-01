import resource, time, sys
t0 = time.perf_counter()
from atomx.angular.csf_hamiltonian import block_terms, ExactCISpaceError, pattern_m_counts
pattern = ((4,-4,4),(4,3,3),(5,-5,5))
print("import+setup s:", round(time.perf_counter()-t0, 3))
# dimension count only (DP, no enumeration)
t0 = time.perf_counter()
counts = dict(pattern_m_counts(pattern))
print("pattern_m_counts s:", round(time.perf_counter()-t0, 6), "total all-M dets:", sum(counts.values()))
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
try:
    block_terms((pattern,), 8, max_determinants=1)  # 2J=8 sector
    print("NO ERROR RAISED -- unexpected")
except ExactCISpaceError as e:
    dt = time.perf_counter()-t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"raised ExactCISpaceError after {dt*1e3:.2f} ms; delta-peak-RSS {(rss1-rss0)/1024:.2f} MiB")
    print("msg:", str(e)[:160])
except Exception as e:
    print("other exception:", type(e).__name__, str(e)[:200])
