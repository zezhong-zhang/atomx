# 实际CPU/CUDA固定工作对照

原完整DC/static/dynamic PP/HH与全部density四rank逐位相等，原4GiB地址及32MiB显存payload门通过。五轮交错配对实际观察：CPU warm complete apply 7.783 µs，GPU含全振幅传输及分离输出下载 133.875 µs；resident host 34.684 µs，event 34.512 µs。后两者包含20次kernel及其host-launch间隔。

GPU有其他compute PIDs，原负载门失败，speedup/efficiency保持null。不是大规模完整SD的性能或GPU硬件效率结论；本例10×12太小，不能推广。原source setup与cold/JIT/geometry/transfer独立记录于[完整实跑结果](actual001/result.json)和[汇总](summary.json)，全部输入输出及PTX保留。峰RSS 506.211 MiB，owned device payload 1,361,840 B。
