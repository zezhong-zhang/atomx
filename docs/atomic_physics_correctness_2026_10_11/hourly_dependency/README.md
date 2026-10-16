# 当前 norm producer 的显式 NIST 消费

小时卡片触发的 NIST 重放在比较开始前正确拒绝了旧 norm 证明：其绑定的 CoreSD 源 SHA 已不同于当前源码。失败记录保留；不能忽略 SHA 差异或以 AST／仅文档等价推断替代当前实算证明。

当前已实算的完整 norm producer 是 `docs/atomic_pci_sd_native_2026_10_09/hourly_iterations/parallel_norm_e6ad12dd6ef19ccd8b06/probe/`。它保留全部 28,218,500 核幅度、五态与 18 M，实际当前源码与同 H／core／space 证明通过。

[NIST probe](../../atomic_pci_sd_native_2026_10_09/parallel_six_nist001/probe.py)新增 `--norm-run`：可显式选择 repo-relative 或 absolute 的仓库内实际 producer 目录；historical `parallel_six_norm001/run002` 仍是 CLI default。选择只改变依赖路径，完整 source SHA、input SHA、finite anchor、五根、norm、实验匹配及原门保持。缺失目录、文件和仓库外目录直接拒绝。

[并行 registry](../../atomx_hourly_progress_2026_10_10/parallel_registry.json)仅改 NIST 的 `arguments` 和八个 norm `input_files`，精确指向新 producer。其他 task、全局资源上限和所有 scientific gates 未改。

本次 bounded 实际入口执行：

```sh
.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_six_nist001/probe.py \
  --output docs/atomic_physics_correctness_2026_10_11/hourly_dependency/actual_next \
  --norm-run docs/atomic_pci_sd_native_2026_10_09/hourly_iterations/parallel_norm_e6ad12dd6ef19ccd8b06/probe
```

输出目录必须不存在。实际 [actual001/result.json](actual001/result.json)通过全部九项原审计门，**excitation、splitting、observed-wavelength 三个精度门仍为 false**；没有新原子根或精度提升。过程为 1.18 s、RSS 382,615,552 bytes，原 4 GiB／120 s／单线程上限未改；不是性能排名。

[handoff](handoff.json)绑定新 norm result、实际 NIST result/audit、源码、registry 及完整 source/input postcheck，并保留旧失败 job SHA。NIST 的实际 30 个 source 和 36 个 input 与 registry 的集合完全一致。

验证包括当前 norm 的 21 个 source hashes、实际 NIST 的 30 个 source／36 个 input hashes、完整五态身份与原失败百分比重放、producer API 的三类无效路径拒绝、registry 仅两字段改动检查、ruff 及 diff/空白检查。证据：

- [前瞻合同](contract.json)、[运行前资源检查](preflight.json)
- [before 快照](before/)、[最小 patch](minimal.patch)
- [API 与 registry 检查](api_and_registry_checks.json)
- [实际审计](actual001/audit.json)、[实际 source/input postcheck](actual001/source_and_input_postcheck.json)
- [文档影响](documentation_impact.json)

父 owner 更新当前 planner／文档并发送 v2 卡片触发注册路径；本子任务没有改那些当前指南或发送卡片。
