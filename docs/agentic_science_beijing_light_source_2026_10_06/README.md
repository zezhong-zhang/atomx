# 北京光源与未来产业参赛材料

完整项目书、评审摘要和核心证据面向 Agentic Science 项目评审与技术答辩。科学数据截至 2026 年 10 月 6 日；已完成有限模型判别，完整相关收敛及物理极点认证仍为 False。

- [完整项目书 Markdown](参赛项目书.md)、[PDF](参赛项目书.pdf)、[Word](参赛项目书.docx)、[本地 HTML](参赛项目书.html)。
- [评审摘要](评审摘要.md)。
- [证据索引](evidence_index.json)、[外部来源记录](sources/index.json)、[实际核验结果](verification.json)。

完整项目书可独立阅读。核心 JSON 与图像是原科学档案的逐字节副本；复制不改变原封存记录。包内验证检查副本完整性、核心数值、状态和本地链接，不重新运行物理计算。

```bash
python verify_evidence.py
```

在完整 Atomx 工作区使用相对仓库链接可以进一步查看[原科学档案](../xfel_tail_breit_pole_2026_10_06/README.md)。独立分发包仅含文档与证据子集，不包含全部矩阵、332 个物理源码或原始会话消息，不能当作完整可执行 ARM 包或真实原始 agent 轨迹。

文档的六周计划、材料应用与智能体对照评测均为拟议实施内容。官网 XAS 题目作为衔接案例；该轮已结束，当前 API 的评分字段与题面描述存在差异。正式参赛应使用目标场次的当前要求。

参赛队名、成员、单位、联系方式、目标场次和数据使用授权由实际负责人填写报名系统；正文没有编造这些信息。本包未进行官网上传、报名或提交。

排版入口为 [build_document.py](build_document.py)。渲染依赖安装在 uv 的临时环境中，不修改 Atomx 的物理运行环境：

```bash
uv run --no-project --with markdown --with weasyprint --with python-docx --with cairosvg python build_document.py
```

来源存档仅供本地溯源；对外分发以材料包中的链接、哈希与短引用为准，平台完整文档不进入分发 ZIP。完整科学证据重放在原环境的可写副本中进行，避免向封存目录写入缓存。
