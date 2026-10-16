# Yb 虚层对照：版本5实际发布

[远程周期表](https://atomx-atomic-spectrum-atlas.black-fairy-6027.chatgpt.site/#Yb:1:models)
已发布版本5，16个模型、64行谱对照。新增核场和闭核68屏蔽场两种固定160点
虚层RCI；旧14模型56行及6019份NIST载荷保持原值。

71项独立数据检查和79项实际浏览器检查通过。新增模型仍不是已收敛的29轨道
MCDHF：扩展驻值未计算，原驻值和实验目标未通过。核场虚层的名义gap RMS为
2019.201 cm⁻¹，屏蔽虚层为2048.365 cm⁻¹，共同基线为2022.830 cm⁻¹。
页面列出上下态分别变化，未拼入其他点的Breit、QED或反冲。

仅站点20个文件提交至 `4b1797f5ec78d9c00a862d281adb9a227a25243b`，推送后
核实远端提交，再从同一提交打包。实际部署成功，读回版本5和原访问策略：
owner-private、策略版本1、一个用户、零编辑者和外部访问者。未提交Atomx源码。

证据：[本地冻结交接](handoff.json)、[根发布审查](root_publication_preflight.json)、
[推送与归档](root_push_and_archive.json)、[实际部署与访问读回](root_publication.json)。
原[版本4](../atlas_update003/PUBLICATION.md)保留。此次文档影响仅更新当前站点
说明与主题索引；本地准备README和交接字节保持其发布前范围，科学输入、源码、
数值门和历史失败未修改。发布不增加物理资格或最高水平认证。
