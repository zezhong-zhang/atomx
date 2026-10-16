当前报告把原生谱表从历史531-CSF/24轨道/320步指针，修正为29轨道/1046完整目标CSF的共同208基线，以及从该点各推进32步的两路末态。两路都使用原f043源码、3451坐标度量、同五UID等权和DC算符；两个“240”是不同分支，未按实验gap挑选唯一末态。历史[旧主表](../reporting_stage006/current_card_evidence.json)原样保留。

[两路结果](../correlated29_history_comparison_preparation/RESULTS.md)已闭合全部接受点。history10/40分别47/43次H-CI和33次完整梯度，各2219项控制通过；前0–11点的保存数值身份相同，第12点分叉。长历史的最终EOL较低、回溯较少，但残差较大；原驻值与实验门均FAIL。没有改变默认、度量或停止阈值。此次并行耗时不认证并行效率或速度比。

[当前卡片证据](current_card_evidence.json)明确把驻值量和第一组四线绑定共同208基线，另表并列两路末态，NIST不确定度未知值继续为null。[报告](../results_stage024.json)同时记录已采用的[两处共享数值改进及123回归](../correlated29_radial_reuse_qualification003_preparation/shared_implementation/README.md)和[第208步物理审查](../physical_next_step208/README.md)。完整5f双jj固定点RCI仍是前瞻筛选，未据未驻值谱认定主因。

[integration_proof.json](integration_proof.json)核对167项保存元数据与全部33点/分支有限H接受门绑定，零新科学/API/卡片发送。[render_verification.json](render_verification.json)通过现有renderer生成含两个末态的预览，旧531/320未再以当前主表显示。stage024实际卡片尚未发送；[stage023真实卡片](../delivery_stage023/delivery_proof.json)已读回position1617，随后两路原生计算前台闭合。服务另做了四个小型参考/性质/SD/GPU探针，未将它们计为Yb优化。

首次集成在写出任何结果或改registry之前发生标准库Path类型错误，记录于[integration_attempt001_failure.json](integration_attempt001_failure.json)；修正后重新绑定驱动/合同完成集成。没有重做科学计算、重发卡片或改写旧证据。网页v7仍为共同208及19模型76行，两分支网页更新另行准备与发布。
