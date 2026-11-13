# 在 MatterVial 增强 MODNet 上检验 KAN：可行性与对照设计

> 2026-10-24 更新：本文件保留最初可行性检查时的状态。数据现已下载、逐样本核验，独立 MODNet 0.1.13 环境及严格内层特征筛选流程已建立。最新审计、执行协议和限制见 [MatterVial phonons 审计记录](mattervial_phonons_audit_20261024.md)。下文“未下载”“尚未训练”的描述是当时状态，不代表最新进度。

## 已核验的资料

论文：[Combining feature-based approaches with graph neural networks and symbolic regression for synergistic performance and interpretability](https://www.nature.com/articles/s41524-025-01938-2)，npj Computational Materials 12, 67（2026），DOI 10.1038/s41524-025-01938-2。

它主要增强输入表示：将 MatMiner 与预训练 GNN 潜变量、近似描述符和 SISSO 公式特征结合，后续仍使用 MODNet。因此它能作为更强的输入基线，但论文的收益不能归因于 KAN。

公开的 [phonons 结果文件](https://github.com/rogeriog/MatterVial_SupportData/blob/main/modnet_mattervial_matbench/matbench_phonons/results/matbench_phonons_matminer_megnet_ofm_mvl_all_roost_mpgap_sisso_orb_v3_average_target_metrics.json) 记录 MM+MV 的平均 MAE **30.08006079904005 cm⁻¹**。将 MM 换成其学习表示的 MV 版本为 **30.587895808389582 cm⁻¹**。这些是作者的结果，不是我们的复现结果。

论文方法为 XGBoost 递归消除先保留 800 个特征，随后使用 MODNet 自身筛选与 GA。[公开 phonons 日志](https://github.com/rogeriog/MatterVial_SupportData/blob/main/modnet_mattervial_matbench/matbench_phonons/9379063_0_phonons.out) 记录 MODData 版本 0.1.13，GA 种群 50、最多 20 代、早停 8、refit=False。此前我们使用 MODNet 0.1.12、种群 20、最多 10 代、早停 4。直接横比时，输入与搜索预算同时变化。

代码：[MatterVial](https://github.com/rogeriog/MatterVial)。数据与逐折结果：[MatterVial_SupportData](https://github.com/rogeriog/MatterVial_SupportData)。作者提供的预计算数据归档 `featurized_datasets.tar.gz`，文件 ID 57963316，公开下载端点可访问，HTTP range 元数据给出的大小为 **10,724,277,360 bytes**。本次只探测文件头，未下载完整归档、未加载其 pickle、未安装 MatterVial 或修改已有 MODNet 环境。

## 可以回答的研究问题

1. **额外准确度**：在相同 MM+MV 输入和计算预算下，KAN 是否优于 MODNet 和对应容量的 MLP？
2. **紧凑性**：在相同精度附近，分组稀疏 KAN 能否减少使用的增强特征、参数量或预测时间？
3. **物理表示**：当前 16 个物理描述符生成的 4 个特征，在强 GNN 特征已经存在时是否仍提供互补信息？

第 3 项可以直接延续已启动的 16→4 内层试验。把固定的 MatterVial 特征作为公共输入追加到所有对照，仍保持分组 MLP、样条、稀疏 KAN 相同参数量。这样测得的是物理模块在更强输入下的边际贡献。

## 对照矩阵

| 输入 | MODNet | 等参数分组 MLP | 分组样条 | 分组稀疏 KAN |
| --- | --- | --- | --- | --- |
| MM 与固定物理输入 | 基础/同物理输入对照 | 当前试验 | 当前试验 | 当前试验 |
| 相同 MM+MV 与固定物理输入 | 增强基线 | 增强容量对照 | 增强函数类对照 | 增强 KAN |

最关键的差值是同一行内 KAN 相对增强 MODNet/MLP 的变化；不能用“增强 KAN 对旧 MODNet”同时混入特征、预训练数据与网络结构三种差异。我们的旧 phonons MODNet 34.0134、全 KAN 33.0218 只能作为旧输入协议的结果。

## 接入前必须核验的科学细节

- 预计算数据的样本 ID、目标列、Matbench 外层索引和特征组/版本是否对应作者的 30.0801 配置。
- XGBoost 预筛选、NMI 排序、特征组选择及任何 SHAP 重排的拟合样本范围。对我们的公平比较，监督步骤在每个内层训练部分拟合，超参数只通过内层验证选择，最后才评估外层测试。
- [公开 SISSO 公式库](https://github.com/rogeriog/MatterVial/blob/master/mattervial/featurizers/formulas/SISSO_FORMULAS_v1.txt) 含标注为 `matbench_phonons`、`matbench_steels`、`matbench_dielectric` 的来源分组，调用函数加载固定公式。文件本身不足以确认推导公式时使用了哪些样本、如何排除目标外层折。**这是一项尚未完成的来源审计，不能据此断言论文泄漏，也不能默认可以直接复用所有公式。**
- 如不能确认目标相关公式来源，应增加无 SISSO 的公平基线，或在各内层训练部分重新产生公式。此时应明确称为协议修正后的比较，不能强称精确复现论文。
- 两种预测器共享同一份冻结 GNN 表示及其预训练数据来源。不要只给 KAN 加外部表示。训练时间应同时报告特征提取成本、下游训练和搜索成本；预计算特征的提取耗时不能算作零。
- 集成成员数、参数预算、候选搜索次数及种子配对。先对齐增强 MODNet，随后固定共同预算比较；不要一边用 50×20 GA，另一边只跑一个手选配置。

## 当前执行边界

2026-10-23 已完成代码/公开结果/日志的可行性检查，尚未复现 MatterVial 的 30.0801，也未训练使用 MatterVial 的 KAN。当前正在执行的是原先授权的 phonons 16→4 小规模试验：100 次内层训练，五种变体、两个学习率、两个种子、五个内层折。协议见 `docs/phonons_physical_pilot_20261023.md`。

下一步按顺序推进：完成当前试验并检查有效性 → 审计和准备 phonons 增强特征 → 复现增强 MODNet → 在相同输入与预算下跑对照。若 KAN 无稳定 MAE 收益，仍可检验压缩和跨种子结构稳定性，但不能事后把任何一项变化都包装成优势。
