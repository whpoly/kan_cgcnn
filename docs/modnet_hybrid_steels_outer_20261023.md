# steels 提前外层测试：2026-10-23 协议补充

用户明确要求“那跑一下外层测试”，因此在 steels 全部五折内层选型完成后，提前评估该数据集，不再等待 phonons、dielectric 的内层搜索全部完成。本补充覆盖原协议中“全部三个任务完成选择后才开始任何外层测试”的执行时序要求。

模型、候选网格、学习率、输入特征、内外层划分、随机种子、50 成员集成、预测后处理和内层选择准则均保持原样。其他两个数据集继续预先固定的搜索，不以此次测试成绩改变第二轮方案。

独立入口为 `scripts/evaluate_modnet_hybrid_dataset.py`。它先验证原训练源码、环境和 steels 输入哈希，将五折的 `selection.json` 与 `inputs.json` 原样复制到 `benchmarks/hybrid-steels-outer-20261023`，并在读取测试数据前冻结五折选择及全部所选模型、成员记录的 SHA-256。

测试调用原来的 `run_modnet_hybrid_nested.evaluate_fold`，直接加载已有模型，不额外训练、选择或加权。原 MODNet 也重新预测，并与第一轮逐样本结果核对。

该独立目录使用原核验器检查全部 6,000 个 steels 候选记录、25 行外层结果、测试样本覆盖、配对特征/划分/种子、等参数量约束和成员预测重算。目录中的 `hybrid-protocol.json` 是单数据集核验视图；原始三数据集协议另存为 `source-hybrid-protocol.json`。冻结清单的作用域仅限 steels 的五折，不能称为三个数据集均已完成。

结果入口为 `dataset-outer-report.md` 和 `dataset-outer-summary.json`。`complete_three_task_benchmark` 保持 false，直至原来的三数据集流程完成。原三数据集训练目录中的选型文件、模型文件、运行协议及指纹依赖源码均不修改。

所选成员累计训练时间来自已有训练记录，不是本次预测耗时，也不是并行墙钟时间。搜索时间已经包含所选成员，不应相加。本轮依然属于看过第一轮 benchmark 后提出的探索性后续实验。
