# 第二轮：残差 KAN、末隐藏层 KAN 与等参数量 MLP

范围：steels、phonons、dielectric，沿用第一轮已经复现对齐的官方特征、五个外折、五个内折及各成员种子。第一轮训练文件保持不变；第二轮使用独立目录 `benchmarks/hybrid-nested-20261022`。

## 预先确定的模型

1. **residual_kan**：保留 MODNet 全部四个隐藏块和原线性输出 `W h + b`，增加 `alpha * FastKAN(h)` 修正；可训练的标量 alpha 初始化为零。原线性层和主干使用配对 MODNet 相同的初始权重，全部从头联合训练，不读取第一轮训练权重。
2. **last_hidden_kan**：只替换第四个、也是最后一个隐藏 Dense 块为 FastKAN。前三个隐藏块保持 ELU，最终线性输出不变；未替换层与配对 MODNet 初始化一致。
3. **residual_mlp**：与 residual_kan 对应的控制。保留相同线性读出和零初始化 alpha，修正分支为 `Dense(ELU) -> Dense(linear)`。
4. **last_hidden_mlp**：与 last_hidden_kan 对应的控制。只把第四块换成 `Dense(ELU) -> Dense(linear)`，其他块和最终线性输出不变。
5. MODNet 基线直接复用第一轮同样 50 个成员的模型，重新核对外折预测和分数，没有假装重新训练。

两个 MLP 控制均通过调整实际参与运算的隐藏宽度匹配对应 KAN 的**总参数量**，没有添加无用参数。整数宽度取最近值，单成员总参数量误差要求不超过 1%；全部 150 组原始配置、三个网格数预检的最大误差约为 0.46%。它们分别控制两种新 KAN 的容量，并不是上一轮全 KAN 四倍参数量的宽 MLP。

FastKAN 沿用第一轮实现：Gaussian RBF、网格范围 [-2, 2]，RBF 分支 LayerNorm（epsilon=1e-5），基础分支 Dense(SiLU(x))。本轮不是原始 B-spline KAN。

## 内层选择与对照预算

每个外折保留第一轮 MODNet 按内层成绩选出的十组基础配置。每组基础配置的输入特征、原始隐藏宽度、batch size、输入缩放器及五个内折成员固定。本轮不通过外层成绩重新选择这些基础配置，也不重新搜索特征。

原版 FitGenetic 在创建内折之前会先打乱外层训练集顺序。本轮直接按保存的成员训练/验证 ID 还原内折及顺序，并核对五个验证折恰好覆盖一次外层训练集，避免在原始顺序上重新划折导致不一致。

- 每个 KAN、每组基础配置试六种组合：RBF 网格数 `{3, 5, 8}` × 学习率倍率 `{0.5, 1.0}`。学习率倍率乘以该组 MODNet 原学习率。
- 对每个基础配置，先计算每个候选的五个内层验证 MAE 的算术平均，选择最小者；平分按预先固定的候选编号选择。
- 对应 MLP 的容量固定为该 KAN 在内层选中的容量，然后尝试六个学习率倍率 `{0.25, 0.5, 0.75, 1.0, 1.5, 2.0}`，同样按五个内层验证 MAE 选择。
- 两种 KAN 和两个 MLP 的追加候选预算相同，各为六个候选 × 十组基础配置 × 五个内折 × 十五个外折 = **4,500 次训练**，合计 **18,000 次**。这是相同训练次数预算，不是相同墙钟计算预算。
- 每个最终模型仍使用十组胜出配置 × 五个内折模型，即 50 成员均值；不在整个外层训练集额外 refit。保留原有数据划分和集成规模。
- MLP 的网格数标签仅表示所匹配的 KAN 参数预算，MLP 本身没有 RBF。它们在给定容量下独立选择学习率，不能解释为在所有可能容量中独立搜索的最佳 MLP。

## 训练规则

使用 MODNet 0.1.12、TensorFlow 2.10.1 与第一轮完全相同的隔离环境。最多 800 epochs，原尺度标签、MAE、TensorFlow Adam。EarlyStopping 沿用官方版本：监视训练 loss，min_delta=0.001，patience=30，不恢复最佳权重。候选排名使用训练结束后的内层验证 MAE。

初始化前及 fit 前重置为对应的第一轮成员种子，让模型间的样本洗牌种子配对。原始 predict 行为（包括历史越界预测替换）保持一致，同时保存原始与后处理成员预测。

## 防止测试集参与选择

训练工作进程只接收保存的外层训练集及内层索引，不接收外层测试数据或测试标签。新候选选择函数只读取内层验证 MAE。

**所有三个任务、十五个外折的候选选择全部完成并写入带 SHA-256 的冻结清单之后，流程才开始任何新的外层测试预测。** 恢复运行会检查源代码、依赖、数据、基础配置及冻结选择；发生不兼容变化则拒绝混用旧缓存。

描述符排序按用户要求沿用第一轮：在外层训练集上排序，内折复用。它不访问外层测试标签，但不是每个内折重新筛选特征的严格嵌套特征选择。也要区分：这些后续假设提出前已经看过第一轮外层成绩，因此这是同一 benchmark 上的探索性后续实验，不能声称是从未接触的独立确认集。

## 核验与交付

小规模 smoke 独立存放，最多一个 epoch，绝不作为正式性能成绩。模型测试验证零门控的初始预测、梯度、未替换层权重、实际参数量、保存恢复和内层均值选择规则。

正式完成后自动核验全部候选的特征、样本、种子、内层选择、等参数量条件，以及最终 75 行（十五外折 × 五种模型）的成员预测和 MAE，输出：

- `hybrid-status.json`：当前任务、外折、模型、完成成员数；失败时保留错误。
- `hybrid-protocol.json` 和 `source-snapshot/`：协议及运行源码。
- 每折 `inputs.json`、各模型 `*-inner-search.json`、`selection.json`、`comparison.json`。
- 根目录 `selections-locked.json`：先于新外层预测生成的完整选择清单。
- `hybrid-fold-results.csv`、`hybrid-summary.csv`、`hybrid-verification.json`、`hybrid-report.md`。

报告分别记录选中集成成员的累计训练时间和全部候选的累计搜索时间，后者已经包括前者；不重复相加，也不把并行累计时间当成实际等待时间。

## 入口

```powershell
& 'C:\Users\User\.conda\envs\modnet-v012-matbench\python.exe' -u scripts/run_modnet_hybrid_nested.py `
  --baseline-dir benchmarks/official-aligned-20261021 `
  --output-dir benchmarks/hybrid-nested-20261022 `
  --n-jobs 6
```

相同命令会恢复兼容的已完成成员；正式运行强制涵盖全部三个任务和五个外折。

2026-10-22 的后台启动维护：Windows 曾发生控制台挂起，并在同一时段重启命令执行服务。启动脚本现通过 Windows 进程管理启动一个短 launcher，再以 `CREATE_NO_WINDOW` 和独立日志文件句柄运行训练，避免依赖调用命令会话的控制台及输出管道。没有修改训练代码、候选预算或冻结协议；操作系统重启仍会中断进程，之后可按原检查点恢复。启动实现依据 [Windows 进程管理说明](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects) 和 [Python 子进程接口](https://docs.python.org/3/library/subprocess.html)。
