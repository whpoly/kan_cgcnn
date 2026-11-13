# 官方 MODNet 与 KAN 对齐实验

本轮任务：steels、phonons、dielectric；先重新训练官方 MODNet，再比较只换输出头与全 KAN。此前 `controlled-audit-20261020` 是简化对照，不能充当这里的官方基线。

## 公开参考与可复现性边界

已下载 Matbench 的 MODNet v0.1.12 预测文件，并通过独立 Matbench API 核对 15 个外折的样本编号、真实标签和 MAE：

| 数据集 | 公开五折平均 MAE |
| --- | ---: |
| steels | 87.7627474983 |
| phonons | 34.2751061541 |
| dielectric | 0.2711019242 |

这一步是对公开预测重新计分，不代表本机已经重新训练出这些成绩。

v0.4.0 发布说明明确说全部任务使用 GA，但同一归档的运行脚本仍是旧的网格搜索。Matbench 的示例脚本也声明仅供示意，且其布尔型 `refit=True` 与库中整型集成数量的接口不完全一致。公开记录没有提供当年的训练种子、各折 GA 参数或训练权重，因此不能承诺逐位复现。当前采用 **MODNet 0.1.12 自带 `matbench_benchmark` 的 GA 默认配置**，并明确记录这一选择。

- [MODNet benchmark v0.4.0 发布说明](https://github.com/modl-uclouvain/modnet-matbench/releases/tag/v0.4.0)
- [Matbench v0.1.12 原始条目和预测](https://github.com/materialsproject/matbench/tree/main/benchmarks/matbench_v0.1_modnet_v0.1.12)

## 固定协议

- 官方五个外折：KFold，shuffle=True，random_state=18012019。
- steels：CompositionOnlyFeaturizer；结构任务：DeBreuck2020Featurizer(fast_oxid=True)。使用原始 MODNet/matminer 特征，不使用 structure-lite。
- 外折训练集调用原版 relevance–redundancy 特征排序，`n=-1`，使用官方预计算 cross-NMI。内折沿用外折的特征排序，与该版原始实现一致；这不是每个内折重新筛选特征的严格嵌套协议。
- GA：每代 20 个候选，最多 10 代，五折内层验证，连续 4 代最优模型不变则结束；`refit=False`。
- 每个候选直接调用官方 `Individual.evaluate`：最多 800 epochs，原始尺度标签，MAE，TensorFlow Adam；学习率、batch size、特征数、缩放器和宽度来自 GA。
- EarlyStopping：monitor=loss，min_delta=0.001，patience=30，restore_best_weights=False。没有自行改成验证早停或目标标准化。
- 最终选验证 MAE 最好的 10 组配置，每组保留五个内折模型，共 50 个成员求均值；该 GA 分支不额外 bootstrap，也不在全外折训练集上重新拟合。
- 重置每个成员的初始化种子和训练前种子，使输出头替换不会改变 minibatch shuffle 的种子。随机种子是本轮实验新声明的，不是公开发表时的种子。
- 原版训练、评估、预测预处理保持一致，包括预测阶段对超出训练标签范围一定幅度的值做随机替换这一历史行为。

## 三种模型

每组 GA 配置的宽度可以不同，但配对的三个模型使用相同宽度、特征顺序、训练/验证样本、学习率、batch size、缩放器、种子和集成成员。

1. MODNet：前三个 Dense 块使用 ELU，第四个隐藏块为线性 Dense，最后为线性 Dense 输出。
2. KAN 输出头：仅替换最后的 Dense；四个隐藏块保持官方结构，并与配对 MODNet 使用相同初始权重。所有层均重新训练，隐藏块不冻结。
3. 全 KAN：全部五个 Dense 替换成 FastKAN 层；不额外叠加 ELU。

本轮 KAN 明确定义为 **FastKAN / Gaussian RBF**：每个输入三个基函数，中心 -2、0、2，宽度 2；RBF 分支使用 LayerNorm，epsilon=1e-5；另加 `Dense(SiLU(x))` 基础分支。RBF 权重 Normal(0, 0.1)，基础分支 GlorotUniform。这是 TensorFlow 版本，不是原始 B-spline KAN。全 KAN 的参数量更大，会如实报告；本实验是相同宽度的替换消融，不是参数量匹配实验。

KAN 不另外搜索超参数，而是冻结 MODNet 选出的 50 个成员配置。这回答“保持 MODNet 设置，仅更换网络模块会怎样”，不能据此断言 KAN 在独立调参后的最佳成绩。

## 执行顺序与核验

先完成全部 15 个 MODNet 外折。事先将五折平均 MAE 相对公开值的绝对偏差 10% 定为排查触发线；任一任务超出时，流程记录 `baseline_alignment_requires_investigation` 并停在基线阶段，先排查，不靠测试集反复调参追分。这条线只是诊断门槛，不构成统计等价或精确复现的证明。通过后再运行两种 KAN。

每个训练成员单独保存：配置、输入特征、训练和验证 ID、种子、训练曲线、参数数、训练时长、模型权重。每个外折保存 50 个成员的预测和集成预测。缓存包含源码、输入数据和配置指纹，不能把不同协议的缓存混用。

`verify_modnet_matched_run.py` 核对每个配对成员的特征、样本、种子、宽度和超参数，并从成员预测重新计算集成 MAE。只有三个任务的 45 行正式外折结果都齐全，才生成最终汇总；smoke 文件不会进入正式结果。

## 本地入口

官方环境：`C:\Users\User\.conda\envs\modnet-v012-matbench\python.exe`。

```powershell
& 'C:\Users\User\.conda\envs\modnet-v012-matbench\python.exe' -u scripts/run_modnet_official_matched.py `
  --output-dir benchmarks/official-aligned-20261021 `
  --cache-dir benchmarks/official-aligned-20261021/precomputed `
  --reference benchmarks/official-aligned-20261021/reference/results.json.gz `
  --n-jobs 4
```

相同命令可以恢复未完成的运行。`matched-status.json` 是本轮训练状态；旧准备脚本的 `STATUS.json` 只表示特征准备状态。正式训练结果不能从 smoke 目录推断。
