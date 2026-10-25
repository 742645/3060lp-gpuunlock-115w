# 原厂/解锁配对测试完成状态

2026-10-25：用户确认保持 NVIDIA **580.178.04**，回退含义为恢复未解锁状态。用户重启宿主机后，已核实恢复半速原厂状态，随后完成两组各 15 项的配对测量。完整对照表见 [配对基准报告](PAIRED_BENCHMARK_REPORT.md)。

## 已完成

- 固定 CUDA 12.0、内核 `6.8.0-147-generic`，准备同一份二进制与库 SHA-256 清单。
- 同一矩阵尺寸 `2048/4096/8192`、同一数学输入、种子、预热 10 次、每组 20 次、5 组中位数；两组保持 80 W 和自动时钟策略。
- 两组分别通过 15 个带种子的 CPU 参考检查，全部大型 GEMM 输出检查均为 0 错误。
- 通过独立、未计时的 CUPTI 跟踪和对应 SASS 核对，两组各 15 种模式/尺寸组合均符合下表。
- 测试原厂组前先关闭自动解锁；确认普通启动且 `fmla32=1` 后才采集。随后单次启动成功执行新解锁事务，确认 `fmla32=0` 后采集第二组。
- 配对验证通过后已恢复默认解锁启动项，当前显卡处于解锁全速状态，保留普通 Ubuntu 入口。

| 路径 | 乘法/累加与输出 | 实际确认指令 |
|---|---|---|
| 普通 FP32，无 Tensor | FP32 / FP32 | FFMA，无 HMMA |
| 普通 FP16，无 Tensor | 严格 FP16 / FP16 | HFMA2，无 HMMA |
| Tensor“FP32” | FP32 存储，TF32 乘法，FP32 累加/输出 | HMMA，TF32 |
| Tensor FP16，FP16 累加 | FP16 输入/累加/输出 | HMMA，F16 |
| Tensor FP16，FP32 累加 | FP16 输入，FP32 累加/输出 | HMMA，F32 |

GA106M 的 Tensor FP32 接口并非完整尾数的 IEEE FP32 乘法；TF32 的范围与 FP32 相同，乘法精度较低。[NVIDIA cuBLAS 定义](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t)

大型矩阵使用同一可精确表示的 dyadic 数据以核对所有输出；输入有零填充，调用普通稠密 GEMM，FLOPs 仍按完整矩阵尺寸计算。该输入不能代表一般数据的舍入误差。小型 CPU 参考检查采用不同种子的稠密正负值。

## 已完成的硬件步骤

用户已自行重启宿主机，无需再提供宿主机 SSH。虚拟机 `192.0.2.10` 普通启动后的独立读取确认 `fmla32=1`、`imla4=1`，证明这次基线已恢复未解锁速率。

原厂组采集完成后，通过一次性解锁入口重启虚拟机。当前启动日志包含新的 `pci 0000:00:10.0: gpu_unlock: unlocked`，独立读取变为 `fmla32=0`、`imla4=0`。两组驱动、GPU、内核、二进制、库和 80 W 功率上限均匹配。

已完成顺序：恢复原厂并核验 → 测量原厂 15 项 → 启用单次解锁启动 → 核验事务与全速 → 同一二进制测量解锁 15 项 → 验证元数据、数值、实际指令及功率上限 → 生成对照表。

## 复现脚本与证据

- `benchmarks/paired_gemm.cu`：五条路径，性能与 CUPTI 跟踪分离。
- `scripts/prepare-paired-benchmark.sh`：编译、15 个数值检查、实际指令检查并冻结 SHA-256。
- `scripts/stage-stock-baseline.sh`：关闭默认自动解锁，备份原配置，不进行显卡复位。
- `scripts/collect-paired-benchmark.sh stock|unlocked`：正确状态检查、实际指令与性能采集。
- `tools/paired_results.py`：只有两组均有效且 GPU/驱动/内核/二进制/库/功率匹配时才生成对比。
- `results/2026-10-25/paired-five-paths-preparation/`：准备阶段的原始数值、内核跟踪与反汇编证据。

远端冻结目录：`/home/gpuuser/gpu-unlock-ada/build/paired-five-paths`，其中 `stock/` 和 `unlocked/` 均已完成采集。全部结果保存在本地 `results/2026-10-25/paired-five-paths-results/`。
