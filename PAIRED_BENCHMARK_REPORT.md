# 原厂与解锁配对基准报告

2026-10-25，用户重启宿主机后，已确认显卡恢复未解锁状态，并完成新的原厂／解锁配对测试。全程保持 NVIDIA 580.178.04、同一内核、同一二进制与库、80 W 上限。当前显卡已重新解锁，默认解锁启动项已恢复。

本次主要增益来自使用 FP32 累加的 Tensor 路径：TF32 提升 63.1%–64.9%，FP16 输入／FP32 累加提升 53.0%–63.1%。普通 FP32、严格 FP16 在本次配对中没有加速。

## 8192×8192 结果

单位 TFLOPS，取 5 组吞吐中位数。每组连续执行 20 次 GEMM，计时前预热 10 次，FMA 按 2 FLOPs 计。

| 路径 | 未解锁 | 解锁 | 变化 |
|---|---:|---:|---:|
| FP32，无 Tensor | 8.307 | 8.244 | -0.75% |
| 严格 FP16，无 Tensor | 10.514 | 10.446 | -0.64% |
| TF32 Tensor／FP32 累加 | 13.260 | 21.862 | +64.87% |
| FP16 Tensor／FP16 累加 | 42.341 | 42.196 | -0.34% |
| FP16 Tensor／FP32 累加 | 26.167 | 41.011 | +56.73% |

## 全部尺寸

| 路径 | 方阵尺寸 | 未解锁 TFLOPS | 解锁 TFLOPS | 变化 |
|---|---:|---:|---:|---:|
| FP32，无 Tensor | 2048³ | 6.703 | 6.611 | -1.38% |
| FP32，无 Tensor | 4096³ | 6.863 | 6.790 | -1.07% |
| FP32，无 Tensor | 8192³ | 8.307 | 8.244 | -0.75% |
| 严格 FP16，无 Tensor | 2048³ | 11.693 | 11.656 | -0.31% |
| 严格 FP16，无 Tensor | 4096³ | 11.585 | 11.513 | -0.62% |
| 严格 FP16，无 Tensor | 8192³ | 10.514 | 10.446 | -0.64% |
| TF32 Tensor／FP32 累加 | 2048³ | 12.508 | 20.423 | +63.27% |
| TF32 Tensor／FP32 累加 | 4096³ | 13.253 | 21.610 | +63.06% |
| TF32 Tensor／FP32 累加 | 8192³ | 13.260 | 21.862 | +64.87% |
| FP16 Tensor／FP16 累加 | 2048³ | 38.697 | 41.191 | +6.44% |
| FP16 Tensor／FP16 累加 | 4096³ | 41.966 | 41.691 | -0.66% |
| FP16 Tensor／FP16 累加 | 8192³ | 42.341 | 42.196 | -0.34% |
| FP16 Tensor／FP32 累加 | 2048³ | 24.505 | 37.491 | +52.99% |
| FP16 Tensor／FP32 累加 | 4096³ | 26.589 | 43.355 | +63.06% |
| FP16 Tensor／FP32 累加 | 8192³ | 26.167 | 41.011 | +56.73% |

Tensor FP16／FP16 累加在 2048³ 时本次提升 6.44%，4096³ 和 8192³ 则分别为 −0.66%、−0.34%。这是一对短测，不能将小尺寸的变化认定为稳定的解锁增益。普通 FP32 的变化为 −0.75% 至 −1.38%，严格 FP16 为 −0.31% 至 −0.64%。

## 原厂与解锁状态的证据

- 原厂阶段为普通启动，内核命令行不含 `gpu_unlock.targets=`；独立驱动速率接口返回 `fmla32=1`（半速），`imla4=1`。因此基线不是之前暖重启后保留的解锁状态。
- 原厂组成功保存后，选择一次性 `gpu-unlock-ga106m` 启动并重启虚拟机。新启动日志包含目标卡 `pci 0000:00:10.0: gpu_unlock: unlocked`，独立接口变为 `fmla32=0`、`imla4=0`。
- 两组的 GPU UUID、PCI 地址、驱动版本、内核、基准可执行文件与源码、cuBLAS/cuBLASLt/CUPTI 哈希相同。两组测量前后的实际功率上限均为 80 W。
- 两组分别通过 15 个带种子的 CPU 参考检查。30 个大型 GEMM 案例各在计时后完整检查输出，数值错误均为 0。两组内核日志未发现 `NVRM: Xid`、GPU 掉线或 `RmInitAdapter failed`。
- 每组均在独立未计时进程中，通过 CUPTI 获取各模式／尺寸实际执行的 cuBLAS 内核，再核对安装库中对应函数的 SASS。两组各 15 条路径全部确认通过。性能计时进程未启用 CUPTI 跟踪。

## 精度与实际指令

| 路径 | cuBLAS 计算模式 | 已核验的主要指令 |
|---|---|---|
| FP32，无 Tensor | `CUBLAS_COMPUTE_32F_PEDANTIC` | FFMA，无 HMMA |
| 严格 FP16，无 Tensor | `CUBLAS_COMPUTE_16F_PEDANTIC` | HFMA2，无 HMMA |
| TF32 Tensor／FP32 累加 | `CUBLAS_COMPUTE_32F_FAST_TF32` | HMMA，TF32 |
| FP16 Tensor／FP16 累加 | `CUBLAS_COMPUTE_16F` | HMMA，F16 |
| FP16 Tensor／FP32 累加 | `CUBLAS_COMPUTE_32F` | HMMA，F32 |

Tensor“FP32”在本报告中明确指 FP32 存储、TF32 乘法、FP32 累加和输出；乘法精度低于完整 IEEE FP32。参见 [NVIDIA cuBLAS 定义](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t)。严格 FP16 的输入、缩放、计算与输出均为 FP16。

## 输入与条件限制

五条路径采用相同数学输入、种子 1 和尺寸。大型矩阵使用可精确表示的正负 dyadic 数据，每个输出最多累加 1024 个正负 1/256 乘积，其余项为零。这样可同时完整核对严格 FP16 与其他路径的输出。调用普通稠密 GEMM，FLOPs 按完整矩阵尺寸计算；这些数据不能代表一般输入的数值误差。小型参考检查另外使用稠密、带种子的不同正负值。

保持自动时钟策略，未锁频；测量开始前无其他 CUDA 工作负载。100 ms 遥测统计如下，采样功耗不等于瞬时电气峰值。

| 阶段 | 样本数 | 最高观测温度 | 最高观测功耗 | 最高观测 SM 时钟 |
|---|---:|---:|---:|---:|
| 未解锁 | 539 | 53 °C | 80.13 W | 2002 MHz |
| 解锁 | 482 | 54 °C | 80.14 W | 1987 MHz |

这是一张 RTX 3060 Laptop 6 GiB、30 SM 的一次配对测量。没有测量真实模型推理／训练、长时间稳定性，也没有进行多轮 A/B/A；不应将这里的吞吐增益直接当作应用端增益。

## 原始数据与复核

- [经过验证的配对汇总](results/2026-10-25/paired-five-paths-summary.json)
- [完整原始数据](results/2026-10-25/paired-five-paths-results)，包含各次计时、15 项数值检查、CUPTI 内核跟踪、SASS、功率／温度、驱动与启动日志。
- [原始证据归档](results/2026-10-25/paired-five-paths-results.tar.gz)
- [遥测汇总](results/2026-10-25/paired-five-paths-telemetry-summary.json)

```text
python tools/paired_results.py results/2026-10-25/paired-five-paths-results
python -m unittest discover -s tests -p test_paired_benchmark.py -v
```

远端冻结与结果目录：`/home/gpuuser/gpu-unlock-ada/build/paired-five-paths`。当前默认启动为解锁入口，仍保留普通 Ubuntu 入口；默认入口及实验镜像绑定 `6.8.0-147-generic`。取消默认解锁的方法为 `sudo bash /home/gpuuser/3060lp-work/scripts/disable-experiment.sh`，已有覆盖寄存器可能跨虚拟机暖重启保留，需复位显卡并核对速率后才能确认恢复原厂。
