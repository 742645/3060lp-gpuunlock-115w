# RTX 3060 Laptop GPU Unlock

[中文](README.md) | [English](README.en.md)

在 Linux 虚拟机中，对直通的 **RTX 3060 Laptop / GA106M** 进行 Tensor 速率解锁，并通过 VBIOS 内存覆盖测试 **115W 功耗上限**。

本仓库包含单卡实验适配、启动镜像集成脚本、cuBLAS GEMM 基准、结果核验工具和原始证据。实验基于 [gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) 与 [NVIDIA Linux VBIOS Vix Loader](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader)，VBIOS 编辑使用 [NVIDIA VBIOS Vix Editor](https://github.com/Panchovix/nvidia-vbios-vix-editor)，为安全不提供解锁功耗vbios文件，完整来源见下方“开源项目与致谢”。

**实验状态：** 已完成同卡、同驱动的原厂 80W / 解锁 80W 配对测试，以及 115W VBIOS 覆盖与 Tensor 全速状态同时生效的测试。仅验证了一张 `10de:2520` 显卡；115W 组合的宿主机冷启动完整链路尚未验证。

## 已验证范围

| 项目 | 实测环境或状态 |
|---|---|
| 显卡 | RTX 3060 Laptop，GA106M，30 SM，6 GiB，设备 ID `10de:2520` |
| 系统 | Ubuntu 24.04，KVM PCIe 直通，BIOS 启动，内核 `6.8.0-147-generic` |
| 原厂 80W / 解锁 80W | NVIDIA `580.178.04` 专有驱动，原始 VBIOS `94.06.17.00.5F` |
| 解锁 115W | NVIDIA `615.71.09` 开源驱动，覆盖 VBIOS `94.06.13.00.63` |
| 基准工具链 | CUDA Toolkit `12.0.140`、cuBLAS、CUPTI，编译目标 `sm_86` |
| 80W 解锁状态 | 原厂半速基线、新解锁事务及独立硬件速率查询均有记录 |
| 115W 组合状态 | 当前、默认、最大功耗限制均为 115W，硬件速率为全速，默认启动暖重启复验通过 |
| 其他硬件与负载 | 其他显卡、长期稳定性、实际模型训练和推理尚未验证 |

115W 组沿用了此前解锁后保留的硬件状态：暖重启日志为 `gpu_unlock: unlock failed (-114)`，独立接口随后确认全速。这组结果证明了 VBIOS 覆盖与 Tensor 全速状态可以共存，尚不能证明冷复位后在 615 驱动下重新完成整个解锁链路。

## 基准结果

以下为 **2026-10-25** 留存实测。吞吐单位为 **TFLOPS**，取 5 轮测量的中位数；矩阵尺寸 `N³` 表示 `M=N=K=N`。

两列“对比解锁前变化”均使用**同一行的原厂 80W**作为基线，计算公式为 `(该组吞吐 / 原厂80W吞吐 - 1) × 100%`。吞吐保留三位小数，百分比使用原始精度计算。

**对比边界：** 原厂 80W 与解锁 80W 使用相同驱动、内核、VBIOS、基准二进制及库，可以用于评估该次解锁的影响。115W 组同时更换了驱动和 VBIOS，其百分比代表组合配置相对原厂 80W 的变化，不能单独归因于解锁或功耗提升。三组使用相同的基准二进制、cuBLAS/CUPTI 库和数学输入。

<!-- benchmark-table:start -->
| 模式 | 矩阵尺寸 | 原厂80w | 解锁cmp+80w | 对比解锁前变化 | 解锁cmp+解锁115w | 对比解锁前变化 |
|---|---:|---:|---:|---:|---:|---:|
| FP32（无 Tensor） | 8192³ | 8.307 | 8.244 | -0.75% | 8.927 | +7.47% |
| FP16（无 Tensor） | 8192³ | 10.514 | 10.446 | -0.64% | 12.063 | +14.73% |
| TF32 Tensor / FP32 累加 | 8192³ | 13.260 | 21.862 | +64.87% | 23.422 | +76.63% |
| FP16 Tensor / FP16 累加 | 8192³ | 42.341 | 42.196 | -0.34% | 41.843 | -1.18% |
| FP16 Tensor / FP32 累加 | 8192³ | 26.167 | 41.011 | +56.73% | 42.989 | +64.29% |
<!-- benchmark-table:end -->

“TF32 Tensor / FP32 累加”使用 FP32 输入，由 TF32 乘法与 FP32 累加完成计算；其乘法精度不同于严格 FP32。严格 FP32 与严格 FP16 两条路径均禁用 Tensor 运算。

80W 的匹配对照中，主要提升来自 TF32 / FP32 累加与 FP16 Tensor / FP32 累加路径。严格 FP32、大尺寸严格 FP16，以及 FP16 累加的 Tensor 路径没有显示同等幅度的提升。上述 GEMM 结果不等同于模型端到端加速。

### 测试方法与正确性

- 工作负载：稠密方阵 `C=A×B`，固定 `same-dyadic-v1` 输入，FMA 计为 2 FLOPs。输入分布用于严格数值检查，结果代表该工作负载。
- 计时：每个模式和尺寸预热 10 次，测量 5 轮，每轮 20 次；CUPTI 追踪在独立的非计时进程中完成。
- 数值：每组检查全部输出单元，另运行每种模式三种种子的小矩阵校验；记录中的数值失败数为 0。
- 指令：独立追踪实际 cuBLAS 内核并检查 SASS，分别确认 `FFMA`、`HFMA2`、`HMMA.TF32`、`HMMA.F16`、`HMMA.F32`，覆盖五种模式的全部 15 条路径。
- 状态：测量前后检查功耗限制、硬件速率、其他 GPU 工作负载和内核错误，冻结二进制及库的 SHA256。

115W 组合测试的 100ms 遥测共 450 条，观测最高温度 **55°C**、最高功耗 **97.15W**、最高 SM 时钟 **2002MHz**，没有 Xid、掉卡或初始化失败。**115W 是已生效的功耗上限，本次负载没有用满该上限。**

完整方法与结果见 [80W 配对报告](PAIRED_BENCHMARK_REPORT.md) 和 [115W 组合报告](VIX_COMBINATION_REPORT.md)。

## 工作原理与本仓库适配

### Tensor 速率解锁

[gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) 在 NVIDIA 驱动绑定前执行 GPU 解锁事务。本次固定版本未启用 Laptop `2520`；本仓库通过 [GA106M 实验补丁](patches/ga106m-experimental.patch) 和 [适配工具](tools/adapt_ga106m.py) 增加默认关闭的 `ga106m_experimental` 开关。

该实验适配要求 `10de:2520`、显式开关和单个精确 PCI 地址；全选或多地址请求不会选中实验卡。固件与事务检查沿用固定上游版本。成功判定结合事务日志和独立硬件速率查询，不以模块加载成功或启动参数存在作为充分证据。

### 115W VBIOS 内存覆盖（为安全不提供解锁功耗vbios文件）

[Vix Loader](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader) 的本次路径使用匹配版本的 NVIDIA 开源内核模块，在 GSP 初始化时传入 VBIOS 载荷。本次没有刷写显卡 ROM。

提供的 ROM 实际编码为**默认 100W、最大 115W**。第一次覆盖后已确认 100W 与 Tensor 全速共存；该卡不支持通过 `nvidia-smi -pl` 调整限制。随后使用 [Vix Editor](https://github.com/Panchovix/nvidia-vbios-vix-editor) 在新文件中仅将默认功耗改为已有的 115W 最大值，并使用 `ampere-relocated-v1` 导出载荷。

最终载荷为 522752 字节，其中 GSP 接收 484864 字节，符合本次路径的 512 KiB 限制。运行时 VBIOS 版本与实际功耗限制变化共同确认了覆盖生效。

本仓库还提供两处 BIOS 环境下的 `mokutil` 兼容处理、当前内核的独立模块优先级，以及解锁早于 NVIDIA 加载的 initramfs 顺序检查。修改范围见 [Vix 适配工具](tools/adapt_vix_loader.py) 与 [镜像集成脚本](scripts/prepare-vix-combination.sh)。

## 复现流程

发布版中的账户路径、网络地址、GPU/启动标识均为匿名样例，内核日志只保留 GPU 事务和错误相关记录；测量值保留原始数值。处理规则见 [脱敏说明](SANITIZATION.md)。

这些脚本记录了特定虚拟机的实验流程，含固定路径、设备、内核和校验值。复用前应按目标环境逐项配置，并保留原始 ROM、普通启动入口和匹配驱动版本的回退包。

SSH 工具从本地环境变量读取 `GPU_SSH_HOST`、`GPU_SSH_USER`、`GPU_SSH_HOST_KEY` 和 `GPU_SSH_PASSWORD`，上传目录由 `GPU_REMOTE_WORKDIR` 指定。变量说明见 [.env.example](.env.example)；工具不会自动加载该文件。主机密钥应通过可信渠道核对，密码只在本地调用时设置。

| 阶段 | 对应文件与要求 |
|---|---|
| 准备环境 | [依赖安装脚本](scripts/install-dependencies.sh)；目标机需要内核头文件、编译工具、DKMS、CUDA/cuBLAS/CUPTI 和匹配驱动 |
| 适配与冻结 | [GA106M 适配](tools/adapt_ga106m.py)、[路径基准准备](scripts/prepare-paired-benchmark.sh)；固定上游版本与基准文件 |
| 原厂 80W 基线 | [原厂启动准备](scripts/stage-stock-baseline.sh)、[配对采集](scripts/collect-paired-benchmark.sh)；先确认硬件为半速且没有解锁请求 |
| 解锁 80W | [独立实验启动镜像](scripts/stage-experiment-boot.sh)、[配对采集](scripts/collect-paired-benchmark.sh)；要求新的成功事务与全速读数 |
| 115W 组合 | [615 驱动准备](scripts/install-vix-driver.sh)、[Vix 集成](scripts/prepare-vix-combination.sh)、[115W 默认值准备](scripts/prepare-vix-default115w.sh)、[组合启动镜像](scripts/stage-vix-combination.sh) |
| 组合基准 | [115W 采集脚本](scripts/collect-vix-benchmark.sh)；实际功耗限制与全速状态通过门控后才测量 |

Vix 工具链需要 Python 3.11 或更新版本。本地保存结果的核验工具使用 Python 标准库，可以在不连接显卡的情况下复核。

### 本地复核

在本仓库根目录运行：

```bash
python tools/paired_results.py results/2026-10-25/paired-five-paths-results
python tools/vix_evidence.py results/2026-10-25/vix-combination/benchmark115w
python -m unittest discover -s tests -p "test_paired_benchmark.py" -v
python -m unittest discover -s tests -p "test_vix*.py" -v
python tools/verify_release.py
```

前两条命令核验保存的实验记录，随后两条命令运行配对结果、指令、Vix 兼容与组合状态的定向测试，最后一条命令检查发布文件、证据包及脱敏后的字节校验清单；这些命令不会安装驱动或触发显卡解锁。

### 启动与回退

实验收尾时，115W 组合入口设为默认，并完成一次暖重启复验；普通 Ubuntu 入口及原始 initrd 保留。通过 GRUB 选择普通入口可恢复使用原始 VBIOS。恢复 580 驱动还需要同时恢复匹配的内核模块和用户态包。

本卡的 Tensor 解锁状态能够跨虚拟机暖重启保留。恢复普通入口后仍须查询实际速率；获得干净原厂基线需要物理显卡复位，并确认速率恢复。启动镜像绑定当前内核与驱动版本，升级后需要重建和重新验证。本次未启用组合镜像的自动重建。

## 原始数据

| 内容 | 链接 |
|---|---|
| 原厂 80W / 解锁 80W 结构化汇总 | [paired-five-paths-summary.json](results/2026-10-25/paired-five-paths-summary.json) |
| 115W 组合结构化汇总 | [vix-combination-summary.json](results/2026-10-25/vix-combination-summary.json) |
| 80W 原始记录、追踪和 SASS | [配对结果目录](results/2026-10-25/paired-five-paths-results) |
| 115W 原始记录、追踪和 SASS | [组合基准目录](results/2026-10-25/vix-combination/benchmark115w) |
| 默认启动暖重启复验 | [启动复验目录](results/2026-10-25/vix-combination/post-default-reboot) |
| 115W 镜像、来源与编辑清单 | [provenance](results/2026-10-25/vix-combination/provenance) |

## 开源项目与致谢

本实验使用了以下开源项目。核心解锁路径、VBIOS 编辑/加载能力和 NVIDIA 驱动源码分别来自这些上游，本仓库提供针对单张 GA106M 的实验适配、集成与验证记录。

| 项目 | 本实验中的用途 | 实测固定版本 |
|---|---|---|
| [JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) | GPU 解锁模块、固件生成与硬件速率查询 | [`4e7fefcc4923`](https://github.com/JustAResearcher/gpu-unlock-ada/commit/4e7fefcc4923f9e083d5324dc75a7d2f906af77f) |
| [Panchovix/nvidia-linux-vbios-vix-loader](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader) | VBIOS 内存覆盖、驱动模块与独立启动镜像管理 | [`418942a5fe40`](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader/commit/418942a5fe40269787d4b57e97ab0d2a922011cc) |
| [Panchovix/nvidia-vbios-vix-editor](https://github.com/Panchovix/nvidia-vbios-vix-editor) | ROM 默认功耗编辑、校验与 Ampere 载荷导出 | [`7305642a0ea1`](https://github.com/Panchovix/nvidia-vbios-vix-editor/commit/7305642a0ea19401c655b99903de2c3f81b0e82d) |
| [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) | 615.71.09 开源内核模块源码 | [`615.71.09`](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/615.71.09) / `61dcc93722ec` |

许可证见本仓库 [LICENSE](LICENSE)，以及各上游的许可证和版权声明。实验所用 NVIDIA 固件与 CUDA 库由各自的上游渠道提供。本项目与 NVIDIA 无隶属或背书关系。
