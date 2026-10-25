# RTX 3060 Laptop：115W VBIOS 内存覆盖与 GPU 解锁实测

2026-10-25，在 `192.0.2.10` 的直通 GA106M（`10de:2520`，30 SM，6 GiB）上，组合测试成功：运行时 VBIOS 为 `94.06.13.00.63`，当前、默认、最大功耗限制均为 **115W**；独立硬件查询的 `fmla32`、`fmla16`、`imla4` 均为 **0 / 全速**。默认启动入口已经设置，并通过再次重启验证。

本轮是虚拟机暖重启验证。Tensor 解锁来自前次成功事务的保留状态，本轮日志为 `gpu_unlock: unlock failed (-114)`，随后独立查询确认全速。**尚未验证宿主机断电或显卡冷复位后，在 615 驱动下重新执行解锁的完整组合路径。**

驱动已按授权切换到 **615.71.09 开源模块**，内核为 `6.8.0-147-generic`。本次仅使用内存 VBIOS 覆盖，未向显卡闪存写入 ROM。

用户提供的 `/home/gpuuser/3060-115w.rom` 实际编码为默认 **100W**、最大 **115W**。第一次组合启动已确认采用该 VBIOS，当前限制为 100W，同时保持 Tensor 全速。该卡的 `nvidia-smi -pl 115` 返回“不支持”，即使命令退出码为 0，实际限制仍为 100W。

因此使用 Vix Editor 在新文件 `/home/gpuuser/3060lp-work/3060-default115w.rom` 中，仅将 `power_default_w` 从 100 改为 115，保留已有的 115W 最大值。编辑清单确认只有这一项字段变化。使用 `ampere-relocated-v1` 导出的载荷为 522752 字节，GSP 实际接收 484864 字节，小于 512 KiB 限制。第二次启动后运行时三个功耗限制均为 115W，加载的模块版本、源码版本标识和覆盖参数均与准备的镜像一致。

五路径基准使用此前冻结的同一二进制、源码、cuBLAS 与 CUPTI 库，SHA256 全部一致。矩阵为同一 `same-dyadic-v1` 输入，尺寸 2048、4096、8192；每组预热 10 次，测量 5 轮，每轮 20 次，以下为 TFLOPS 中位数。每次矩阵乘法按完整稠密 GEMM 计数；输入分布是用于严格精度校验的既定分布，结果代表此工作负载。

| 计算路径 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|
| 严格 FP32，无 Tensor | 6.926437 | 7.040192 | 8.927004 |
| 严格 FP16，无 Tensor | 12.136735 | 11.980945 | 12.062705 |
| FP32 输入，TF32 Tensor，FP32 累加 | 22.838573 | 22.532797 | 23.421823 |
| FP16 Tensor，FP16 累加 | 39.424781 | 40.479457 | 41.843009 |
| FP16 Tensor，FP32 累加 | 36.543709 | 43.370890 | 42.989051 |

“FP32 输入 / Tensor”路径实际使用 TF32 乘法与 FP32 累加，不能视为完整 IEEE FP32 乘法精度。严格 FP16 路径使用 `CUBLAS_COMPUTE_16F_PEDANTIC`。独立的非计时 CUPTI 追踪与 SASS 检查覆盖全部 15 条路径、10 个实际内核，分别确认 FFMA、HFMA2、HMMA.TF32、HMMA.F16、HMMA.F32；两条 SIMT 路径没有 Tensor 指令。

15 组测量的全部输出单元校验均通过，另有 15 个小矩阵、三种种子的数值校验通过。测试前后均为 115W 限制与全速硬件状态，内核日志无 Xid、掉卡或初始化失败。100ms 采样共 450 条，观测最高温度 **55°C**、最高功耗 **97.15W**、最高 SM 时钟 **2002MHz**。**115W 是生效的限制，本次负载没有达到 115W 实际消耗；这些采样值也不是电气或散热能力的极限证明。**

与历史 80W、580.178.04、已解锁结果的 8192 阶矩阵对照如下。新结果同时更换了驱动和 VBIOS，差异不能单独归因于功耗提升，也不代表所有工作负载都提高。

| 计算路径 | 历史 80W / 580 | 本次 115W / 615 | 差异 |
|---|---:|---:|---:|
| 严格 FP32，无 Tensor | 8.244142 | 8.927004 | +8.28% |
| 严格 FP16，无 Tensor | 10.446377 | 12.062705 | +15.47% |
| TF32 Tensor，FP32 累加 | 21.862257 | 23.421823 | +7.13% |
| FP16 Tensor，FP16 累加 | 42.196049 | 41.843009 | -0.84% |
| FP16 Tensor，FP32 累加 | 41.011077 | 42.989051 | +4.82% |

使用的固定源码版本：

- [Vix Loader](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader)：`418942a5fe40269787d4b57e97ab0d2a922011cc`。
- [Vix Editor](https://github.com/Panchovix/nvidia-vbios-vix-editor)：`7305642a0ea19401c655b99903de2c3f81b0e82d`。
- [NVIDIA 开源模块 615.71.09](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/615.71.09)：`61dcc93722ecb418bb5f2e00923f05b4b8051dd1`。
- [gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada)：`4e7fefcc4923f9e083d5324dc75a7d2f906af77f`，沿用本仓库的默认关闭 GA106M 实验适配。

Vix 的 Ubuntu/apt 和 Ampere 路径在上游属于实验支持。本地仅补了 BIOS 启动环境下的两处 `mokutil` 兼容检查：只有退出码 1、精确“不支持 EFI 变量”消息、且不存在 `/sys/firmware/efi` 时才接受；UEFI/Secure Boot 检查仍保留。补丁与实际运行时版本记录在原始证据中。另为当前内核设置独立模块的 depmod 优先级。

镜像提取检查确认 `00-gpu-unlock-experiment` 排在 Vix 的 NVIDIA 加载脚本和 udev 之前，逐字节核对 GPU 解锁模块、所需固件和选定 ROM，并验证 NVIDIA 模块的 615.71.09 版本及 per-GPU 补丁元数据。原始 615 普通 initrd 与备份逐字节一致。

关键 SHA256：

| 文件 | SHA256 |
|---|---|
| 用户原始 ROM | `cda18b6b1b913d0700e20b2083f54154e7cde2974eac1497e0aec97845b67fa8` |
| 原始 ROM 的归一化载荷（100W 默认） | `b184ef3350e144df407664160f13e0e3dfda9e4ad1b0acd73a28f9afcc683039` |
| 新的 115W 默认 ROM | `d041d6451f697e356bec8ed70d613497e50f1648cf5a5da4c2b2f7bb09590994` |
| 最终 115W 默认载荷 | `e32f881baea5e74307ac14a612c3344bfcc9f8b5648245d4cb8da44870fded25` |

当前默认 GRUB 入口为 `gpu-unlock-vix-115w`。组合入口绑定当前内核与 615.71.09 模块；升级内核或驱动后需要重建并重新验证。未开启 Vix 自动更新重建服务。

普通 Ubuntu 入口仍保留，选择它会使用原始 VBIOS。恢复普通启动的命令如下；暖重启仍可能保留 Tensor 解锁，清除该硬件状态需显卡物理复位并再次查询验证。

```bash
sudo grub-set-default gnulinux-simple-00000000-0000-4000-8000-000000000001
sudo grub-editenv /boot/grub/grubenv unset next_entry
sudo reboot
```

精确版本的 16 个 580 回退 deb、原始配置、启动镜像和校验清单保存在虚拟机 `/home/gpuuser/3060lp-work/backups/vix115/`。回退 580 时需要同时恢复匹配的内核模块和用户态包；旧 580 实验启动入口已移出当前菜单，避免与 615 用户态混用。

[结构化摘要](results/2026-10-25/vix-combination-summary.json)、[全部原始证据压缩包](results/2026-10-25/vix-combination.tar.gz)、[测量记录](results/2026-10-25/vix-combination/benchmark115w/gemm.jsonl)、[默认启动复验](results/2026-10-25/vix-combination/post-default-reboot/combination-proof.json) 已保存在本仓库。
