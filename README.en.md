# RTX 3060 Laptop GPU Unlock

[中文](README.md) | [English](README.en.md)

An experiment to unlock Tensor instruction rates on a passed-through **RTX 3060 Laptop / GA106M** in a Linux virtual machine, and test a **115W power limit** using an in-memory VBIOS override.

This repository contains an experimental single-card adaptation, boot image integration scripts, a cuBLAS GEMM benchmark, evidence validators, and recorded results. It builds on [gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) and [NVIDIA Linux VBIOS Vix Loader](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader), with VBIOS editing performed using [NVIDIA VBIOS Vix Editor](https://github.com/Panchovix/nvidia-vbios-vix-editor). For safety, power-unlocked VBIOS files are not provided. See the credits below for all upstream projects and pinned revisions.

**Status:** A matched stock 80W versus unlocked 80W comparison was completed on the same card and driver. A separate test verified that the 115W VBIOS override and full Tensor rates can coexist. Only one `10de:2520` card has been tested. The complete 115W combination has not been verified following a host cold boot or physical GPU reset.

## Tested configuration

| Item | Recorded configuration or result |
|---|---|
| GPU | RTX 3060 Laptop, GA106M, 30 SMs, 6 GiB, device ID `10de:2520` |
| System | Ubuntu 24.04, KVM PCIe passthrough, legacy BIOS boot, kernel `6.8.0-147-generic` |
| Stock 80W / unlocked 80W | NVIDIA `580.178.04` proprietary driver, original VBIOS `94.06.17.00.5F` |
| Unlocked 115W | NVIDIA `615.71.09` open kernel modules, overridden VBIOS `94.06.13.00.63` |
| Benchmark toolchain | CUDA Toolkit `12.0.140`, cuBLAS, CUPTI, compilation target `sm_86` |
| 80W unlock evidence | Stock half-rate baseline, a fresh successful unlock transaction, and independent hardware-rate readbacks |
| 115W combination | Current, default, and maximum power limits all report 115W; hardware rates report full speed; the default entry passed a warm-reboot check |
| Untested scope | Other cards, long-term stability, and actual model training or inference |

The 115W run used an unlock state retained from an earlier successful transaction. Its warm-boot log reports `gpu_unlock: unlock failed (-114)`, followed by independent full-rate readbacks. This demonstrates coexistence of the VBIOS override and full Tensor rates; it does not establish that a fresh unlock under driver 615 will complete after a cold reset.

## Benchmark results

These measurements were recorded on **2026-10-25**. Throughput is in **TFLOPS**, reported as the median of five measurement rounds. `N³` denotes `M=N=K=N`.

Both change columns use **stock 80W in the same row** as their baseline: `(configuration throughput / stock 80W throughput - 1) × 100%`. Throughput is rounded to three decimal places; percentages are calculated from the full recorded precision.

**Comparison limits:** The stock and unlocked 80W runs use the same driver, kernel, VBIOS, benchmark binary, and libraries, allowing a matched comparison of this unlock. The 115W run changes both the driver and VBIOS. Its percentage represents the combined configuration's difference from stock 80W and cannot be attributed solely to the unlock or power limit. All three runs use the same benchmark binary, cuBLAS/CUPTI libraries, and mathematical inputs.

<!-- benchmark-table:start -->
| Mode | Matrix size | Stock 80W | CMP unlock + 80W | Change vs. stock | CMP unlock + 115W | Change vs. stock |
|---|---:|---:|---:|---:|---:|---:|
| FP32 (no Tensor) | 8192³ | 8.307 | 8.244 | -0.75% | 8.927 | +7.47% |
| FP16 (no Tensor) | 8192³ | 10.514 | 10.446 | -0.64% | 12.063 | +14.73% |
| TF32 Tensor / FP32 accumulation | 8192³ | 13.260 | 21.862 | +64.87% | 23.422 | +76.63% |
| FP16 Tensor / FP16 accumulation | 8192³ | 42.341 | 42.196 | -0.34% | 41.843 | -1.18% |
| FP16 Tensor / FP32 accumulation | 8192³ | 26.167 | 41.011 | +56.73% | 42.989 | +64.29% |
<!-- benchmark-table:end -->

The TF32 Tensor / FP32 accumulation path takes FP32 inputs but performs TF32 multiplication with FP32 accumulation. Its multiplication precision differs from strict FP32. The strict FP32 and strict FP16 paths both disable Tensor operations.

In the matched 80W comparison, the main gains occur in TF32 / FP32 accumulation and FP16 Tensor / FP32 accumulation. Strict FP32, larger strict FP16 matrices, and Tensor operations with FP16 accumulation do not show comparable gains. These GEMM results are not end-to-end model speedups.

### Method and correctness

- Workload: dense square `C=A×B` with fixed `same-dyadic-v1` inputs. FMA counts as two FLOPs. The input distribution supports strict numerical checks; the results describe this workload.
- Timing: 10 warmups, then five rounds of 20 iterations for each mode and size. CUPTI tracing runs in separate, untimed processes.
- Numerics: every output element is checked, with additional small-matrix checks using three seeds for each mode. The recorded numerical failure count is zero.
- Instructions: actual cuBLAS kernels are traced and their SASS inspected, confirming `FFMA`, `HFMA2`, `HMMA.TF32`, `HMMA.F16`, and `HMMA.F32` across all 15 mode/size paths.
- State: power limits, hardware rates, other GPU workloads, and kernel faults are checked before and after measurement. Binary and library SHA256 values are frozen.

The 115W run collected 450 telemetry samples at 100ms intervals. Observed maxima were **55°C**, **97.15W**, and **2002MHz** SM clock, with no Xid, GPU loss, or initialization failure. **115W is the applied limit; this workload did not consume the full limit.**

See the [80W matched report](PAIRED_BENCHMARK_REPORT.md) and [115W combination report](VIX_COMBINATION_REPORT.md) for the complete results and method. These detailed reports are in Chinese.

## How it works

### Tensor rate unlock

[gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) executes its GPU unlock transaction before the NVIDIA driver binds. The pinned revision used here did not enable the Laptop `2520` device. This repository adds the default-off `ga106m_experimental` option through the [GA106M patch](patches/ga106m-experimental.patch) and [adaptation tool](tools/adapt_ga106m.py).

The experimental path requires `10de:2520`, an explicit opt-in, and one exact PCI address. Requests selecting all GPUs or multiple addresses do not select this experimental card. Firmware and transaction checks follow the pinned upstream version. Success requires transaction evidence and independent hardware-rate readbacks; module loading or a boot parameter alone is insufficient.

### 115W in-memory VBIOS override (VBIOS files not provided for safety)

The tested [Vix Loader](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader) path uses matching NVIDIA open kernel modules to supply a VBIOS payload during GSP initialization. The experiment did not flash the card's ROM.

The supplied ROM encoded a **100W default and a 115W maximum**. The first override confirmed 100W operation alongside full Tensor rates. This card does not support setting the limit with `nvidia-smi -pl`. [Vix Editor](https://github.com/Panchovix/nvidia-vbios-vix-editor) was then used to change only the default to the existing 115W maximum in a new file, exported with `ampere-relocated-v1`.

The final payload is 522752 bytes, of which 484864 bytes are sent to GSP, within the tested path's 512 KiB limit. The runtime VBIOS version and actual power-limit change jointly confirm that the override applied.

This repository also supplies two legacy-BIOS `mokutil` compatibility fixes, explicit module priority for the tested kernel, and initramfs ordering checks that place the unlock before NVIDIA loads. See the [Vix adaptation tool](tools/adapt_vix_loader.py) and [image integration script](scripts/prepare-vix-combination.sh).

## Reproducing the experiment

Account paths, network addresses, GPU UUIDs, and boot identifiers in this release are anonymized examples. Kernel logs retain GPU transaction and fault-related records; measured numerical values are preserved. See the [sanitization notes](SANITIZATION.md), in Chinese, for the publication policy.

The scripts record one VM's workflow and contain fixed paths, device settings, kernel versions, and checksums. Review and configure them for the target system, and retain the original ROM, normal boot entry, and rollback packages matching the driver version.

The SSH tools read `GPU_SSH_HOST`, `GPU_SSH_USER`, `GPU_SSH_HOST_KEY`, and `GPU_SSH_PASSWORD` from the local environment. `GPU_REMOTE_WORKDIR` selects the upload directory. See [.env.example](.env.example); the tools do not load this file automatically. Verify the host key through a trusted channel and supply the password locally when invoking the tools.

| Stage | Files and requirements |
|---|---|
| Environment | [Dependency preparation](scripts/install-dependencies.sh); the target needs matching kernel headers, build tools, DKMS, CUDA/cuBLAS/CUPTI, and driver packages |
| Adapt and freeze | [GA106M adaptation](tools/adapt_ga106m.py), [benchmark preparation](scripts/prepare-paired-benchmark.sh); pin upstream revisions and benchmark files |
| Stock 80W baseline | [Stock boot preparation](scripts/stage-stock-baseline.sh), [paired collection](scripts/collect-paired-benchmark.sh); first confirm half rates and no unlock request |
| Unlocked 80W | [Separate experiment image](scripts/stage-experiment-boot.sh), [paired collection](scripts/collect-paired-benchmark.sh); require a fresh successful transaction and full-rate readbacks |
| 115W combination | [615 driver preparation](scripts/install-vix-driver.sh), [Vix integration](scripts/prepare-vix-combination.sh), [115W default preparation](scripts/prepare-vix-default115w.sh), [combined boot image](scripts/stage-vix-combination.sh) |
| Combined benchmark | [115W collection](scripts/collect-vix-benchmark.sh); measurement starts only after actual power-limit and full-rate checks pass |

The Vix toolchain requires Python 3.11 or newer. The saved-result validators use the Python standard library and can run without connecting to the GPU.

### Local verification

Run from the repository root:

```bash
python tools/paired_results.py results/2026-10-25/paired-five-paths-results
python tools/vix_evidence.py results/2026-10-25/vix-combination/benchmark115w
python -m unittest discover -s tests -p "test_paired_benchmark.py" -v
python -m unittest discover -s tests -p "test_vix*.py" -v
python tools/verify_release.py
```

The first two commands validate recorded evidence. The next two run focused tests for the matched results, instructions, Vix compatibility, and combined state. The last audits publication files, archives, and their sanitized-byte checksums. These commands do not install drivers or trigger a GPU unlock.

### Boot and rollback

At the end of the experiment, the 115W combined entry was made the default and verified with a warm reboot. The normal Ubuntu entry and original initrd were retained. Selecting the normal entry in GRUB restores the original VBIOS. Returning to driver 580 also requires restoring matching kernel modules and userspace packages.

This card retains its Tensor unlock state across VM warm reboots. Read back the actual rates after returning to the normal entry. A clean stock baseline requires a physical GPU reset followed by confirmation of the stock rates. Boot images are tied to the tested kernel and driver and must be rebuilt and revalidated after upgrades. Automatic rebuilding of the combined image was not enabled.

## Recorded data

| Content | Link |
|---|---|
| Stock / unlocked 80W summary | [paired-five-paths-summary.json](results/2026-10-25/paired-five-paths-summary.json) |
| 115W combined summary | [vix-combination-summary.json](results/2026-10-25/vix-combination-summary.json) |
| 80W records, traces, and SASS | [Matched results](results/2026-10-25/paired-five-paths-results) |
| 115W records, traces, and SASS | [Combined benchmark](results/2026-10-25/vix-combination/benchmark115w) |
| Default-entry warm-reboot check | [Reboot evidence](results/2026-10-25/vix-combination/post-default-reboot) |
| 115W image, source, and edit records | [Provenance](results/2026-10-25/vix-combination/provenance) |

## Upstream projects and credits

This experiment uses the following open-source projects. The core unlock path, VBIOS editing/loading capabilities, and NVIDIA driver sources come from these upstreams. This repository provides the single-card GA106M adaptation, integration, and validation records.

| Project | Use in this experiment | Pinned revision |
|---|---|---|
| [JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) | GPU unlock module, firmware generation, and hardware-rate queries | [`4e7fefcc4923`](https://github.com/JustAResearcher/gpu-unlock-ada/commit/4e7fefcc4923f9e083d5324dc75a7d2f906af77f) |
| [Panchovix/nvidia-linux-vbios-vix-loader](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader) | In-memory VBIOS override, driver modules, and separate boot image management | [`418942a5fe40`](https://github.com/Panchovix/nvidia-linux-vbios-vix-loader/commit/418942a5fe40269787d4b57e97ab0d2a922011cc) |
| [Panchovix/nvidia-vbios-vix-editor](https://github.com/Panchovix/nvidia-vbios-vix-editor) | ROM default-power editing, validation, and Ampere payload export | [`7305642a0ea1`](https://github.com/Panchovix/nvidia-vbios-vix-editor/commit/7305642a0ea19401c655b99903de2c3f81b0e82d) |
| [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) | Source for the 615.71.09 open kernel modules | [`615.71.09`](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/615.71.09) / `61dcc93722ec` |

See this repository's [LICENSE](LICENSE) and each upstream's license and copyright notices. NVIDIA firmware and CUDA libraries are obtained through their respective upstream channels. This project is not affiliated with or endorsed by NVIDIA.
