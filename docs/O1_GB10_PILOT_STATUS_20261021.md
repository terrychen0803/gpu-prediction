# GB10 O1 九配置 pilot：已完成部分與關卡

日期：2026-10-21。這是**GB10 真實 GPU** 結果摘要；完整原始 session、trace、環境檔、checksum 及詳細分析保存在 GB10 節點，不在 Git。程式位於 [`o1_operator`](../o1_operator/README.md)，其他節點操作見 [O1 節點手冊](O1_NODE_RUNBOOK.md)。量測 protocol 為 `o1_operator_pilot_v2_fixed_repetitions_20261021`，程式 SHA-256 為 `aa513968c3a8cd9f1d9baf5c74c8119cdfdebc78a6f3d0863a8ff56fab064182`；基底 Git commit 為 `73478bfd32a486246a2aecf3c8e59e0f3a839972`，當時新增的 O1 程式尚未提交。

## 協定

FP16 eager、out-of-place `torch.mm`／`torch.bmm`／`F.conv2d`，無 autograd、autocast、CUDA Graph 或 compile；Conv 為 NCHW contiguous、無 bias、groups=1。seed 42 的 SHAKE256＋固定 LUT 提供跨架構一致的 FP16 input bytes，18 個 operand SHA-256 見 [manifest](../o1_operator/pilot_input_sha256.json)。每配置／模式／repeat 獨立程序；預熱至少 100 calls 且 100 ms，每 run 五組，每組固定 8,192 calls。G0 為逐次同步後 host 完成時間；GE 主值為單次 CUDA event elapsed；P 無逐算子 marker/event/sync；R 有逐次 NVTX reference scope。GE 的批次 event 平均另外保存，不與主值混用。正確性用 CPU FP32 operation 後轉 FP16 的 reference，`rtol=atol=1e-2`。

GB10 環境：Linux aarch64、GB10 48 SM、driver 580.159.03、Python 3.12.13、PyTorch 2.13.0+cu130、CUDA runtime 13.0、cuDNN 92000、Nsight Systems 2026.4.1。實際 flags 與環境快照在當地 run `environment.json`。

## G0/GE 真實時間結果

九配置 × 兩模式 × 三重複，共 **54/54 run 完成**；correctness 全數通過。表中 µs/call 為各 run 五組中位數後，三 run 再取中位數；spread 為 `(max−min)/median`，18 個聚合均 ≤5%。

| 配置 | G0 µs | GE µs | G0 spread | GE spread |
|---|---:|---:|---:|---:|
| GEMM 64×64×64 | 11.408 | 12.896 | 0.70% | 1.24% |
| GEMM 256×1024×256 | 14.288 | 13.248 | 0.56% | 3.62% |
| GEMM 1024×1024×1024 | 39.873 | 38.432 | 0.28% | 1.42% |
| BMM 2×32×64×64 | 10.800 | 12.032 | 0.74% | 0.80% |
| BMM 8×64×128×64 | 11.056 | 12.160 | 1.30% | 1.32% |
| BMM 4×256×256×256 | 15.776 | 14.368 | 0.61% | 4.01% |
| Conv 1×16×32×32 k3/s1/p1 | 19.056 | 18.144 | 0.08% | 2.29% |
| Conv 4×64×64×56 k1/s1/p0 | 14.848 | 13.616 | 1.19% | 0.47% |
| Conv 1×64×128×56 k3/s2/p1 | 24.160 | 22.016 | 0.00% | 4.80% |

全部 270 組皆執行 8,192 calls；其中 **30 組 G0 未達 100 ms**，最短 92.916 ms。故穩定性門檻通過，但「每組約 100 ms」候選目標未全數通過。G0 與 GE 是不同時間 target，不比較大小推論 GPU 純運算成本。

## P/R 真實 trace 診斷

GEMM、BMM、Conv 各選一配置做 P/R 各一次，共 **6/6 run 完成**，每模式 40,960 logical calls。已記錄 kernel 數分別為 GEMM P/R 各 81,920、BMM 各 40,960、Conv 各 163,840；各配對已記錄序列一致。R 已記錄 kernel 可經 wrapper scope → CPU CUDA launch API → correlation ID 追到 GPU 工作。但六份 trace 全有 Nsight `Not all CUDA events might have been collected`；三份 R 另有 `Not all NVTX events might have been collected`。Nsight 診斷中的 CUDA/CUPTI produced/collected 數量亦不相同，差值不直接當遺失 kernel 數。

因此三個配對均標 `manual_review_required_event_loss`。**已記錄事件的內部一致性不等於完整 trace，也不能據此建立正式 P 逐算子標籤。** P/R 的 30 個量測組另有 20 組低於 100 ms，最短 78.247 ms。未擴收其他六配置／三重複，未跑 115/139 正式矩陣，未建立 O2 分段或 O3 模型。

## v1 與重現性

早期 v1 自適應呼叫數使 Conv P/R 為 26,410／26,840，無法配對；原資料只留作診斷，不與 v2 合併。v2 固定每組 8,192 calls，修正了這一種 P/R 政策不一致。三個 P/R 配置的輸入 hash 與九配置 G0/GE session 相符；兩個 v2 session 的量測程式 hash 一致。六份原始 `.nsys-rep` 在當地逐檔核對 checksum，共 33,513,787 bytes。GB10 當地工作樹含未發布的 E4 測試，離線測試 42 項中 41 通過、1 項因選配 matplotlib 未安裝跳過；本次公開 O1 checkout 的測試為 36 項中 35 通過、1 項相同原因跳過。合成 parser fixture 不計作 GPU 實測。

下一個關卡是釐清 P/R 事件可能遺失警告，並校準新固定呼叫數政策，使短組達約 100 ms。若改協定，使用新 protocol ID；其他節點先執行同版 pilot 與當地 trace 品質審查。
