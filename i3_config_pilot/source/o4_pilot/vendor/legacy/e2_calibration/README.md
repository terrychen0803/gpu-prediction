# E2 硬體校正（v1）

RTX 3090／4090 節點的完整環境檢查、執行、驗收、Nsight 短驗證及封存流程，請看 [E2 節點操作手冊](../docs/E2_NODE_RUNBOOK.md)。可直接交給當地 Codex 的任務文字位於手冊末節。

這是與 E1 ResNet50 session 分開的 CUDA 實測工具。正式協定為 `e2_calibration_v3`。目標是取得 Compute、Memory、Launch、Host／transfer 的有效曲線，不把型錄峰值或 Nsight kernel busy time 當成實測吞吐。使用 Python 3.10+、可用的 CUDA PyTorch 和支援當地 GPU 的 `nvcc`。每台以相同原始碼執行，正式資料存到新目錄。

```sh
python -m e2_calibration --label gb10 --expect-gpu GB10 --smoke --out results/e2_gb10_smoke_001
python -m e2_calibration --label gb10 --expect-gpu GB10 --out results/e2_gb10_full_001
python -m e2_calibration.summarize results/e2_gb10_full_001
```

3090／4090 分別改為 `--label rtx3090 --expect-gpu 3090`、`--label rtx4090 --expect-gpu 4090`。如有多張卡，先以 `CUDA_VISIBLE_DEVICES=<目標 GPU UUID>` 限定裝置；程式仍會核對實際名稱與 UUID。若要從遠端執行，須在該節點使用其本地 CUDA/PyTorch 環境。**smoke 為功能檢查，不納入正式曲線。** 每次使用新的 `--out`；已有目錄會拒絕覆寫。

正式矩陣共 100 個 cell：FP16 GEMM 九組、device copy/read/write × 七個工作集 × 兩種 buffer 政策共 42 組、empty/short kernel × N=1/10/100/1000 × 兩種同步政策共 16 組、H2D/D2H × 八個大小 × pinned/pageable 共 32 組、無排隊工作的一次 host synchronize。每個 cell 在三個獨立程序中各有五個計時群組，目標每群約 100 ms。kernel 採本目錄 [CUDA 原始碼](kernels.cu)，GEMM 採 `torch.mm`，transfer 採 PyTorch copy；各 cell 先在計時外做可驗證的功能檢查。

`runNN/microbench_samples.csv` 為原始群組資料，保留 CUDA event 時長、CPU enqueue 時長、CPU completion 時長、UTC 時間與 monotonic 時間。`microbench_samples.csv` 為三程序合併副本；`curves.csv` 先在每程序取五群中位數，再取三程序中位數；`hardware_profile.json` 記錄環境、原始碼／編譯產物 hash、群組時長不足清單與預先選定錨點。`plan.json` 與 `runNN/environment.json` 保留失敗狀態。中途失敗不把部分資料標記為完成。

Compute 的 FLOPs 定義為 `2MNK`，Memory copy 的邏輯 bytes 為 `2N`，read/write 與 transfer 為 `N`。所得有效 FLOP/s 或 byte/s 不是 GPU 峰值、實際 DRAM 交易或 cache hit rate。Memory 的 `same_buffer` 與 `rotating_4_buffers` 只是重用政策；沒有證明工作集超過當地 cache 容量前，不稱冷／熱。GB10 的 transfer 路徑與離散 GPU PCIe 不應直接按連線名稱解讀。

Launch 的 `batched` 模式連續提交 N 次後同步；`sync_each` 每次提交後同步。CSV 保留 N 次的原始總時長及 N，曲線另列每次完成平均。`sync_each` 的 CUDA event span 包含 host 間隙，主曲線以 host completion 表示；host synchronize cell 是無排隊工作的 Python 呼叫成本。這些數值包含 Python／ctypes 調用，不代表純硬體 launch latency。少量 Nsight trace 須另外確認 `e2_empty_kernel`／`e2_short_kernel` 真正出現在 GPU 軌跡；未確認前不使用 Launch 錨點作正式模型特徵。

若節點的 `nvcc`、pinned memory 或 CUDA API 不支援，保留失敗目錄與 log，不能用 CPU 數字代替。不同 PyTorch、CUDA toolkit、GPU 或來源碼 hash 的資料分組保存，不直接混成同一 profile。RTX 5090 尚待設備可用；這版工具不宣稱四設備 E2 已完成。
