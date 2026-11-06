# O1 v3 CUDA 收尾候選：GB10 局部實測

日期：2026-10-22。這是獨立於已發布 v2 的候選協定 `o1_operator_pilot_v3_fixed_repetitions_cuda_reset_20261022`。其他節點正在執行的 v2 請繼續使用固定 commit `8e6dac845bd6a1162f3beb92d6d4497bc19f0eb3`；不得把 v2/v3 run 當同協定合併。

## 改動與依據

P/R worker 在五組量測及同步完成後、`cudaProfilerStop()` 之前呼叫 `cudaDeviceReset()`，只重置**該程序**的 CUDA context。若 runtime 找不到或 reset 回傳錯誤，worker 失敗並保留 `quality.json`／錯誤檔。`quality.json.profile_finalization` 記錄成功的 reset code。G0/GE 計時邏輯、輸入 bytes、每組 8192 次呼叫及 P/R 配對次數均維持原政策；v3 仍給新 protocol ID，避免舊 run 被回填。依 [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)，使用 `cudaProfilerAPI` 時應在結束分析前完成裝置收尾。

## GB10 實測證據

先用獨立 500-call／40,960-call GEMM 診斷比較 root、一般使用者、`cudaDeviceReset()`、R 的 NVTX capture 及 `stop`／`none`／`stop-shutdown` capture range end。root 本身未消除 CUDA 警告；reset-before-stop 消除了 P/R 的 CUDA 警告；R 的 NVTX 警告在三種收尾下仍存在。診斷原始 trace 在 GB10 `artifacts/runs/o1_gb10_*diag_20261021_v1/` 與 `o1_gb10_v3_r500_nvtx_shutdown_diag_20261022_v1/`，分析在 `analysis_runs/o1_gb10_trace_integrity_probe_20261021_v1/REPORT.md`，含 trace SHA-256。

候選程式實際 runner 量了 GEMM、BMM、Conv 各一個 P/R 配對，共 **6/6 run 完成且 correctness 通過**，source 分別為 GB10 `artifacts/runs/o1_gb10_v3_reset_smoke_20261022_v1/` 與 `o1_gb10_v3_reset_family_smoke_20261022_v1/`。兩個 session 的 measurement code SHA-256 均為 `37363f3cf539636dd21a3690d58c0fcc9707b641379ed6bbc766cb58be70d92b`；input format 仍是 `o1_shake256_lut_fp16_le_v1`。三份 P 都沒有 Nsight CUDA 警告；三份 R 都沒有 CUDA 警告，但有 `Not all NVTX events might have been collected.`。每份 R 記錄了預期的 40,960 個 reference scope，且 P/R 的記錄 kernel 序列相同。GEMM/BMM/Conv 每模式 kernel 數各為 81,920／40,960／163,840。

對應分析為 `analysis_runs/o1_gb10_v3_reset_smoke_20261022_v1/`、`analysis_runs/o1_gb10_v3_reset_family_smoke_20261022_v1/`，及逐 scope ID／kernel 唯一映射稽核 `analysis_runs/o1_gb10_v3_reset_validation_20261022_v1/`。三個 P/R 配對均保留 `manual_review_required_event_loss`；**尚無正式 P 逐算子標籤**。BMM P/R 五組均低於 100 ms，組時長校準仍待新政策；若更改固定呼叫數，再換 protocol ID。v3 尚未量九配置 × 三重複，也未做 O2/O3。

## GB10 候選重現指令

在此 v3 checkout、CUDA 可見且安裝 Nsight Systems 的 GB10 上，使用**全新**輸出路徑：

```sh
/home/good_egg/.conda/envs/pre6g/bin/python -m o1_operator run \
  --out /home/good_egg/RTchen/artifacts/runs/NEW_UNIQUE_V3_SESSION \
  --configs gemm_64_64_64 --modes P R --repeats 1 --expect-gpu GB10
/home/good_egg/.conda/envs/pre6g/bin/python -m o1_operator summarize \
  --root /home/good_egg/RTchen/artifacts/runs/NEW_UNIQUE_V3_SESSION \
  --out /home/good_egg/RTchen/gpu-prediction/analysis_runs/NEW_UNIQUE_V3_ANALYSIS
```

先審查 `trace_audit.json`、`quality.json.profile_finalization`、`rp_alignment.csv` 和 Nsight SQLite diagnostics。不要僅因 CLI 沒顯示警告就判定 trace 完整。v3 候選尚未發布到 GitHub；其他節點請維持已發布 v2。
