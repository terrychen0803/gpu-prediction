# O1 受控算子 pilot

RTX 3090／4090 節點進行中的測試仍使用 GitHub branch `o1-gb10-pilot-v2-20261021`，commit `8e6dac845bd6a1162f3beb92d6d4497bc19f0eb3`，並依 [v2 節點手冊](../docs/O1_NODE_RUNBOOK.md) 完成。此 checkout 是獨立發布的 GB10 v4 方法分支 `o1-gb10-pilot-v4-20261022`，protocol 為 `o1_operator_pilot_v4_calibrated_repetitions_cuda_reset_20261022`。兩版呼叫次數不同，不得合併為同一協定。GB10 結果、限制及其他節點建立自己校準檔的方法見 [v4 說明](../docs/O1_V4_CALIBRATED_PILOT_20261022.md)。

九個固定 FP16 配置包含 GEMM、BMM、Conv2D 各三個。輸入是 seed 42 的 SHAKE256 byte stream 經固定 FP16 LUT 轉換；18 個 operand 的預期 SHA-256 在 [`pilot_input_sha256.json`](pilot_input_sha256.json)。正確性以 CPU FP32 算子輸出轉 FP16 為 reference，候選門檻 `rtol=atol=1e-2`。每個配置／模式／repeat 都是獨立程序，五組量測各用固定呼叫次數；P/R 必須完全同數。

v4 的 [GB10 校準檔](calibration_gb10_20261022.json)從已完成的 v2 G0/GE 與 v3 P pilot 每組最短 host 時長推算，以每組 125 ms 為校準目標，向上取整至 256 次，且不得超過 16,384 次。R 沿用同配置 P 次數並獨立驗證。runner 需要 `--calibration-manifest`，檢查 protocol、九配置、P/R 次數和 `--expect-gpu`，並把檔案及 SHA-256 複製到原始 session。**其他 GPU 需先建立自己的校準檔與測試證據。**

- `G0`：每次 out-of-place functional call 後同步，保存 host 完成時間，無 profiler。
- `GE`：每次 call 用 CUDA event 量 device elapsed，另存整批 event 平均作輔助。
- `P`：不含逐算子 marker/event/sync 的 Nsight `cuda-sw` trace；每組結尾同步。
- `R`：同 P，但每次 call 有 NVTX reference scope；以 CPU scope 包住的 CUDA launch API、correlation ID 對應 kernel。

P/R 在完成量測後，先 `cudaDeviceReset()` 再 `cudaProfilerStop()`。`trace_audit.json` 檢查 R scope ID 全集合、CUDA API 與 kernel 對應、P/R 已記錄序列。`events_allowlist.jsonl` 是 P 的事件欄位草案，沒有 reference label。Nsight 若回報事件可能未收齊，`rp_alignment.csv` 仍標 `manual_review_required_event_loss`；已記錄事件相符不代表完整 trace。尚無可用的正式 P 逐算子標籤、O2 盲分段或 O3 模型。

在 GB10 上使用新的輸出路徑，例如一次 P/R 短測：

```sh
python -m o1_operator run --out /path/to/new_gb10_pr_session \
  --expect-gpu GB10 --configs bmm_8_64_128_64 --modes P R --repeats 1 \
  --calibration-manifest o1_operator/calibration_gb10_20261022.json
python -m o1_operator summarize --root /path/to/new_gb10_pr_session \
  --out /path/to/new_gb10_pr_analysis
```

校準檔生成依據及 GB10 實測目錄、hash、診斷結果列於 [v4 狀態](../docs/O1_V4_CALIBRATED_PILOT_20261022.md)。完整九配置 × 四模式 × 三重複是 108 個程序；目前只完成九配置單次短測，R 的 NVTX 警告尚未排除，因此不擴收三重複作正式資料。`run_inventory.csv` 保留失敗紀錄，`aggregate.csv` 僅在有三個有效 repeat 時提供聚合。原始 trace 與分析輸出只留在節點資料區，不提交 Git。
