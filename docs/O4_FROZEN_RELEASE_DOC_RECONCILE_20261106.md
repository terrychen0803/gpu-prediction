# O4 凍結 Collector：Source Release Manifest 文件 SHA 勘誤（2026-11-06）

**範圍：** 只更新 `o4_pilot/manifests/code_release.json` 中一個非執行文件的 expected SHA-256；原版 `code_release_v1_historical.json` 原文保留。**不變更** `o4_iteration` Python/CUDA Collector、`scripts/run_node.sh`、`scripts/verify_assets.py`、78 個原始被查核檔案的實體內容、`manifests/protocol.json` 或 `manifests/assets.json`。

## RTX3090 上發現的問題

在獨立 checkout `b204ae63ee2dff2bbc161bb5cc8f9aaecfb68c01`，同一台 RTX3090 先從 GB10 取得完整原版 `O4/assets/`（其中 RN-A 102,546,585B 權重、B32 共用輸入及三個小算子 operand），執行 `scripts/verify_assets.py` 發現：

```text
Release file mismatch: docs/NEXT_STEPS_ZH_TW.md
```

使用者另執行 **完整、唯讀 78/78 source file SHA 核對**，結果是且只有：

| Path | v1 expected SHA256 | b204 source checkout actual SHA256 |
| --- | --- | --- |
| `docs/NEXT_STEPS_ZH_TW.md` | `165e958426e7749dd6e60265bf11ddf77c4e892df49d0470f07ebdb36a927410` | `480dcf1fc0f3e26e18e1175b099776e88e8742a269d605feeb52a2b3b109b029` |

Git working tree `git status --short` 空白，其餘 77 個檔案 SHA 正確。原驗證程式先驗 asset 再查 source：這只證明當次資產項目驗證未在首個文件錯誤前失敗；請在新分支重跑完整工具，取得明確 `PASS` 證據。

## 勘誤內容與保障

- `manifests/code_release_v1_historical.json` **完全複製** v1，保留歷史發行狀態。
- `manifests/code_release.json` 宣告新的 `schema=o4_github_source_release_v2_docs_reconciled_20261106`，將上述一個文件 expected SHA 更新為實際 checkout 值。加上 `release_reconciliation` provenance 欄位，提供舊／新 hash 與基底 commit。
- 量測 Protocol SHA 繼續為 `28c25fe1af6935b517d8372d9a078b30f26237d1a1471a60c7038fd43022617d`；原版資產 manifest SHA 繼續為 `0a078f3d8bd2ed790d71c518a3f295e81d78e841d0e990bb47d3a45d3f431ca3`。
- 不覆寫 RTX3090 先前 I1/I2 七組 A 的 raw 或 strict quality；原始 3-measured Development Smoke 也不會被當成 40×3 Pilot。
- 程式碼更新需在 `o4/frozen-release-doc-reconcile-20261106` 對 `main` 的獨立 PR 審查，避免將此發行修復和 I1/I2 新研究程式混為同一歷史 commit。

## RTX3090 的驗證及收集順序

獨立 `gpu-prediction-o4-frozen` checkout 且 `git status --short` 為空時，fetch/switch 到勘誤分支最新提交，然後：

```bash
cd "$HOME/RTchen/gpu-prediction-o4-frozen/o4_pilot"
export PYTHON="$(command -v python)"
"$PYTHON" scripts/verify_assets.py
"$PYTHON" -m unittest discover -s tests -q
(cd vendor/legacy && "$PYTHON" -m unittest discover -s tests -q)
nvidia-smi
PYTHON="$PYTHON" bash scripts/run_node.sh 3090 single
```

只在新 `single` 的三個 Cell 與獨立 Quality Report 審核後，才執行 `three`（27 Cell）與 `resnet`（9 Cell）。單一程序 40 measured／5 transition 和三程序 repeated 40 值均按照原版 Protocol，不以凍結八 warmup/三 measured 的 I2 Smoke 替代。**此段為早期操作順序；RTX 3090 目前已完成 single、three、resnet 的全部 39 個工程 Cell，實際成績見文件末尾。**

若 fail：保留新 run `analysis_runs/o4_3090_*`、`reports` 與 error log；不修改 driver 或底層程式、也不消除 warning 並改寫 verified。所有 CLI 結果的 `collector_git_commit` 應指向**勘誤分支**實際 commit，不能在報告中稱使用未修改的 b204 commit。

## 2026-11-06 RTX3090 節點後續驗證及 Single Pilot 實績

使用者在獨立 frozen checkout 更新至 `2ca671ad8c4d5b73d92efcb07c31ed62971edf0b`，與原 b204 code/protocol/assets `git diff` **空白**。實際 RTX3090 驗證結果：

- `scripts/verify_assets.py`：**PASS: 9 shared asset hashes; release code checked when available; no CUDA work**。
- O4 offline tests：**25/25 PASS**；Legacy tests：**54/54 PASS**；`pip check` 無 broken requirements。
- Preflight：NVIDIA RTX3090，driver 580.178.04，CUDA 13.0；GPU 0% utilization、無 compute processes。
- `bash scripts/run_node.sh 3090 single`：**3/3 completed**、0 failed，P/R Trace **2/2 structural passes**、**1/1 exact ordered P/R signature**，120 valid measured iterations（GI/P/R × 40），GPU overlap 0、clock inversions 0。
- 原始 run：`analysis_runs/o4_3090_single_20261106T141008Z`（保存在節點，未上傳 Git）。
- `collector_git_commit=2ca671ad8c4d5b73d92efcb07c31ed62971edf0b`；`protocol_id=o4_gpu_iteration_smoke_v2_20261106`。
- Single GEMM `gemm_16_64_64` 的單程序 40 iterations 中位數：GI **13.312 µs**，P **25.040 µs**，R **56.544 µs**。不可將 P/R 的 profiler interval 與 GI 相互代替。
- `process spread=0.00%` 僅因 repeats=1，Reporter 已註明 **未通過／重複不足**，**不是跨程序穩定性 PASS**。
- P/R 警告 **`Not all NVTX events might have been collected.`** 保留；`reference_gate=not_verified`、`strict_verified_operations=0`，不是正式 Reference。
- `REPORT.md` 標題仍寫「O4 GB10」，且模板文字寫「未取得4090／3090新資料」；這與當次 `inventory.json.expected_gpu=3090`、實測輸出不一致，屬報告模板過時。此次**不修改凍結報告生成程式**；統計以當次 inventory/snapshot/trace 為準。

**接續門檻：** 可以開始工程 `three`（三小算子 × GI/P/R × 3 independent processes = 27 Cells）；需查所有 27 Cell、重複 spread、P/R 9 對結構／NVTX／clock、正式 Reference gate。通過工程品質審核再做 `resnet`（9 Cells）。不更改 release assets、工作負載、計時協定、不混用 I2 3-measured smoke，也不會因 warning-accepted 自動升級 strict Reference。

## 2026-11-06 RTX3090 原版 Three Pilot 與 ResNet 40×3 完成

以下數值皆由 RTX3090 節點實際收集並貼出的 `inventory.json`、`reports/SUMMARY.json`、`reports/aggregate.csv` 和 `reports/pair_summary.json` 提供，**不是遠端 CI、GitHub 實測或正式 confirmation set**。收集時 checkout `2ca671ad8c4d5b73d92efcb07c31ed62971edf0b`，`scripts/verify_assets.py` 9 個 asset hashes PASS；driver 580.178.04、CUDA 13.0，執行前 GPU 0% 且沒有 compute process。

### Three Pilot：三小算子 GI/P/R × 3 repeats

- Raw run：`analysis_runs/o4_3090_three_20261106T141407Z`。
- **27/27 Cells completed，0 failed，1,080 valid measured iterations**，P/R 18/18 Trace structural passes，9/9 P/R pairs exact（每對 40/40 measured）。
- `composition_supported_traces=18`，`overlapping_recorded_work_events=0`，`cpu_gpu_clock_inversions=0`。
- **8/9 配置×模式三程序 spread ≤5%**；唯一未達門檻的是 `gemm_16_64_64` **R = 29.862351%**。不得刪除較慢程序或宣稱 9/9 stable。

| Workload | Mode | median of 3 process medians (µs) | Spread (%) | Stability |
| --- | --- | ---: | ---: | --- |
| GEMM | GI | 13.312 | 0.1202 | PASS |
| GEMM | P | 23.904 | 3.0120 | PASS |
| GEMM | R | 53.472 | **29.8624** | **FAIL** |
| BMM | GI | 20.480 | 0.1563 | PASS |
| BMM | P | 32.720 | 2.4450 | PASS |
| BMM | R | 60.400 | 2.3576 | PASS |
| Conv | GI | 22.528 | 0.0000 | PASS |
| Conv | P | 31.552 | 4.8682 | PASS |
| Conv | R | 66.656 | 4.7768 | PASS |

原版三小算子每 Operation 皆 1 個已記錄 GPU Event，原版三類相關 event/busy coverage=100%；**只對這三個特定算子配置成立**，不代表 DNN 新 family 與 shapes 都支援。全部 P/R 仍出現 `Not all NVTX events might have been collected.`，`reference_gate=not_verified`，`strict_verified_operations=0`。

### ResNet RN-A 40×3 Pilot

- Raw run：`analysis_runs/o4_3090_resnet_20261106T142247Z`。
- **9/9 Cells completed，0 failed，360 valid measured iterations**；6/6 P/R Trace structural passes；3/3 P/R pairs exact（每對 40/40 measured）；overlap 與 CPU/GPU clock inversion 均 **0**。
- **GI/P/R 三種模式全部三程序 spread ≤5%**，數值：

| Mode | Median of 3 process medians (µs) | Process spread (%) | Stability |
| --- | ---: | ---: | --- |
| GI | **2971.135974** | 2.101286 | PASS |
| P | **4759.919882** | 0.705891 | PASS |
| R | **10658.607960** | 0.421215 | PASS |

- 120 個 measured forward（3 repeats × 40），每 forward 有 **175 個 Reference ATen**：共 **21,000** Reference Operation，**20,880** 有 recorded GPU work、**120** zero-recorded-GPU work（即每 forward 1 個）。`max_events_per_operation=5`；共 **480** 個 >L=4（每 forward 4 個），這些 operations 的事件佔已記錄 GPU Events **5.813953%**。
- 舊 GEMM/BMM/Conv 三家族**相關** recorded events **34.011628%**、recorded busy union **60.289245%**。此為 heuristic family relation，**不是現成 N0 模型的可預測覆蓋率**。
- `reference_gate=not_verified`，`strict_verified_operations=0`；全部 P/R 仍有 `Not all NVTX events might have been collected.`。對齊、closure、repeat stability 不會自動解除 Reference 完整性疑慮。
- 不把 R 的 profiled iteration event 時間（約 10.659 ms）當 GI 真值（約 2.971 ms）；也不把 I2 三樣本工程 smoke 與此 40×3 Pilot 混合訓練。

### RTX3090 已達成範圍及未完成項目

原版 Frozen Pilot **single 3 + three 27 + resnet 9 = 39/39 工程 Cells completed**；其中 three 有 GEMM R spread 29.86% 的**獨立穩定性失敗**，整套 Reference 仍為 **unverified**。

另一方面，新 I2 七 A 的獨立 3-measured engineering smoke 已完成 21/21 Cells、7/7 inventory、14/14 trace replay，見 [RTX3090 I1/I2 紀錄](https://github.com/terrychen0803/gpu-prediction/blob/o4/rtx3090-evidence-compat-20261106/docs/O4_RTX3090_I1_I2_20261106.md)，**不是此原版 40×3 Pilot**。

後續順序：保留與唯讀審查 GEMM R 三程序失穩及所有 NVTX/CUPTI 事件警告；對 MV/SW/YO 未匹配 `cudaMemsetAsync` 作獨立參數診斷，不猜測 0B；再為七 A 新 Collector 制定版本化 40×3 協定與不可洩漏資料分割。RTX4090 的主來源 P 是下階段獨立收集任務，不將目標 GPU Reference 或 GI 倒灌入 predictor。

## 2026-11-06 臨時研究決策：NVTX Warning 於 Engineering 階段暫時不阻擋（未確認軟體 Bug）

研究者目前**懷疑** Nsight Systems 的 `Not all NVTX events might have been collected.` 是軟體／工具層面的非預期警告，但**尚無可確認根因的修復版本、NVIDIA Issue 或事件完整性獨立證據**。以下為研究進度上的**暫行風險接受決策**，**不是**已證實 Nsight 有 Bug，亦非證明所有 NVTX/CUDA events 已完整收集。

- Engineering 層級：**暫不將這個單一 NVTX Warning 當作阻擋 GPU 收集、既有 Trace 結構分析、P/R 簽章檢查、穩定性診斷與後續探索的硬性失敗條件**。在其他獨立檢查皆通過時，可標記 `warning_accepted_for_engineering`（**只作文字紀錄，不竄改原始 JSON 狀態或程式 Gate**）。
- Evidence 層級：原始 `DIAGNOSTIC_EVENT`、`warnings`、`trace_status=warning_accepted`、`reference_gate=not_verified`、`strict_verified_operations=0` **全部維持原樣**。不能因暫時忽略 Warning 就推論無 Event Loss、提高 Reference 級別、把 R 的時間視為 GI，或將資料升級成正式可用的嚴格 Operation Ground Truth。
- 其他品質問題**不在豁免範圍**：MV/SW/YO 未匹配 `cudaMemsetAsync` 的 `count_bytes` 尚未驗證；部分 capture 的 SQLite/Event completeness 限制、Operation→Kernel mapping、未知或缺失 GPU work 仍需個案審核。對新設備、新 Nsight 版本、新 workload 不得自動沿用本決策。
- 將來如果 NVTX scope 缺失、P/R 簽章不一致、明確 CUPTI event dropped、trace 結構或 clock/correlation 故障，**必須另行 FAIL／Unverified**，不能用本註記概括忽略。
- 在寫論文與對外報告時應描述為「Nsight NVTX warning was provisionally accepted for engineering exploration; strict reference completeness remains unverified」，不得描述為「confirmed NVIDIA software bug」或「lossless trace」。

### GEMM R 三程序失穩仍維持 FAIL

`o4_3090_three_20261106T141407Z` 的三次 `gemm_16_64_64` R Process 詳細資料：

| Repeat | 40 measured 中位數 (µs) | P10 (µs) | P90 (µs) | Warmup | Trace status |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | **69.055997** | 67.520000 | 74.336000 | 60 | warning_accepted |
| 2 | 53.088000 | 52.191999 | 57.344001 | 60 | warning_accepted |
| 3 | 53.472001 | 52.096002 | 57.824001 | 60 | warning_accepted |

Repeat 1 的程序整體偏慢，而 repeats 2、3 相近；三程序 `spread=(max(medians)-min(medians))/median(medians)` = **29.862351%**，仍超過 5% 門檻。當前沒有證據能將這個執行時間差直接歸因於 NVTX 警告；**暫時忽略警告不等於忽略 GEMM R Stability Failure**。保留所有 120 筆 measured 原始資料，不剔除 repeat 1，也不改寫三程序穩定性報表。

**影響範圍：** 本決策只改變工程階段對已知警告的處理政策；**不修改 Frozen Collector、Protocol、Release Manifest、測量數值或 Strict Quality Gate**。如需程式支援 explicit warning-policy，必須另開版本與對照測試。
