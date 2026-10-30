# RTX 3090 O1 Formal 139 重收集與時間穩定性診斷報告

- **實驗日期：** 2026-10-30（RTX 3090 重新 Bootstrap 與正式收集）；其後完成 9-config/54-run 補充診斷。
- **節點：** NVIDIA GeForce RTX 3090（單一可見 CUDA GPU）。
- **研究目的：** 在原始 RTX 3090 O1 時間資料受到背景工作負載干擾、三次重複穩定性不佳的情況下，重新收集具有相同 O1 Formal 139 配置與 protocol 的資料，區分系統干擾與不同計時方式引起的變異。
- **方法來源：** [O1 formal branch](https://github.com/terrychen0803/gpu-prediction/tree/o1-formal139-method-20261022)，固定 Git commit **89431b415ebf4cbe3c886795b39d7b7aba32e7a8**。
- **Bootstrap protocol：** `o1_formal_139_bootstrap_8192_v2_20261022`
- **Formal protocol：** `o1_formal_139_calibrated_v2_20261022`
- **研究狀態：** 收集完整性 **PASS**；原定嚴格時間品質（所有 G0/GE spread ≤5%）**未完全通過**。**不得將 collection PASS 說成 timing-quality PASS。**

## 1. 方法與環境

O1 Formal 139 的算子配置為 GEMM 27、BMM 24、Conv2D 64、Add 12、LayerNorm 12，全部維持原本 frozen catalog、輸入生成與 split。不因為 RTX 3090 的結果修改算子形狀、5% 門檻或 O1 protocol。

本次重收集沿用同一實作，於機器近乎閒置的狀態啟動。執行前紀錄顯示 GPU 無執行中的程序，GPU 利用率 0%，顯示記憶體使用量 1 MiB，CPU load average 為 0.22、0.16、1.15；這是**啟動前快照**，不是整個實驗期間無背景負載的直接證明。

- PyTorch：**2.13.0+cu130**。
- PyTorch CUDA runtime：**13.0**。
- NVIDIA Driver：**580.178.04**。
- GPU：**GeForce RTX 3090，24 GiB**。
- 程式單元測試：**46 tests，OK（1 skipped）**。
- 執行前 PyTorch 顯示缺少 NumPy 的警告；單元測試與資料收集均完成，此警告應在後續環境重現時保留紀錄。

### Bootstrap 與 Calibration

每個配置的 G0、GE、P 各執行一個 bootstrap repeat，總計 **139 × 3 = 417 runs**；每次量測為五組，每組 8,192 calls。依各配置/模式 bootstrap 最短組時間，將正式 calls/group 校準到目標約 125 ms（呼叫數以 256 為單位向上取整，至多 65,536）。R 與 P 使用相同 calls/group。

- Bootstrap：**417/417 completed**。
- 新 Calibration manifest SHA-256：**0a4df0fc7990dac77f931ea7347e922e4dbd867563bedd586afe01a9a9f3e294**。
- 為了不中斷本次收集，Calibration 使用 `--allow-p-warnings`；此選項只允許帶警告的 P trace 作為校準來源，**並非 NVTX/reference 完整性已被證實**。

## 2. 正式收集結果：舊資料與新資料

正式收集每個配置執行 G0、GE、P、R 各三次，**139 × 4 × 3 = 1,668 runs**。評估中 G0 是單次操作加同步的 host completion time；GE 主標籤是逐次 CUDA Event device elapsed time；P/R 是未標記/加 NVTX 參考的 trace，不可混為同一種時間標籤。

| 指標 | 舊 RTX 3090 Formal | 新 RTX 3090 clean Formal |
|---|---:|---:|
| Formal 完成數 | 1,668 / 1,668 | **1,668 / 1,668** |
| 失敗數 | 0 | **0** |
| 三次重複 G0/GE spread ≤ 5% | 118 / 278（42.4%） | **244 / 278（87.8%）** |
| 所有 group ≥ 100 ms 的配置/模式聚合 | 524 / 556 | **556 / 556** |
| P/R recorded structure alignment | 417 / 417 | **417 / 417** |
| Trace warning runs | 417 | **417** |
| Reference label completeness | 須人工審查 | **須人工審查** |

時間穩定性通過率改善約 **45.3 個百分點**。以每配置/模式的狀態比較：**147 組 FAIL→PASS、21 組 PASS→FAIL、110 組狀態不變**。因此不能把每個配置的改善都歸因於某一個固定的環境因素；但總體改善與減少外部干擾的目標一致。

**新正式資料驗收：** collection PASS；timing quality PARTIAL/FAIL（距離預先設定的 278/278 嚴格標準還有 34 組），故原驗收腳本沒有刪除舊資料。

## 3. 剩餘 34 組不穩定資料

採用每個 run 的五組 single-call median，再比較三個 repeat 的

\[
\operatorname{spread}=\frac{\max(t_1,t_2,t_3)-\min(t_1,t_2,t_3)}{\operatorname{median}(t_1,t_2,t_3)}
\]

以 spread ≤5% 判定穩定性。新資料不穩定 34 組，**GE 28 組、G0 6 組**；按算子家族統計：GEMM 14、Conv2D 9、Add 6、LayerNorm 5，BMM 0。GE 不穩定的許多算子僅約 9–21 µs；約 1 µs 的絕對變化就可能造成超過 5% 的相對 spread。

原始 GE 每次操作使用 CUDA Event 並在每個 call 後同步；同一 run 另外收集 `batch_event_us_per_call`，在連續多次 operation 前後放 Event，將整批時間除以 calls。後者與逐次同步 **不是同一種 estimand／預測標籤**。

| GE 三次重複穩定性（5% 門檻） | 配置數（共 139） |
|---|---:|
| Single-call stable | **111** |
| Batch-event stable | **125** |
| Single、Batch 皆穩定 | 101 |
| Single 不穩定、Batch 穩定 | 24 |
| Single 穩定、Batch 不穩定 | 10 |
| Single、Batch 皆不穩定 | 4 |

在 **28 組**原本 GE Single-call 不穩定配置中，**24 組（85.7%）**使用 Batch Event 三次重複時通過 5% 門檻。這支持進一步研究逐次同步、量測解析度、CUDA Event 與 CPU/GPU 提交節奏的影響，但**不能僅由此證實**硬體計時解析度為 1.024 µs，也不能以 Batch 數值悄悄覆蓋 Single-call 正式標籤。

## 4. Critical 9 診斷：54 runs

以新 Calibration manifest，對 9 個重點配置各收集 G0、GE × 3 repeats，**9 × 2 × 3 = 54 runs**；目的為檢查部分不穩定現象能否重現，**不是挑選合格數字替換原正式量測**。

| 配置 | G0 spread | GE spread |
|---|---:|---:|
| add_65536 | 2.83% PASS | 8.92% FAIL |
| add_32768 | 2.35% PASS | 8.49% FAIL |
| gemm_16_1024_64 | 3.74% PASS | 6.01% FAIL |
| gemm_512_1024_64 | 4.71% PASS | 0.00% PASS |
| layernorm_1_1024 | 2.61% PASS | 0.00% PASS |
| conv_4_16_32_64_k1_s2_p0 | 2.57% PASS | 5.89% FAIL |
| gemm_128_256_64 | 2.28% PASS | 1.05% PASS |
| gemm_16_64_1024 | 4.60% PASS | 6.46% FAIL |
| conv_1_16_32_64_k1_s1_p0 | 2.55% PASS | 4.72% PASS |

結果為 **G0 9/9 PASS，GE 4/9 PASS，合計 13/18 PASS**。若只看這個診斷樣本，GE 波動比 G0 更容易重現；仍須避免把這些專門挑選的配置當成全體 139 配置的無偏估計。

## 5. 解釋、限制與資料使用政策

1. 重新收集顯著改善了 RTX 3090 的整體時間穩定性，且全部新配置/模式聚合都滿足 100 ms 最低 group window；但尚未達成 **278/278** 的完整穩定性門檻。
2. Single-call 與 Batch Event 的數值和執行語意不同。研究模型至少應明確區分 `isolated_device_event_us` 與 `batch_event_us_per_call`；G0 host completion time 亦不可與它們直接混用。
3. 新資料 **417/417 P/R recorded sequence alignment** 並不代表 NVTX 事件完整。全部 417 個 warning-bearing trace pair 仍為 **manual_review_required_event_loss**，不得宣稱具有 verified reference labels。
4. 可將新資料指定為主要 O1 RTX3090 收集資料，但應在分析中以 configuration/mode 為單位保留 stability flag。不得只重跑失敗配置並用新數字覆寫，使整體品質看起來人為提高。
5. 舊資料因高波動應標為 **superseded / high-variance**；為節省磁碟可在**完成新資料本機封存、遠端備份及 checksum 驗證之後**刪除舊 RTX 3090 的本機 O1 raw/analysis。此資料生命週期決策不代表新資料達到最初設定的全通過標準。
6. 本報告來自節點 console 執行結果及其摘要，未把完整原始 trace 提交 GitHub。需進行後續學術結論或重新分析時，應使用 raw `.nsys-rep`／SQLite、`call_samples.csv`、`groups.csv`、`run_inventory.csv`、`aggregate.csv`、`rp_alignment.csv` 與校準 manifest。

## 6. 封存與可追溯目錄

以下均為節點 `~/RTchen` 下的相對路徑：

**新 clean session（需要一起備份）：**

- `o1-runs/rtx3090_formal139_clean_20261030_001_bootstrap/`
- `o1-runs/rtx3090_formal139_clean_20261030_001_full/`
- `o1-analysis/rtx3090_formal139_clean_20261030_001_calibration.json`
- `o1-analysis/rtx3090_formal139_clean_20261030_001_analysis/`
- `o1-analysis/rtx3090_formal139_clean_20261030_001_collection.log`
- `o1-runs/rtx3090_critical9_diagnostic_001/`
- `o1-analysis/rtx3090_critical9_diagnostic_001/`

**不納入新資料、待遠端 checksum 驗證後可刪除的舊本機 O1 Formal：**

- `o1-runs/rtx3090_formal139_bootstrap_001/`
- `o1-runs/rtx3090_formal139_full_001/`
- `o1-analysis/rtx3090_formal139_calibration_001.json`
- `o1-analysis/rtx3090_formal139_full_001/`
- `o1-analysis/rtx3090_formal139_collection.log`

**注意：** RTX3090 O2、O1 舊 pilot、任何遠端封存資料均不在本次本機刪除範圍。

## 7. 建議後續工作

- 保留並使用全部新 O1 raw 資料，但在 modeling 報告中分別呈現全樣本／通過原定品質門檻的樣本結果。
- 若要研究無逐步同步的 workload，可另以 Batch Event 建模；不可宣稱此數值為原 isolated GE latency。
- 若再次改善量測法，請使用**新的 protocol／dataset version**，而不是修改既有 raw 與歷史 acceptance。
- 跨 GPU 比較應核對 O1 配置／輸入／code hash、各節點 Calibration provenance、時間 label 定義及品質標記。

---
本文件僅記錄 2026-10-30 RTX 3090 重收集與隨後補充診斷之實驗證據。資料封存與刪除的實際執行狀態，需依節點端 SHA-256 驗證日誌另行確認。
