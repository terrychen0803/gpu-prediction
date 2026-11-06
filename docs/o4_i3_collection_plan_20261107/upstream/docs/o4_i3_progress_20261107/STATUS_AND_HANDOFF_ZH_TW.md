# GPU-Prediction O4：2026-11-07 I3 進度、風險與交接狀態

本文件為**根據使用者提供的兩份本地 I3 Markdown 原文所整理的狀態摘要**；研究決策另外參照 GitHub 統一工程分支 [O4 節點狀態](../O4_NODE_STATUS.md) 與[收集原則](../O4_COLLECTION_POLICY_20261107.md)。來源檔：

- [REPORT_ZH_TW2.md（原文／不修改）](REPORT_ZH_TW2.md)
- [PORTABLE_RUNBOOK_ZH_TW.md（原文／不修改）](PORTABLE_RUNBOOK_ZH_TW.md)
- [I3→I5 規劃與務實資料政策（新撰）](ROADMAP_AND_PRAGMATIC_DATA_POLICY_ZH_TW.md)

**證據層級**：GitHub 保存的是使用者提供的本地實驗敘述、此處的整理與已有的 O4 節點報告；本次沒有連到 GB10 GPU 重新執行、獨立讀取 I3 原始 SQLite、補上資料包中 `analysis/`、`evidence/`、`source/`、`assets_A0/` 等。原文內的相對連結保留，不保證在 GitHub 可直接開啟。**將文件發布到 GitHub ≠ 將可執行 I3 工具、資產或 Raw Trace 全部發布。**

## 1. 四種不可混淆的狀態

| 狀態 | 本次 I3 成果 | 需要注意 |
|---|---|---|
| **實際 GPU 實測** | GB10 九種 A0 `_a`受控算子，選定27 GI/P/R Cells／81 measured；含來源追溯補跑共33 Cells／99 measured，無 GPU 失敗 | **RTX4090／RTX3090 沒有執行本次 I3 協定**；只測九個 `_a`，非十八個 a/b 全測 |
| **工程與重新解析完成** | Exact taxonomy／ownership、完整 R ATen 參數、Source snapshot、18 資產、split feasibility、Portable handoff；155 個測試、22 個 SQLite／319 個 audit 逐 byte 重播 | 本地回歸結果，不是 GitHub 新增測試 CI；也不能憑 replay 證明 Reference 真實完整 |
| **規劃或待執行** | `_b`配置 GPU、更多 signature／context、40×3、RTX I3、I4 新 recognizer/N0、I5 ITER、OOF、I7 confirmation | 不能把工程 Smoke 當完整 Train／Validation／Test |
| **Reference 品質** | 本批 18 份 P/R 均有 NVTX Warning；`strict_verified=0`、`formal_training_eligible=false` | 本次使用者決定可先做**有條件的探索性使用**，不是把歷史品質標記改為 PASS |

## 2. I3-0：在已有七 A 跨 GPU 資料上做的分析

**歷史資料重新解析（非新 GPU 執行）**

| 指標 | 實際報告 |
|---|---:|
| 三台 GPU、七 A 受控窗口 matched measured forwards | **2,520** |
| 觀測的 Reference Operation | **2,448,720** |
| P-only GPU Events | **2,031,600** |
| 已建立明確 interface 對應的 ATen Overloads | **70** |
| 已記錄、但未成功歸屬的 Qwen Events | **720** |
| 節點／工作／repeat／ATen 參數缺口項目 | **1,404** |
| Recorded Event 的重複 ownership／非連續 ownership | **0／0** |

**重要修正**：舊版 `upsample_bilinear2d` 曾因字串含 `linear` 被錯誤分類成 Linear；新 Exact Taxonomy 改成 bilinear resize。報告引用 UN-A 的 4090 repeat1 約 **17.55%** Recorded Busy-Union、3090 約 **13.72%**、GB10 約 **11.65%**；這是原始分類錯誤更正後的 family 分配觀察，**不是 GI 效能占比，也不是新訓練分類器的準確率**。歷史表未覆寫。

**候選分段長度**（按這批已記錄、非零 Reference ATen 的 Operation 數加權）：

| L | 非零 Op 覆蓋率 | 超界 Op |
|---:|---:|---:|
| 4 | 99.4856% | 8,880 |
| 8 | 99.9791% | 360 |
| 16 | 100% | 0 |
| 32 | 100% | 0 |

**判讀**：單一整體比例受 Qwen 大量短 Operation 稀釋。下一版 I4 將 L=4／8／16 列為候選消融，依每任務的高 Busy-Union 超界部分、blind segmentation 費用／錯誤比較，**尚未凍結 L=16**。

**歷史 O3 可重用程度**：舊 27D 特徵、float32 normalization、15 Detector＋15 N0 權重及 normalizer 的 hash 經報告核對；但七 A 三設備 21 個指定 P Iteration 中，**19 個有非 kernel 事件／缺失 legacy geometry**，另兩個不等於確定相容。新的 A0 Smoke 只有三筆 measured，無法提供舊 O3 完整窗口。因此**本次沒有執行凍結 O3 模型推論**；仍應保留 O1/O2 Raw、O3 Model／Normalizer 供 Legacy Baseline 或相容性研究，不應說「舊模型失敗」。

## 3. I3-1：GB10 九種受控 Operator A0 結果

**新受控協定**：`o4_i3_A0_smoke_v1_20261107`。每一種 `_a`配置獨立 GI/P/R 程序，各有 8 warmup＋1 transition＋3 measured。18 個固定資產候選＝九種 Family × a/b 兩個 batch 變化；**只有九個 a 完成 GPU Smoke**。38 個 Operand 檔經 SHA 核對，約 23 MiB；`assets_A0/ASSETS.json` SHA256：`02e4fbe40f617d571d42b55d3e343c8ea4272b6f26febefa16f0de8bc4594e69`。

| A0 Configuration | GB10 GI 中位數（µs） | GI 相對 L2（CPU FP32 parity） | 最大 Event／ATen |
|---|---:|---:|---:|
| conv_a | 45.088 | 0.0208% | 1 |
| gemm_a | 11.520 | 0.0212% | 2 |
| bmm_a | 11.392 | 0.0209% | 1 |
| linear_a | 12.864 | 0.0208% | 1 |
| batchnorm_eval_a | 19.488 | 0.0208% | 3 |
| relu_a | 6.112 | 0.0000% | 1 |
| add_a | 6.912 | 0.0225% | 1 |
| maxpool_a | 15.040 | 0.0000% | 1 |
| adaptive_average_pool_a | 9.056 | 0.0214% | 1 |

**解讀限制**：表格是每個受控配置單一程序／3 measured 的 GI 中位數，並非 DNN 內單個 Operator 的 P CTX 真值；不同 Shape 的 GI 不能直接作演算法快慢排名或跨設備結論。CPU FP32 數值 parity 固定門檻為 relative L2、scaled max 各 ≤2%；所有選定 Cells 通過，R 的 active TorchDispatch Output 也要求相同。

**27 vs 33 Cells 的差別**：本輪正式選用 9×3＝27 Cells／81 measured；因 Source Snapshot／資產 criteria 追溯，MaxPool 和 AdaptiveAveragePool 另外各補 GI/P/R 一輪，共多 6 Cells／18 measured。實際執行總量 33 Cells／99 measured、11 個 P/R pairs、22 captures；原批次保留、不按時間結果選最佳補跑。

**新 R 參數**：ATen 完整 schema（含 default、tensor stride/dtype/layout/device/storage offset、output metadata）僅存 evaluator；P 只輸出 numeric allowlist，報告檢查 112 個事件不含真實 op/name/shape/config ID。這建立比舊三類 Reference 更好的參數記錄，但不表示 Label 已完全 Verified。

**其他品質現況**：18 份選定 P/R（9組×2）皆為確認過的 `cuda-sw`、無 CUDA dropped／未知 blocker、unmatched Work API 為 0、P/R 已記錄事件結構一致；仍**全部有 NVTX Warning**。記錄到 **12** 筆 GPU Start-before-associated-API-entry、**2** 對 temporal overlap；沒有修改 raw clock 或宣稱 PDL 原因。設備／套件與程式變更的資訊由本地 source snapshot 保存，但此 GitHub 文件 PR 不含完整 snapshot。

## 4. 工程測試、未解問題與資料使用決策

- **測試**：I3 新測試16＋O4 既有85＋Legacy54＝**155 全部通過**（本地報告）；原 O4 分發141檔不變。22份 SQLite／11組 pairs 由獨立 process 重解析，**319個 audit 產物逐byte一致**，舊 inventory 190個 SHA 再次核對。
- **已知資料缺口**：720個 Qwen unknown owner；歷史 ATen 參數缺口1,404；某些長片段；NVTX Warning；少量 CPU/GPU 時序異常；未完成 PDL Attribute 查證。它們須記錄用途與可能影響，不能當零成本或強行回填。
- **工程適用 vs 嚴格 Ground Truth**：資料可先做 development inventory、API/Feature 測試、已知可用片段的 **探索性** classifier/N0 prototype 及單機 GI-baseline，但若必要 Events 已知缺失或標籤時間軸失效，不能把受影響 CTX label 放進訓練。新用途標記詳見[務實資料政策](ROADMAP_AND_PRAGMATIC_DATA_POLICY_ZH_TW.md)。所有舊 `strict_verified=0`／`formal_training_eligible=false` **保持不變**。
- **正式統計限制**：一個 Family 只有兩個 asset Signatures，GB10 只測其 a，尚不足以做有效的 train／validation／test 獨立 Group 分離；同 shape 重複 3／40 次或多台 GPU 重跑，不會增加不同 Signature 的數量。報告的 Split Feasibility 拒絕正式拆分。

## 5. 與 GitHub 統一工程主線怎麼對齊？

| 軸線 | 目前有的證據 | 本次應如何表示 |
|---|---|---|
| I2 六任務七A | [O4_NODE_STATUS.md](../O4_NODE_STATUS.md) 記錄 RTX4090 七A unified v2 **63/63 Cells、2,520 valid measured iterations**（操作者交付摘要，未對原raw獨立重算）；其他節點見各自報告 | **I2 的七 A 40×3 和這次 I3 的九種 A0 是不同協定與研究單位**，不可將兩者 counts 互相加總或當作本輪 RTX I3 已完成 |
| I3-0 | 三台 I2 七A既有資料重新解析 2,520 matched forwards | 原資料 re-analysis；非本輪新發射 GPU 工作 |
| I3-1 | GB10 九類 `_a` Smoke 27/27 選定 Cells | RTX4090／RTX3090 **本輪未收 I3** |
| I4 | 舊 O3 權重特徵 hash 可以核對，輸入相容性有缺口 | 新 Recognizer／N0 尚未訓練，舊權重未在新 Smoke 推論 |
| I5 | 主要方向仍 4090→GB10／3090 | 尚無新 T-ITER 訓練、OOF 或正式六任務 GI 預測 |
| I6/I7 | 將來跨六任務與確認集 | 未做新 I3 之外的正式驗證 |

**注意發布路徑**：本文件 PR 是從既有 GitHub `o4/unified-engineering-20261106` 分支起稿；該統一工程分支本身仍有未合併的 Draft PR。不要將本文件內容描述為目前 `main` 已具備全部工具。若最終要把文件送進 main，應先處理既有統一工程 PR 與本文件的合併先後。

## 6. 立即待辦與交接責任邊界

1. **GB10 本地**：備份 `source/`、`assets_A0/`、raw／source snapshot／evidence／analysis，核對完整 SHA；如需讓 RTX 自行從 GitHub clone 使用 I3，必須另開程式與資產交付流程，此文件 PR 不能取代。
2. **RTX4090／RTX3090**：各先驗證 A0 bytes → CPU tests → `A0_batchnorm_eval_a` Plan-only → GI/P/R 小型 Smoke。能跑的其他 A0 工作獨立追加，不必同步進度或等全部七 DNN A 都重收。
3. **跨設備研究**：按 4090→GB10 及 4090→3090 **分方向**取可比 Signature 交集。建構每個 Family 的新 configurations，再按獨立 Signature 分組；明確區分 SUT/P-only input 與 R/evaluator 真值。
4. **模型研究**：先保留 Legacy O3 Frozen Baseline，試新 feature/taxonomy 和小型 exploratory I4 Prototype。足夠 groups 才作 OOF；GI target 與原有 host／CTX 分開。不要宣稱使用者提供的兩份本地報告等於 I4/I5 已訓練完成。

> **本次完成判定**：I3-0／I3-1 的本地工程及 GB10 A0 Smoke 已完成；三台 I3 正式收集、有效 Split、I4／I5 新模型與 Strict Reference **尚未完成**。低門檻表示早一點使用已界定限制的資料開始做探索，不是對任何有缺口的事件或時間標籤無條件信任。