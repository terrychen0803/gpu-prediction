# 目標與現有專案一致性審查

此文件記錄 O1 實作前的審查時點。其後 GB10 已有 [O1 pilot 程式與受限結果](O1_GB10_PILOT_STATUS_20261021.md)；以下「尚未實作」敘述應按原審查日期閱讀。

日期：2026-10-21。審查的遠端基準：`73478bfd32a486246a2aecf3c8e59e0f3a839972`（main）。本次工作為閱讀程式、文件與設計整合；未在此電腦實作新 GPU 工具、重跑量測或驗證節點原始資料。

## 1. 使用者最新確認的目標

從來源設備 inference 的 trace／metrics，自動找出算子邊界與類型，建立算子層級的跨設備時間模型，再組合完整 inference 服務時間。要量化它與 wrapper 提供真實邊界／類型／shape 的方法之間的預測差距。不是只用整次 ResNet50 的統計特徵做迴歸，也不是只做 operation-aware 與 trace 模型的排行榜。

依使用者最新補充，本輪實作與量測設備為 RTX 3090、RTX 4090、GB10；本電腦交付設計、審查與交接文件。5090 留作未來擴充。四個完整模型仍是長期驗證方向，但優先補算子辨識與建模；先用受控算子與可涵蓋的子圖，逐步回到完整 ResNet50、其他 CNN 與 Qwen3-0.6B。

Wrapper 可在訓練／評估端產生標準答案。正式 trace 模型不得接收真實算子邊界、類型、shape、模型名稱、工作配置或測試目標時間。從允許的來源 trace 推估出的類型則可供時間模型使用。算子辨識與時間預測是否達標，都需實驗回答。

## 2. 證據範圍：Git、節點報告與使用者回報

| 內容 | 本次能確認的事 | 證據限制 |
|---|---|---|
| E1 | 有固定輸入、四模式、六 phase、計時、解析與彙整程式 | Git 文件記錄三設備 108 runs；原始 session 不在倉庫，未獨立重算 |
| E2 | 有 compute/memory/launch/host 校正程式，含受控 torch.mm | Git 文件記錄三設備各 100 cell；完整結果位於節點 |
| E3 | 有 12 配置量測入口及設備內 C0/C1 校正程式 | 文件記錄 GB10 72 runs；其他節點更新可能尚未同步 |
| E4 | 使用者提供節點已完成的程式／結果摘要 | 本次 Git main 沒有 e4_prediction/ 或 20261021 E4 報告，不能從此 checkout 驗證 |

使用者回報的六方向平均 MdAPE 為固定倍率 42.52%、硬體模型 37.50%、trace＋硬體模型 16.70%。本文件只記錄其來源為使用者貼上的節點摘要，並未重現；它也不是 operation-aware 對照結果。「六方向平均 MdAPE」與所有樣本合併的 MdAPE 不同，後續需保留原聚合方法。

節點接手先保存 E4 的實際 commit／未提交 patch、特徵 allowlist、split manifest、逐配置預測表、時間單位與品質標記。不能因 Git 尚無檔案就重建或覆蓋已做的 E4，也不能把舊文件的「未實作」當成節點的即時狀態。

## 3. 審查結果與修改方向

| 項目與程式證據 | 與新目標的關係 | 處理 |
|---|---|---|
| [benchmark.py](../inference_pilot/benchmark.py) 的 phased_forward 只有 stem/layer1～4/head | 大階段不是單一算子；不能訓練所需的算子辨識 | 保留 E1；新收集器建立明確且不重複計數的算子 taxonomy |
| [trace.py](../inference_pilot/trace.py) 的 P0 依 cudaDeviceSynchronize 和 manifest 計數分段 | 它辨識請求，不辨識請求內算子；屬同步輔助 | 原 parser 作回歸基線；新增與 manifest 算子資訊隔離的事件匯出與分段器 |
| 同檔 P1 要求恰好六個 PHASES 並讀 NVTX 真實名稱 | 不能直接擴寫 phase 清單就稱為無 wrapper 辨識 | 新建旁路 reference／predictor 資料格式，不改寫既有量測契約 |
| [E3 程式](../e3_calibration/__main__.py) FEATURES 是五個整請求統計 | 可驗證來源校正，不能恢復被聚合掉的次序和算子邊界 | 新主表示保留 event order、API–kernel 關聯與必要事件粒度 |
| E3 fit_c1／evaluate 按完整配置留出 | 方向正確，但只證明設備內來源時間校正 | 新管線連辨識器、字典、型別 gating、時間模型一起按配置群組切分與 cross-fit |
| [E2 make_operation](../e2_calibration/__main__.py) 的 torch.mm | 是硬體校正工作；沒有 BMM／Conv2D 辨識資料集和資訊條件對照 | 重用硬體 profile；另建算子工作負載資料，不把每個 repetition 當新配置 |
| E1/E3 主標籤為 host forward + synchronize | 適合最終平台服務時間；與單算子 GPU event elapsed 不同 | 新算子協定分別保存 event-span 與 host 服務時間，禁止混算 |
| [E1/E2 現況](E1_E2_STATUS_20261020.md)、[VALIDATION](../inference_pilot/VALIDATION.md) 記錄 CUDA/NVTX 可能未收齊警告 | 缺事件直接影響邊界與類型辨識；parser 通過不足以解決 | O0 先診斷與新 trace 稽核；未解除的資料只列探索組，不能支撐主要辨識主張 |
| [report.protocol](../inference_pilot/report.py) 核對程式、輸入協定、框架等 | 已有良好一致性檢查，但 signature 未涵蓋全部 Nsight 版本／trace backend 與 E2 編譯差異 | 新 contract 明列 collector、backend、toolchain；舊 run 不回填 |
| E2 文件記錄 4090 nvcc 12.8、GB10/3090 13.0 | 硬體效應與軟體效應可能混合 | 保留 profile 的來源；不能把異 build 結果說成純硬體比較 |
| 舊 NEXT_EXPERIMENTS 與 HANDOFF 有狀態／配置數不一致 | 會讓節點繼續沿錯誤優先順序執行 | 保存歷史版本；現行入口改為 O 系列。歷史 runbook 保留執行用途 |

結論：E1～E3 的量測基礎與新目標相容，但還沒有形成「從軌跡辨識算子，再逐算子跨設備預測」的證據鏈。這是研究單位與功能缺口，不能以完整模型回歸誤差來替代。使用者回報的 E4 應保留為整請求基線。

## 4. 可行性與目前不能承諾的能力

| 能力 | 可行性判斷 | 必須通過的證據 |
|---|---|---|
| 受控 eager GEMM/BMM/Conv2D 的事件與標籤對應 | 可實作；Nsight／框架版本需節點確認 | O0/O1 正確性、API correlation、有效事件完整性 |
| 已知邊界下，以 trace 建立跨設備時間模型 | 可實作，精度未知 | 與相同資料、相同時間目標的 wrapper 基線比較 |
| 未知邊界與類型的封閉集合辨識 | 有條件可行，需研究 | 混合序列留出、同類連續呼叫、多 kernel 算子、混淆案例與未知類別 |
| 從任意 trace 唯一恢復原始 Python operation 與 shape | 不能保證，一般情況不可識別 | 同底層執行可來自不同語意；需允許 ambiguous/unknown |
| 對短算子提供可靠 Nsys metrics | 尚未確認，可能無法完成 | 三節點指標盤點、採樣覆蓋與時鐘對齊；不可插值造出每算子 counter |
| 融合與多 stream 的完整 DNN 組合 | 明顯比 isolated 困難 | 先單 stream eager；融合與多 stream 另立協定與執行群定義 |
| 三設備的已知設備轉移與未見設備測試 | 可設計六方向及留一設備測試；不能保證廣泛泛化 | 留出設備的工作負載時間完全不參與 fit；只允許事先獨立硬體校正，三種 GPU 的證據仍有限 |

例如 BMM 的 batch=1 可與 GEMM 使用相同底層路徑；某些 Conv2D 會降低成矩陣乘法。若允許特徵相同而語意不同，分類器無法憑空恢復資訊。正確行為是回報候選類型／拒絕，而非用 oracle 標籤強制選一類。可部署的 kernel-name 提示允許使用，但要和去名稱模型分開報告。

## 5. 本次整合的界線

本次新增／更新研究設計、資料契約、節點交接、報告範本與文件導覽；不修改 E1～E3 Python/CUDA 程式、輸入、設定或凍結 hash。新 O 系列尚未實作，文件中的檔案格式與模組名稱是節點實作規格，不是現有命令。

具體下一步依 [現行實驗設計](NEXT_EXPERIMENTS.md) 與 [節點交接](CODEX_HANDOFF.md)。先確認節點已有 E4 工作與 O0 的可觀測性，再決定是否值得收集大矩陣。任一 gate 未通過時縮小主張或保留未完成，不用改門檻把失敗變成通過。

## 6. 本次文件檢查

已核對修改範圍只有 Markdown、相對檔案連結存在、code fence 成對、九配置／115／139 配置及三設備量測單元計數。`git diff --check` 無空白格式錯誤。未改 Python/CUDA、固定輸入或設定，故未重跑 GPU／程式測試；這些文件檢查不表示新 O 協定已在節點通過。外部方法依原論文及 NVIDIA 官方文件核對，連結列於現行設計。
