# 給分析節點 Codex 的研究交接報告

更新：2026-10-20。倉庫：https://github.com/terrychen0803/gpu-prediction 。既有程式基準提交：`7e2e715238e5595ffe0c68ad94c7db8c4b76a5bd`。後續文件以目前 checkout 的 commit 為準，開始工作時請記錄 `git rev-parse HEAD`。

這份報告整理使用者已確認的研究方向與目前進度，讓另一台電腦的本地 Codex 可以接手資料檢查、分析與後續實作。GB10／3090／4090 已用本倉庫的 [固定輸入](../inputs/README.md)各完成 36 個 E1 run，並各完成 E2 的 100 cell。原始量測不在 Git；E1 的 54 份 trace 事件完整性仍待審。E3 的新配置量測與 C0/C1 pilot 已實作，節點命令以 [E3 手冊](E3_NODE_RUNBOOK.md)為準；跨設備預測模型仍是計畫。

## 1. 可直接貼給另一台 Codex 的任務

```text
請接手這個 gpu-prediction 專案的本地資料分析。
先讀 AGENTS.md、README.md、docs/CODEX_HANDOFF.md、docs/NEXT_EXPERIMENTS.md，
再核對 inference_pilot/VALIDATION.md 與目前程式。

研究是 GB10／RTX 3090／RTX 4090／RTX 5090 的跨設備 inference 服務時間預測。
四類任務為 ResNet50、YOLO26、Neural Receiver CNN、Qwen3-0.6B。
主方法只在來源設備做短 dry run，使用 Nsys trace 與事前硬體描述預測目標時間。
目前已實作 ResNet50 E1 量測／解析／彙整、E2 硬體校正工具、E3 新配置入口與 C0/C1 pilot；跨設備預測模型尚未完成。

請先檢查 checkout 狀態、程式版本、現有 session 資料及環境。
若已收到 session（可能放在 results/ 或外部 artifacts/runs/），先盤點與品質審查，再對有效資料執行 summarize／plot；
將輸出放到新的 analysis_runs/ 子目錄，不覆寫原始量測。
需要重新 parse 時，在工作副本執行並保留原始檔與 checksum。
以 docs/ANALYSIS_REPORT_TEMPLATE.md 建立本次分析報告，逐項標記通過、失敗或未測。
先判斷 E1 是否通過，再依後續計畫實作最小必要的校正／建模步驟。
若沒有資料，列出缺少的 session／檔案及下一個可執行指令，不生成假量測。

分析器可以看到兩台的 G 作評估；predictor 不可讀測試工作的目標 trace／時間。
所有設備、run、日期與裁切必須按配置分組切分；不能隨機切 iteration。
不要改寬驗收門檻、默默改 batch／精度，或把未知 Nsight schema 的欄位補成零。
請用繁體中文說明結果、證據、限制與下一步；缺少必要資料時提出具體問題。
```

若資料放在其他路徑，啟動任務時再提供實際資料根目錄。此文件不自動授權從其他節點下載私有檔案、修改遠端設定或發送訊息。

## 2. 使用者已確認的決策

| 項目 | 最新決定 | 已被取代或尚未確定的內容 |
|---|---|---|
| 目的 | 預測跨設備推論服務時間，協助 AI RAN edge 任務安排 | 排程效用尚未實測 |
| 場景 | Near-RT RIC／edge 應用 | 不宣稱已滿足 PHY 硬即時需求 |
| 設備 | GB10、RTX 3090、RTX 4090、RTX 5090 | 前三台已有新版固定輸入 E1；5090 暫不可用 |
| 執行種類 | 四任務全為 inference | 先前 training step、全參數微調的回答已被修正 |
| LLM | Qwen3-0.6B | 原始「Qwen 3.5 0.6B」已更正 |
| Receiver | 沒有指定原始論文／模型實作 | 需凍結可重現代理架構，不宣稱通訊效能 |
| 方法 | 來源短 dry run + Nsys trace + 設備描述／microbenchmark | 主模型不依賴框架 operator type 或 tensor shape |
| Wrapper | 研究階段用 NVTX markpoint 驗證 phase—API—kernel 對應 | 不能據此宣稱部署完全沒有 instrumentation |
| 優先順序 | 先完成可信量測與預測器，再做排程 | 多租戶、動態 batching、Graph／compile 為延伸 |

YOLO26n、四任務三組配置、Receiver 通道數等是目前研究提案中的起始設計；除了已實作的 ResNet50 外，仍需在建立 harness 時確認版本、形狀與計時範圍。不要把候選設定說成使用者已逐項確認的實作。

## 3. 討論如何收斂到目前方案

最初希望對三設備建立完整 counters 與 Compute／Memory／Launch 數學模型。參考 Habitat 後，關注它對不同 operation 的跨設備預測。接著把問題收斂為：能否減少對 PyTorch operator 參數的依賴，直接使用 GPU 執行行為表示工作負載。

因此目前主線是 Nsys trace。GPU Metrics 是可選增強資料；Nsight Compute 或其他 per-kernel counters 是另一個資訊條件的比較組。尚未確認四設備共同的 metrics 清單，不能從舊討論抄一張 counter 表就當成實際可取得欄位。

Wrapper 與這條主線相容：它先建立研究用 reference，確認 host phase 提交了哪些 kernel。部署模型的輸入再排除 phase 名稱、operator 類型與 shape，讓實驗回答「移除框架語意後會損失多少精度」。六個 ResNet50 phase 並非六個 operators，也不能從模型名稱推 kernel 數。

Habitat 的 operation 縮放提供方法啟發，但目前沒有重現其完整 wave scaling 或 MLP。Nsys trace 無法單獨保證精確 FLOPs、DRAM bytes、occupancy 或 target 的 kernel 實作。相同名稱也不能證明跨架構實作不變；target trace 僅可在評估後用來診斷誤差。

## 4. 現在到底完成了什麼

| 功能 | 狀態 | 證據／入口 |
|---|---|---|
| ResNet50 FP16 eager，B1/8/32，224² | 已實作；前三台固定輸入各 36 run，5090 未測 | benchmark.py；原始 session 須另交接 |
| G／A／P0／P1，六 phase，獨立程序、順序隨機化 | 已實作 | runner.py、__main__.py |
| Warm-up、正確性、checksum、環境與計時保存 | 已實作 | benchmark.py、common.py |
| Nsight profile／SQLite export／保守解析 | 已實作；GB10 的 Nsight 2026.4.1 schema 可解析，事件完整性待審 | trace.py；本版程式 hash 見 GB10 固定輸入摘要 |
| CSV 彙整、四模式成本、設備倍率、五類診斷圖 | 已實作 | report.py、plots.py |
| 合成 SQLite／防洩漏／圖表／固定輸入測試 | 目前 26 項通過、1 項可選圖表測試跳過 | tests/、VALIDATION.md |
| CLI 僅列計畫 | 已驗證 12 個唯一 run | suite --dry-run |
| GB10／3090／4090／5090 實測 | 前三台固定輸入各 36 run；5090 未測 | [GB10 固定輸入摘要](GB10_E1_FIXED_INPUT_20261020.md)；三設備整合在本機 analysis_runs；原始 session 不在 Git |
| Compute／Memory／Launch／Host microbenchmark | [E2 工具](../e2_calibration/README.md)已實作；GB10／3090／4090 各 100 cell 已收、品質與工具鏈差異見本機整合報告；5090 待收 | NEXT_EXPERIMENTS.md E2 |
| E3 單設備校正 pilot | [E3 工具](../e3_calibration/README.md)與 [節點手冊](E3_NODE_RUNBOOK.md)已實作；舊三配置只供流程檢查，新 12 配置矩陣待收齊／審查 | NEXT_EXPERIMENTS.md E3 |
| 跨設備學習、盲週期切分 | 待實作 | E4～E5 |
| YOLO、Receiver、Qwen harness／排程器 | 待實作 | E5～E6 |

GB10 固定輸入實測使用 Linux aarch64、PyTorch 2.13.0+cu130、TorchVision 0.28.0+cu130、Nsight 2026.4.1。18 個 trace 的 parser 通過，但均有「Not all CUDA events might have been collected」警告；9 個 P1 另有 NVTX 同類警告，故 E1 trace 完整性仍待審。請保留結構檢查與事件完整性兩種證據的區別。

## 5. 檔案導覽與文件優先順序

- [README](../README.md)：目前可執行的安裝、量測、解析與分析命令。
- [第一步實驗計畫](../第一步實驗_ResNet50量測與Trace對應驗證.md)：E1 的目標、計時定義與驗收。
- [後續實驗計畫](NEXT_EXPERIMENTS.md)：新的分階段研究設計、模型、資料切分與計數。
- [驗證狀態](../inference_pilot/VALIDATION.md)：已有證據與硬體未驗證項目。
- [分析報告範本](ANALYSIS_REPORT_TEMPLATE.md)：接手後保存結論的格式。

原電腦尚有早期訓練方案、三模式 Nsys 方案、閱讀清單、PPTX 與貼文。此倉庫不依賴那些本地絕對路徑；這份報告已整合相關決策。若收到舊文件，先看使用者最新修正，再看目前協定與程式；舊方案的 20 次 warm-up、30 次量測或 G/P/PM 計數不適用當前 E1。文件與程式若衝突，記錄問題並修正，不能默默選較寬鬆版本。

## 6. 接手當天如何開始

在分析電腦上：

```sh
git clone https://github.com/terrychen0803/gpu-prediction.git
cd gpu-prediction
git status --short --branch
git rev-parse HEAD
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
```

若已有 checkout，先檢查未提交修改再 `git pull --ff-only`；不要用 reset 清掉別人的工作。只做 SQLite 解析／統計／畫圖的電腦不需要 GPU 或 PyTorch；標準函式庫可做解析／統計，Matplotlib 提供圖表。執行目前 checkout 的全部測試並記錄通過、失敗與 skip 數；目前為 20 項。

量測節點才需要 CUDA 相容的 PyTorch／TorchVision 與 Nsight Systems；依 README 先 doctor、prepare，再跑 batch 1 G／P1。doctor 在沒有 Nsight 時可能非零，即使 G/A 可用；純分析機不以 doctor 的 GPU ready 當分析准入條件。

完整 session 建議手動放在 `results/` 下，保持原 device/batch/mode/run 目錄。GB10 固定輸入原始 session 在量測節點的外部 `/home/good_egg/RTchen/artifacts/runs/gpu_prediction_gb10_fixed_input_full_20261020/`，不會隨 Git clone 出現；交接者必須另外提供實際 session 根目錄、SHA-256 清單、量測程式 hash／commit 與分析報告。請用 manifest 的 `code_sha256`、`input_sha256`、`input_file_sha256` 核對。不要只搬 summary.csv；重新驗證需要 manifest、environment、quality、execution、latency、warmup，profile run 還需要 trace 與 parser 輸出、log。結果與原始 trace 不納入 Git。

```sh
python -m inference_pilot summarize results/4090_session02 results/5090_session02 --out analysis_runs/session02_review
python -m inference_pilot plot results/4090_session02 results/5090_session02 --out analysis_runs/session02_review
```

上述是範例路徑，只有實際存在相應 session 才能執行。plot 會重跑 summarize 並寫入同名彙整檔，因此在新的 analysis_runs 目錄執行；不修改原始 session。若要保留前次分析版本，用另一個輸出目錄。

重新 parse 會在 run 目錄寫出衍生 CSV／JSON，請先複製該 run 到分析工作副本：

```sh
python -m inference_pilot parse analysis_runs/reparse_work/rtx4090/b01/P1/run01
```

不要在交接原始資料上直接重解析。先保存原檔的 SHA-256 清單與來源路徑，在報告記錄 parser commit；不要修改原 manifest 的測量程式 code_sha256。不同 parser 產物需保留分析版本資訊。

## 7. 接手資料的品質檢查順序

1. 列出全部 session/run，包括失敗與 planned_only。保存 inventory：來源路徑、label、實際 GPU 名稱／UUID、batch、mode、repeat、軟體版本、protocol、狀態與缺檔。
2. 確認 execution=completed、quality=passed、量測數與 manifest 一致。P0/P1 還要 trace_quality=passed；parser 通過仍保留 manual_review_required。
3. 人工閱讀 profile.log、export.log、trace_diagnostics.json 與 Nsight report 的事件遺失診斷。沒有人工確認時狀態寫「待審」，不要把它改成已通過。
4. 檢查原始資料是否重複、session 是否被複製、輸入／權重 checksum 是否相同、裝置標籤是否正確；同一 run 不能計兩次獨立實驗。
5. 檢查程式 hash、完整 settings、torch／torchvision／CUDA／cuDNN、精度政策與計時協定。現有 summarizer 對軟體差異分 protocol，無倍率時先查這些條件。
6. 先看 G 的 warm-up、穩態與 run 間漂移，再看 A/P0/P1 成本、phase 覆蓋率與序列。記錄失敗原因，不能只挑好看的 run。
7. 三個合格且穩定的 G runs 才能產生當前設備倍率。最後才檢查跨設備倍率及 leave-one-batch-out sanity。

目前程式不會自動保證人工事件遺失審查、跨日期去重、工作負載版本登錄或完整防洩漏 pipeline。這些需分析者額外確認並在報告中留下證據。

`code_sha256` 目前按 Python 檔案原始 bytes 計算；跨作業系統的 LF／CRLF 換行也可能造成不同 hash。若相同 commit 卻分成不同 protocol，先核對實際檔案與換行政策，不能直接移除 hash 檢查；未來修改 hash 定義需版本化並保留舊量測的原值。

## 8. 資料契約與易誤解欄位

| 檔案／欄位 | 真正意義 | 使用限制 |
|---|---|---|
| latency.csv / latency_us | host forward + 最後 device synchronize 的服務時間，含模式所造成成本 | G 作目標；P 時間作校正評估，嚴格 trace 模型不直接讀 |
| warmup.csv | 同模式的預熱樣本 | 不混進穩態標籤 |
| features.csv / iteration | run 內量測編號 | ID，不是 workload 特徵 |
| boundary_method | 分段方法文字 | 方法紀錄，不能當數值物理特徵 |
| gpu_span_us | 首 kernel 開始至末 kernel 結束 | 不等於 host S0 |
| kernel_busy_union_us | GPU kernel 區間聯集 | 不把重疊 kernel 重算；不是全設備利用率 |
| kernel_gap_fraction | 該請求 span 內沒有其 kernel 執行的比例 | 不等於整張 GPU idle，也不能直接判定 memory-bound |
| launch_count | 有 kernel 對應的唯一 API 數 | 不包含所有 CUDA API／純同步呼叫 |
| grid_blocks_p50 / block_threads_p50 | kernel 的 grid block 數／block thread 數中位數 | 缺欄位為 missing；不是 occupancy 或 wave 數 |
| phase_mapping.csv | kernel—API—phase 診斷，含 kernel 名稱與時間 | 主特徵需 allowlist，不能整張直接送學習器 |
| reference_windows.csv / phase_ranges.csv | 研究用分段與 NVTX 範圍 | 僅 oracle／評估用途；盲分段不得讀 |
| device_ratios.csv | 同協定的目標 G／來源 G | 是分析標籤；測試值不能給 predictor |

主數值特徵現有：kernel_count、kernel_duration_sum_us、kernel_duration_p10_us/p50_us/p90_us、kernel_busy_union_us、gpu_span_us、kernel_gap_fraction、short_kernel_fraction_lt10us、stream_count、grid_blocks_p50、block_threads_p50、launch_count、memcpy_count、memcpy_bytes。缺值保留未知，不填 0；無 memcpy 的合法 0 和缺少欄位是不同情況。

schema_version=1 是現有 manifest 版本，不等於 Nsight SQLite 固定版本。parser 對 process/thread 的 global ID 編碼有假設，且只支援單 worker／device／context。若讀不到主執行緒、correlation 模糊或欄位不符，先檢查真實 schema 和 Nsight 版本；不要用最近時間猜對應來強行通過。

來源與目標的同配置可能有不同 kernel 數、tiling 或 fusion。現階段主要預測整個請求。主模型的特徵不含 target 工作的 trace；事後診斷可以使用，但要在報告標明其角色。

## 9. 建模與分析的交付要求

先以 [分析範本](ANALYSIS_REPORT_TEMPLATE.md) 交付 E1 結果。若資料只有 B1/8/32，不直接報正式泛化能力；E2 三設備已收，下一步是依 [E3 手冊](E3_NODE_RUNBOOK.md)補 ResNet50 配置網格、審查 trace 品質，再以 [E3 pilot](../e3_calibration/README.md)驗證校正流程。C2 與跨設備模型尚未實作。

新增建模時至少保存：`dataset_manifest`、`split_manifest`、特徵 allowlist、校正器／模型設定、seed、資料／程式 hash、逐配置方向的 predictions、排除表、總結報告。這些是後續交付規格，不是目前命令已輸出的檔案。

每條 prediction 建議包含 configuration_id、source/target、outer_fold、模型版本、predictor 使用的來源資料 ID、預測值與狀態。target actual 由 evaluator 另行 join，失敗預測也留一列。報告按配置分組計算 MdAPE／MAPE／P90 APE／MAE 與 coverage，並附各方向結果。

只在訓練 fold fit scaler／字典／校正器／模型；校正器使用 OOF 來源估計接到跨設備模型。相同配置、各設備、裁切及雙向配對不能跨 train/test；hardware microbenchmark 是預先允許的校正資料，不代表可以看留出 GPU 的 AI trace。

後續優先做小型 ridge／硬體縮放模型；三種硬體與少量配置不適合先上大型 MLP。效能模型的可解釋係數應稱縮放敏感度，不能因為命名為 compute/memory 就稱為已測得真實瓶頸比例。

## 10. 常見停點與下一步

| 情況 | 應做的事 |
|---|---|
| 沒有 session 或只有 dry-run 計畫 | 報資料尚未收集，提供 README 的 smoke／suite 命令 |
| 只有一台／一個 G run | 先做可支持的品質分析，列出缺少的設備或 repeats |
| G 可跑、P1 失敗 | 保留 G；檢查 Nsight CLI、capture、export、schema、correlation |
| trace_quality passed 但有 warning | 保留待審狀態，人工確認原始 report 與 log |
| 兩台沒有倍率輸出 | 查 protocol、checksum、repeats、G 穩定性；不刪除檢查 |
| GB10 與桌機套件 build 不同 | 建立不同 protocol，提出跨軟體比較設計或可行環境方案 |
| 欄位／kernel 路徑隨版本變化 | 建立版本適配與有意義的 regression fixture，再重分析副本 |
| 未知 shape／counter 取不到 | 先用實際 trace 共同欄位；額外來源資料單列模型 |
| 結果沒有超越固定倍率 | 如實報告，檢查量測、資料量與表示，不隱藏基線 |

分析完成後請交付：有效與失敗資料盤點、圖表及可追溯 CSV、品質結論、可支持與不可支持的研究主張、下一步最小新增實驗與所需程式。真實 GPU 結果與合成測試必須清楚分開。
