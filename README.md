# GPU 跨設備 inference 預測研究

## 目前主線：由 trace／metrics 辨識算子並預測時間

更新：2026-10-21。研究設備為 **RTX 3090、RTX 4090、GB10**。目標是從來源 GPU 的 dry-run trace／可取得 metrics，自動找出算子邊界、辨識類型，再建立逐算子跨設備時間模型並組合 inference 服務時間；比較其與 wrapper 已知邊界／類型／shape 的預測差距。

**O1 九配置 pilot 已在 GB10 實作並部分實測；O2～O5 仍是後續工作。** O1 的九配置 G0/GE 時間量測完成，P/R 三家族 trace 診斷仍有事件完整性警告，不能當正式逐算子 P 標籤。程式與跨節點命令見 [O1 節點手冊](docs/O1_NODE_RUNBOOK.md)，數字與限制見 [GB10 O1 摘要](docs/O1_GB10_PILOT_STATUS_20261021.md)。既有 E1～E3 程式與凍結 hash 保留；使用者回報 E4 已在節點完成，目前 Git 尚未收錄其程式／報告，請先保護當地成果。

建議閱讀順序：

1. [目標與現有專案一致性審查](docs/GOAL_ALIGNMENT_REVIEW_20261021.md)
2. [O0～O5 完整實驗設計](docs/NEXT_EXPERIMENTS.md)
3. [資料契約與資訊隔離](docs/OPERATOR_DATA_CONTRACT.md)
4. [三節點 Codex 交接與第一輪實作](docs/CODEX_HANDOFF.md)
5. [O 系列報告範本](docs/OPERATOR_REPORT_TEMPLATE.md)

模型已指定以 **MLP 為主**。可實作的公式、特徵、MLP-S/F/V 架構、分段動態規劃、損失函數與分組訓練順序，見 [T-v1 MLP 數學規格](docs/T_V1_MLP_MODEL_SPEC.md)。這是新增設計，尚無新模型程式或訓練成績。

第一步是三設備 O0 可觀測性盤點及 GEMM/BMM/Conv2D 九配置 pilot。必須驗證真正無標記 trace 的邊界標籤能否可靠取得；算子辨識與 metrics 可用性均尚未證實，不能只憑既有整模型誤差推論。

## 既有 E1～E3 工具與操作說明

以下為已實作工具，保留用於舊協定重現及校正；其中四設備擴充、E 系列「下一步」是歷史規劃，現行優先順序以上述 O 系列為準。不要把以下 six-phase 的 phase 當成單一 operation。

文件入口：[第一步實驗計畫](第一步實驗_ResNet50量測與Trace對應驗證.md) · [共用輸入與 checksum](inputs/README.md) · [三設備 E1／E2 現況](docs/E1_E2_STATUS_20261020.md) · [GB10 固定輸入 E1](docs/GB10_E1_FIXED_INPUT_20261020.md) · [後續實驗與建模計畫](docs/NEXT_EXPERIMENTS.md) · [3090 節點操作](docs/RTX3090_NODE.md) · [分析節點 Codex 交接報告](docs/CODEX_HANDOFF.md) · [分析報告範本](docs/ANALYSIS_REPORT_TEMPLATE.md)。

現在 Git 內可執行的是 ResNet50 E1 資料流程、[E2 硬體校正工具](e2_calibration/README.md)、[E3 來源時間校正 pilot](e3_calibration/README.md)，以及 [O1 九配置 pilot](o1_operator/README.md)。重現 E3 配置請依 [E3 節點手冊](docs/E3_NODE_RUNBOOK.md)；節點 E4 成果與新 O 系列的狀態見上方說明。

這套程式實作四種量測模式、六個 forward 階段標記、穩態檢查、Nsight SQLite 解析、跨設備統計與診斷圖。研究用時間標籤需來自真實 CUDA 推論；主指標以 host timer 量測 forward 到 GPU 完成的服務時間，GPU trace 統計另外保存。缺少 GPU 或必要工具時會停止，不使用 CPU 推論或模擬數字代替實測。

合成 SQLite 資料已驗證解析與統計邏輯。GB10、3090、4090 已各用同一固定 FP16 輸入完成 B1/8/32、四模式各三次，共 108 個正式 run；本機整合見 `/home/good_egg/RTchen/gpu-prediction/analysis_runs/three_gpu_fixed_input_e1_20261020/REPORT.md`。54 份 trace 的事件完整性警告仍待審；5090 暫不可用。原始資料與本機分析位於 `/home/good_egg/RTchen/`，不在 Git。新設備第一次執行仍應先使用 batch 1。

## 1　安裝環境

需要 Python 3.10 以上，以及支援該設備的 CUDA 版 PyTorch、對應版本的 TorchVision、Nsight Systems CLI。

先依 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/) 或 GB10 平台支援的發行方式安裝框架。各設備都需要與驅動、CUDA 架構相容的框架 build，不能只依賴 `pip install torch` 成功。四台設備盡量使用相同框架與 CUDA／cuDNN 版本，並保留實際版本資訊；本程式會將不同軟體設定分組，避免直接混合比較。

以下指令都在包含 `inference_pilot` 資料夾的專案根目錄執行。在 Linux 使用你的環境中的 `python` 或 `python3`；Windows 使用已安裝 CUDA 版 PyTorch 的 Python。

```sh
python -m inference_pilot doctor --out doctor.json
```

`ready_for_G_A` 代表可執行基本量測；`ready_for_P0_P1` 還需要找到 `nsys`。doctor 只檢查可取得的工具與 CUDA 狀態，真正的 kernel 相容性以 batch 1 試跑為準。若只缺 Nsight，doctor 回傳非零狀態，但 G／A 仍可單獨執行。

要產生圖表，再安裝：

```sh
python -m pip install -r requirements-analysis.txt
```

統計彙整、SQLite 解析與單元測試只需要 Python 標準函式庫。Nsight CLI 參數以當地 `nsys profile --help` 為準。Windows 的 CLI 可能需要管理員執行；Linux 需依當地 profiler 權限設定。[NVIDIA 官方說明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## 2　準備固定模型權重

```sh
python -m inference_pilot prepare --out assets/resnet50_imagenet1k_v2.pth
```

這一步會下載 TorchVision 的 `IMAGENET1K_V2` 權重，輸出 `.pth` 和 `.pth.json`。將這兩個檔案一起複製到其他設備，確保 checksum 一致。權重下載與模型載入都不在計時區間內；程式不會自動改用隨機權重。已有檔案時不覆寫。

## 3　第一次真正試跑

在 4090 上先完成一個 G 模式：

```sh
python -m inference_pilot run --label rtx4090 --expect-gpu 4090 --mode G --batch 1 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --out results/4090_g_smoke
```

檢查 `quality.json` 是否為 `passed`。若成功，再測帶標記的 profile：

```sh
python -m inference_pilot run --label rtx4090 --expect-gpu 4090 --mode P1 --batch 1 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/4090_p1_smoke
```

這會自動啟動 `nsys profile`、收集穩態 trace、匯出 SQLite 並執行 parser。`quality.json` 是 CUDA worker 的檢查；`trace_quality.json` 是 trace 解析檢查；`execution.json` 是整條流程是否完成。P1 的 trace 還需要檢查 `trace_diagnostics.json` 與 Nsight 介面的診斷資訊。

`--expect-gpu` 會核對實際裝置名稱，避免把其他 GPU 的結果誤標為 4090。多 GPU 系統可加 `--device cuda:1`；實際 CUDA index、GPU 名稱與 UUID 另外保存在環境檔。GB10 裝置名稱應先以 doctor 確認，再提供適當的名稱片段。

## 4　執行第一輪 12 runs

先查看將執行的工作，不下載、不量測：

```sh
python -m inference_pilot suite --label rtx4090 --expect-gpu 4090 --batches 1 --repeats 3 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/4090_plan --dry-run
```

`--dry-run` 是 CLI 的「僅列計畫」模式，與研究中的「在來源 GPU 實際執行短 dry run profiling」不同。其輸出標為 `planned_only`，不會進入結果彙整。

正式執行時使用新的輸出目錄：

```sh
python -m inference_pilot suite --label rtx4090 --expect-gpu 4090 --batches 1 --repeats 3 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/4090_b1_fixed_input_session01
```

預設四種模式各 3 個獨立程序，共 12 runs；每輪會打亂配置與模式順序。第一個失敗會停止，保留資料供排查。可用 `--keep-going` 收集其他 runs，但失敗資料不會被當成通過。

| 模式 | 量測內容 |
|---|---|
| G | 無 NVTX 標記、無 profiler 的完成時間 |
| A | 加入 iteration／六個 phase 標記，無 profiler |
| P0 | CUDA trace，無 iteration／phase 標記 |
| P1 | CUDA 與 NVTX trace，含 iteration／六個 phase 標記 |

所有模式使用同一段 six-phase forward 程式；與 TorchVision 原始 forward 的輸出對照在計時外完成。G 的時間因此是這個研究 harness 下的服務時間，包含該 forward 分組程式的小量 Python 成本。

## 5　擴到四台設備

3090 的安裝、權重校驗及 batch 1 試跑順序請先看 [3090 節點操作](docs/RTX3090_NODE.md)。

3090：

```sh
python -m inference_pilot suite --label rtx3090 --expect-gpu 3090 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/3090_fixed_input_session01
```

4090：

```sh
python -m inference_pilot suite --label rtx4090 --expect-gpu 4090 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/4090_fixed_input_session01
```

5090：

```sh
python -m inference_pilot suite --label rtx5090 --expect-gpu 5090 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --out results/5090_fixed_input_session01
```

GB10：

```sh
python -m inference_pilot suite --label gb10 --expect-gpu GB10 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/gb10_fixed_input_session02
```

預設 batches 為 1、8、32，每台 36 runs；四設備正式矩陣共 144 runs。權重檔位置不同時，加 `--weights /absolute/path/to/resnet50_imagenet1k_v2.pth`。工具不在 PATH 時，加 `--nsys /absolute/path/to/nsys`。GB10 使用 `--nsys-cuda-trace cuda-sw`；其他節點先依當地 Nsight 支援與 smoke test 決定後端，並在比較 trace 前核對後端差異。

每個 session 使用新目錄。程式拒絕覆寫已有 run；尚未實作自動 resume。某個 run 失敗後可使用 `run` 指令與新目錄重做，並保留原失敗記錄。

## 6　固定的量測協定

- 224 × 224、NCHW contiguous、模型及輸入 FP16、autocast 關閉。
- `eval()`、`inference_mode()`、eager、CUDA Graph 關閉、單一預設 stream。
- 正式 E1 使用 `--input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin`，在計時前載入同一份 checksum 驗證的 FP16 bytes。沒有此參數時程式仍保留 CPU 隨機生成路徑，但它不屬於本輪同輸入協定，結果必須分開。
- 一次請求結束前呼叫 device synchronize；計時含 forward、標記開關所造成的成本與最後等待。
- 不在 phase 之間同步。NVTX push／pop 在 host timer 內，方便量化標記開銷。
- 先做至少 50 次 warm-up，再觀察連續三個 20 次視窗；相鄰中位數變化都小於 5% 才量測。最多 500 次。
- 四種模式都先執行 5 次 transition，再收集至少 100 次、累計至少 1 秒的量測。量測迴圈 wall time 上限 30 秒。
- P0／P1 的 capture 在 warm-up 後啟動；transition 保留但不進主要特徵。
- 每個 run 前後保存 `nvidia-smi` 快照；它不提供完整的時間連續功率／溫度記錄。環境欄位缺失會保留原因。

設定集中於 `inference_pilot/settings.json`。若要修改，複製一份後使用 `--settings my_settings.json`。完整設定進入 manifest 與比較條件，不會把不同協定的結果直接混在一起。

### 固定跨設備輸入

本倉庫的 [inputs/](inputs/README.md) 已包含固定 `.bin` 與 `.bin.json`。這份 B32、NCHW contiguous、FP16 little-endian 檔案可取前 B1／B8 筆；執行時會核對完整檔案及所用 batch 前綴的 SHA-256。**3090／4090／5090 直接使用 Git 中的原檔，不要在當地重新執行 `prepare-input`**，因各架構的 CPU `torch.randn` 可能產生不同 bytes。只有設計另一個新協定時才使用：

```sh
python -m inference_pilot prepare-input --out /path/to/new_input.bin --max-batch 32 --seed 42
```

每個新節點先以 `run --mode G --batch 1` 與 `run --mode P1 --batch 1`，兩個命令都加入 `--input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin`；核對 manifest 中的 `input_sha256` 後，再以相同參數執行 `suite --batches 1 8 32 --repeats 3`。GB10 的正式 session 為 `/home/good_egg/RTchen/artifacts/runs/gpu_prediction_gb10_fixed_input_full_20261020`。先前以各機 CPU 隨機生成的 x86 session 不回填成新版協定。

## 7　Trace 對應與限制

P1 以 `pilot.iter/measure/000000` 等 range 建立真實標記邊界，再以同程序、同執行緒的 CUDA API 和 correlation ID 找 kernel。Kernel 可以晚於其 phase 的 CPU range 才執行；parser 不靠 kernel 名稱或 GPU 時間重疊猜測 phase。

P0 使用量測程式的每次 `cudaDeviceSynchronize` 分段，明確標為 `synchronization_assisted_not_blind`。同步次數必須等於 manifest 的 transition 加 measured 次數；多出或少掉都會停止，避免默默選錯 iteration。它尚未實作純週期的無同步邊界辨識。

目前 adapter 支援單一 worker、單一 CUDA device／context，以及標準 Nsight `CUPTI_ACTIVITY_KIND_RUNTIME/DRIVER`、kernel、`NVTX_EVENTS`、`StringIds` schema。會檢查欄位並保留 `trace_schema.json`；未知 schema、重複且無法唯一確認的 correlation、GPU 工作跨越完成邊界等情況會回報失敗，供後續加入版本適配。

六個 phase 是 stem、layer1、layer2、layer3、layer4、head。主 `features.csv` 不含 phase 名稱、batch、模型名稱或目標時間；研究標籤保存在另外的 CSV。Kernel 名稱只保留作診斷。Grid／block 缺少時留空，不用 0 代替未知值。

這版 parser 用 kernel duration 的聯集計算 busy time，避免把重疊 kernel 重複計算。`kernel_gap_fraction` 只代表該請求 kernel 未執行的比例，不代表設備完全閒置。Phase duration 加總也不是 host 服務時間分解。

自動解析通過不代表 trace 一定沒有事件遺失。`trace_quality.json` 會保留 `manual_review_required`，並收集 profiler 警告與 diagnostic table 供檢查。

重新解析已有 SQLite：

```sh
python -m inference_pilot parse results/4090_b1_session01/rtx4090/b01/P1/run01
```

## 8　輸出資料

| 檔案 | 內容 |
|---|---|
| schedule.json | Session 的隨機化順序 |
| request.json／command.json | 實際設定與命令列參數 |
| environment.json／manifest.json | 設備與套件、權重／輸入 checksum、程式 hash、計時協定 |
| warmup.csv／latency.csv | 每次 warm-up、transition 與測量時間 |
| quality.json／execution.json | Worker 與整體流程的完成狀態 |
| trace.nsys-rep／trace.sqlite | P0／P1 原始與匯出 trace |
| trace_schema.json／trace_quality.json | Schema 與解析檢查 |
| features.csv | 每次推論的 trace 數值特徵 |
| phase_mapping.csv／phase_summary.csv | Kernel、API、phase 對應與階段統計 |
| phase_ranges.csv／reference_windows.csv | 僅供研究與畫圖的 NVTX／同步邊界 |
| profile.log／export.log／trace_diagnostics.json | Profiler、export 與事件遺失診斷線索 |

任何失敗結果需保留原始檔。解析失敗後可能存在前一次解析產物，使用前務必以最新 `trace_quality.json` 狀態為準；彙整程式會檢查該狀態。

## 9　彙整與畫圖

將不同設備的完整 session 目錄複製到同一台分析主機，然後執行：

```sh
python -m inference_pilot summarize results/4090_session02 results/5090_session02 --out results/comparison
python -m inference_pilot plot results/4090_session02 results/5090_session02 --out results/comparison
```

彙整輸出：

- `summary.csv`：每個 run 的完成時間統計。
- `aggregate.csv`：各配置／模式的 run 中位數與穩定性。
- `overheads.csv`：A/G、P0/G、P1/A、P1/G 的差比例與絕對微秒差。
- `device_ratios.csv`：相同配置、相同協定的目標／來源 G 模式倍率。
- `leave_one_batch_out_sanity.csv`：以其他 batch 的倍率預測保留 batch；只作小樣本流程驗證。
- `excluded_runs.csv`／`summary_quality.json`：被排除資料與不足重複次數等條件。

設備倍率要求同一軟體／程式／量測協定與相同輸入 checksum，兩邊各至少 3 個 G runs，且 run 中位數的 `(max−min)/median ≤ 5%`。若條件不滿足，保留一般統計但不產生該倍率。兩台軟體版本不同時先核對版本；不要直接刪掉檢查來混合比較。

圖表包含 G 時間曲線、四種模式分布、P1 CPU NVTX／CUDA API／GPU kernel 時間線、phase duration 與數量、跨設備倍率。P1 圖只展示一個 run 的診斷，不代表無 profiler 的分解結果。

## 10　驗證與程式入口

```sh
python -m unittest discover -s tests -v
```

合成 trace 測試涵蓋非同步 NVTX 對應、跨程序 correlation 衝突、GPU 區間重疊、同步／NVTX 缺失、未知 schema、選填欄位缺少、排程不覆寫，以及保留配置的目標倍率不洩漏到預測輸入。這些測試不代表已在四台 GPU 上量測。

| 模組 | 作用 |
|---|---|
| inference_pilot/benchmark.py | ResNet50、計時、正確性與 warm-up |
| inference_pilot/runner.py | 獨立程序、隨機化順序、Nsight 啟動與匯出 |
| inference_pilot/trace.py | SQLite schema 檢查、API 關聯、特徵抽取 |
| inference_pilot/report.py | 量測成本、穩定性與跨設備倍率 |
| inference_pilot/plots.py | 診斷圖 |
| inference_pilot/__main__.py | CLI 入口 |

本版完成第一步資料流程，另有 [E2 Compute／Memory／Launch／Host 校正工具](e2_calibration/README.md)；學習式預測模型仍是後續工作。
