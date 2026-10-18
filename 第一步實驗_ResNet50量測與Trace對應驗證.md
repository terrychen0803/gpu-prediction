# 第一步實驗　ResNet50 量測與 Trace 對應驗證

狀態：實驗規劃，尚未執行 GPU 量測。

本實驗承接「以 GPU 執行軌跡進行跨設備推論時間預測」研究構想。第一個里程碑是取得可信任的完成時間與可解釋的 trace，讓後續建模使用的資料有可靠依據。本文的次數與門檻是事前設定的起始值，並非文獻保證或已達成結果。

## 1　這一步要回答的問題

1. 相同設定的推論完成時間，在重複執行下是否穩定？
2. wrapper／NVTX 標記能否正確連到這次推論實際提交的 GPU kernels？
3. 加入標記與開啟 Nsight Systems，各自改變多少時間或執行行為？
4. 同一個工作換到另一台設備後，整體與各階段的變化是否不同？

通過上述檢查後，才能合理研究「哪些來源 trace 特徵能解釋跨設備時間倍率」。本輪結果主要證明量測流程可行；只有三個配置，還不足以證明學習模型具有泛化能力。

## 2　最小起點與擴充順序

先在 RTX 4090 執行 ResNet50、batch size 1，完成一個配置的整條資料流程。通過檢查後，在 4090 與 5090 跑完下表；GB10 再使用同一份實驗程式與資料格式加入。

| 配置 | Batch size | 輸入形狀 | 精度 | 觀察目的 |
|---|---:|---|---|---|
| C1 | 1 | 1 × 3 × 224 × 224 | FP16 | 小工作量下的提交與短 kernel 行為 |
| C2 | 8 | 8 × 3 × 224 × 224 | FP16 | 工作量放大後的時間與序列變化 |
| C3 | 32 | 32 × 3 × 224 × 224 | FP16 | 較大 batch 的運算與資源使用行為 |

這些設定沿用既有方案。不能在某台設備遇到問題時默默改 batch 或精度；應記錄不可行原因，另定三台共同可行的配置。

固定條件：

- 使用 TorchVision 的 ResNet50，記錄 PyTorch／TorchVision 版本與程式 commit。
- 使用 `IMAGENET1K_V2` 權重，先下載並保存 checksum；模型載入和下載在量測範圍外。該名稱代表固定權重版本，比可隨版本調整的 DEFAULT 更清楚。[1]
- `model.eval()`、`torch.inference_mode()`、eager 執行、CUDA Graph 關閉、單一預設 stream。
- FP16 明確定義為模型浮點參數和輸入轉為 FP16，關閉 autocast；記錄可能存在的內部累加精度，不宣稱所有指令均為 FP16。
- 固定 NCHW contiguous layout，cuDNN benchmark 關閉；記錄 deterministic 與 TF32 政策，各設備一致。設定一致不代表跨架構會選到同一 kernel。
- 以固定 CPU seed 產生標準常態合成輸入，保存或記錄 checksum，再移到 GPU 並預熱。此處評估執行時間，不評估分類準確度。
- 一個 worker 同時只有一個尚未完成的請求；每個請求可包含一個 batch。
- 關閉其他實驗負載；記錄背景顯示負載、CPU 執行緒數、功率與時脈政策、溫度及軟體版本。

GB10 的 CPU 與記憶體環境可能與兩張桌上型 GPU 不同。包含 host 提交的結果應稱為設備／平台服務時間；GPU trace 時間另外報告，避免把主機差異全部歸因於 GPU。

## 3　時間標準答案怎麼定義

主指標是已預熱、輸入已駐留設備時，一個批次從開始呼叫 forward 到 GPU 工作完成的服務時間。

```text
確認上一個請求完成
t0 = 高解析度 host timer
執行一次 forward
等待這次 GPU 工作完成
t1 = 高解析度 host timer
保存 t1 - t0
```

模型載入、輸入生成、H2D、影像解碼和外部後處理均在此時間範圍外。每次請求使用相同同步政策；同步是此實驗的服務協定之一。不能只量 forward 在 Python 上返回的時間，因為 CUDA 工作通常非同步執行。[2]

同時保存兩種輔助 trace 統計，但不要混成同一種時間：

- GPU span：同一次請求中，第一個 kernel 開始到最後一個 kernel 完成的時間。
- Kernel busy time：該請求 kernel 執行區間聯集的長度。若 kernels 重疊，聯集長度與 duration 加總不同。

Host 服務時間與 GPU span 的差值包含多種因素，不可全部稱為純 launch overhead。各 phase 的 GPU span 也不保證可直接相加成整體服務時間。

## 4　四種模式分辨標記與 profiler 的影響

| 模式 | Nsight Systems | Iteration 與 phase 標記 | 用途 |
|---|---|---|---|
| G | 關閉 | 關閉 | 無 profiler 的時間標準答案 |
| A | 關閉 | 開啟 | 估計研究標記的影響 |
| P0 | CUDA trace | 關閉 | 接近未來部署的 trace 資料 |
| P1 | CUDA ＋ NVTX trace | 開啟 | 檢查 phase、API、kernel 的對應 |

四種模式使用同一份模型、輸入、同步政策與計時程式。標記開關應可獨立控制。P0 不開框架 operator／shape tracing；capture 起訖控制可以存在，但不能將其誤稱為已解決所有 iteration 邊界問題。

第一輪先收 CUDA trace 與必要 NVTX。GPU Metrics 在後續另加一組比較，以免取樣條件與基礎 trace 的問題混在一起。Nsight Systems 的 GPU Metrics 是設備層級取樣，不能直接視為每個短 kernel 的專屬 counters。[3]

令每模式各 run 的中位數再取中位數，得到 G、A、P0、P1：

```text
標記時間差比例          = A / G - 1
無標記 profiling 差比例 = P0 / G - 1
有標記 profiling 差比例 = P1 / A - 1
總差比例                = P1 / G - 1
```

另報上述差異的絕對微秒數。小工作負載的固定成本可能造成大的相對比例。這些是條件下的量測差異，可能包含交互作用，不能預設兩種開銷一定可線性相加或固定扣除。

## 5　wrapper 先標六個階段

以同一條 forward 執行路徑加入 range，涵蓋完整推論與六個連續階段：

```text
iteration_i
  stem       conv1 → bn1 → relu → maxpool
  layer1
  layer2
  layer3
  layer4
  head       avgpool → flatten → fc
```

六個 phase 是研究用的工作分組，不是六個 operators。ResNet50 的模型名稱也不代表會有 50 個 GPU kernels。某個 framework operation 可以提交多個 kernels，有些操作則不會產生獨立 kernel。

Iteration range 涵蓋 forward 與最後完成等待；phase range 只包住該段 host 提交程式，不在 phase 之間 synchronize。若為加入 range 而重寫 forward 順序，先比較有／無標記輸出、形狀與有限值，確認標記沒有漏掉 residual、flatten 或其他計算。

同設備、相同權重與輸入下，建議以 FP16 `rtol=1e-2, atol=1e-3` 作為起始對照容忍度，並保存最大絕對差與相對差；若不通過，先檢查實作與數值條件，不能只放寬容忍度。輸出檢查移到計時區間外。

對應步驟：

1. 在同一程序／CPU 執行緒中，找到某個 CUDA launch API 所屬的 NVTX phase。
2. 以匯出資料提供的 API–GPU correlation 關係，找到該 launch 提交的 kernel。
3. 使用程序、context 等必要欄位限定關聯範圍，不能假設 correlation ID 在所有資料中全域唯一。
4. 將每個 kernel 指派給唯一的最內層 phase；iteration 作為上層標籤，避免父子 range 重複加總。
5. 無法對應的 kernel 放入 unassigned 並檢查原因，不以最近時間或最近名稱強行補標。

NVTX 標記的是 host 程式範圍；kernel 可能在 range 結束後才開始執行，因此不能依 GPU 起訖時間是否落在 CPU range 內判斷 phase。[3]

## 6　每個 run 的執行方式

1. 獨立啟動程序，保存環境與設定；先完成一次不計時的正確性檢查。
2. 以 50 次 warm-up 為起點，用每 20 次的時間中位數觀察穩定性。連續三個視窗的相鄰中位數差均低於 5% 才進入量測；若累計 500 次仍不穩定，標記並排查背景負載或初始化，不直接納入。
3. 每 run 至少收集 100 次完整推論，且實際計時區間合計至少 1 秒。每次只保存必要數值到記憶體，全部完成後寫檔，不在迴圈內列印或檢查輸出。
4. 量測區間上限先設 30 秒；未達樣本條件時記錄為未完成並另查原因。
5. P0／P1 只 capture 穩態區間；啟動 capture 後先完成少量過渡請求，例如 5 次，另標為 transition，分析時只取後續完整請求。
6. 每配置的四種模式各做 3 個獨立 run；每輪隨機化模式順序並保存順序。檢查溫度與時脈是否隨順序漂移。

重複次數的計算：

- 單台 4090、C1 起步：4 模式 × 3 runs = 12 runs。
- 4090 與 5090、三配置：2 設備 × 3 配置 × 4 模式 × 3 runs = 72 runs，包含前述 12 runs。
- GB10 加入相同三配置：再增加 36 runs，總計 108 runs。

同一 run 內的 100 個 iteration 是重複量測，不是 100 種獨立工作配置。第一輪可報中位數、分布及 run 間變化，不宣稱可靠的尾延遲 SLA。

## 7　Nsight 收集與匯出

先記錄 `nsys --version`，並查看當地的 `nsys profile --help`、`nsys export --help`。確保 driver、框架與 Nsight 版本支援實際設備，不預設三台的每個欄位都相同。

收集原則：

- P0 使用 CUDA trace；P1 使用 CUDA 與 NVTX trace。
- 初期關閉不必要的 CPU sampling 和 context-switch tracing，記錄實際使用選項。
- 可用 cudaProfilerStart／Stop 控制穩態 capture；完整程式啟動與 warm-up 不作工作特徵。[3]
- 保存 `.nsys-rep`，另匯出 SQLite 供 parser 使用。
- 先檢查 SQLite 的實際 table／column schema，再建立關聯；不要直接假設其他版本範例的表名與欄位在本機相同。
- 記錄 trace 方法、警告、事件遺失與 buffer overflow；缺少欄位標成 missing，不能用 0 假裝量測值。

需要處理的資料至少包含 kernel 開始／結束、CUDA API 開始／結束、process／thread／stream，以及可建立 API–GPU 關聯的欄位。Grid／block 等欄位在確認本機可匯出後再加入。

P0 的 per-request 同步事件可作為第一個分段基準。它利用了 harness 的同步政策，因此結果應明確標為「同步輔助的邊界」，不能宣稱已完成完全依賴週期行為的盲分段。不能直接用 Python timer 與 trace 的絕對時間戳做 join，除非已確認並校正時鐘關係。

## 8　先抽取哪些特徵

第一份特徵表以每個完整推論為一列。Phase 表另存，用於對應檢查與解釋；其標籤不能混入不使用框架資訊的主模型輸入。

| 欄位 | 定義或用途 |
|---|---|
| kernel_count | 該次推論的 kernel 數量 |
| kernel_duration_p10／p50／p90 | kernel duration 分布；由 trace 取得 |
| kernel_duration_sum_us | duration 加總，與聯集時間分別保存 |
| kernel_busy_union_us | kernel 執行區間聯集長度 |
| gpu_span_us | 第一個 kernel 開始到最後一個 kernel 結束 |
| kernel_gap_fraction | 1 − busy_union / gpu_span；只能稱為 kernel 未執行比例，不代表整張 GPU 閒置 |
| short_kernel_fraction | duration < 10 μs 的 kernel 比例；10 μs 是固定診斷門檻，不代表瓶頸分類 |
| launch_count／stream_count | 提交次數與使用 stream 數 |
| memcpy_count／bytes | 若存在，保存傳輸事件與位元組數；意外的 H2D 應排查 |
| grid／block 統計 | 匯出支援時才納入，否則標 missing |

Kernel 名稱保留在原始資料作診斷，第一份數值表示可先不使用名稱。來源 GPU 的 trace 特徵會受來源硬體影響，因此未來模型仍需同時知道來源與目標設備的描述。

## 9　資料怎麼保存

建議每 run 保存以下檔案，路徑包含 device／config／mode／run_id：

```text
environment.json     GPU、CPU、OS、driver、CUDA、框架、Nsys 與執行政策
manifest.json        配置、權重與輸入 checksum、seed、warm-up、capture 條件
latency.csv          每個請求的完成時間與 transition 標記
trace.nsys-rep       原始 trace，只有 P0／P1 有
trace.sqlite         匯出的活動資料，只有 P0／P1 有
features.csv         每次推論的 trace 特徵
phase_mapping.csv    P1 的 NVTX／API／kernel 關聯與 unassigned 原因
quality.json         Worker 正確性、warm-up 與量測完成狀態
trace_quality.json   Parser 邊界／對應檢查與人工審查需求
trace_diagnostics.json 事件遺失／警告診斷線索，需配合原始 report 審查
```

另外產生一份跨設備 `summary.csv`，每列為 device × config × mode × run。先對每個 run 求中位數，再彙整 run 間分布。

未來建模時，predictor 只能讀來源特徵與合法的設備描述；evaluator 才能讀測試工作在目標設備的時間與 trace。Config、batch、shape 和 phase 名稱可保存在 manifest／研究標籤，不自動成為主模型輸入。

## 10　第一輪應產生的結果

第一張圖：G 模式的每次推論時間曲線，顯示 warm-up 與穩態，分別標出三個獨立 run。

第二張圖：G、A、P0、P1 的時間分布與 run 中位數，說明標記與 profiling 對不同 batch 的影響。

第三張圖：P1 中一個完整 iteration 的 CPU NVTX、CUDA API 與 GPU kernel 時間線，展示至少一個經 correlation 確認的對應例子。

第四張圖：兩台設備六個 phase 的 kernel duration 占比，以及各 phase 的 kernel 數；圖中清楚標示這是有標記、有 profiler 條件下的診斷資料。這些占比不能冒充無 profiler 的服務時間分解。

第五張圖：三個 batch 的無 profiler 跨設備時間倍率。

```text
r(B) = 5090 的 G 模式時間中位數 / 4090 的 G 模式時間中位數
```

若 r(B) 隨 batch 改變，表示在這些設定下使用單一倍率可能有系統性誤差；若差異小於量測波動，則目前資料尚未支持使用更複雜模型。兩種結果都要如實報告。

可附加一個簡單預測 sanity check：每次保留一個 batch，以其餘兩個 batch 的倍率中位數預測保留配置，輪流做三次。測試倍率不得參與估計。此結果只用於確認配對與評估程式正確，不能當成正式泛化證據。

## 11　進入下一步的判斷標準

以下是起始驗收值；未通過時保留失敗資料，先查原因，不依結果修改門檻來宣稱通過。

| 檢查項目 | 起始要求 | 未通過時的處理 |
|---|---|---|
| 執行正確性 | 輸出形狀正確、值有限；標記開關符合預定容忍度 | 修正 forward 或數值條件 |
| G 模式穩定性 | 同配置三個 run 中位數的 (max−min)/median ≤ 5% | 查 warm-up、時脈、溫度與背景負載 |
| Iteration 完整性 | P1 中納入分析的完整範圍數與 harness 計數一致 | 檢查 capture 邊界與 transition 排除 |
| Phase 對應 | 已指派 kernel 數與 duration 各占該 iteration 總量至少 99%；其餘有原因記錄 | 檢查 correlation、執行緒、巢狀範圍與缺失事件 |
| Trace 完整性 | 無未處理的 event loss／overflow 警告 | 縮短 capture 或調整收集設定後重收 |
| 標記影響 | 結構沒有意外變化；時間差須報告，超過 5% 就優先精簡標記 | 檢查 hooks、字串建構與過多範圍 |
| Profiling 影響 | 可重現且能量化；不設定必須小於 5% | 若偏差大且不穩定，先調整 trace 方法 |

P0／P1 的跨 run 序列不要求逐事件完全一致。若差異很大，需先分辨是否真的換演算法、存在不穩定執行路徑，或 parser／capture 選錯區間。

## 12　接下來如何接到建模

資料流程通過後，第二步沿用此 harness 執行 Compute、Memory、Launch microbenchmark，建立三台設備的有效效能曲線。再擴充 ResNet50 的 batch／解析度組合，以及其他模型，收集足以分組保留測試的配置。

建模順序為：固定倍率基準 → trace 特徵與設備描述的倍率模型 → 加入硬體校正曲線 → 檢驗 phase 表示與移除框架邊界。每次增加資訊都與上一版使用相同資料切分比較。

Habitat 的價值在於依執行行為差異選擇分析式或學習式預測。[4] 本實驗先取得對應關係與誤差證據，之後才判斷哪些情況值得用分析式縮放，以及哪些情況需要學習模型。

## 參考依據

本文的配置、run 數與驗收值為本研究建議；工具行為參考以下官方文件與論文。

1. [TorchVision ResNet50 文件](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
2. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
3. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
4. [Habitat 原論文](https://arxiv.org/html/2102.00527v2)
