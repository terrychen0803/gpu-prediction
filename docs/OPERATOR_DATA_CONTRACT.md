# O 系列資料契約與資訊隔離

版本提案：`operator_protocol_v1_draft_20261021`。**這是節點待實作規格，不是現有檔案或 CLI。** 第一輪 pilot 後凍結正式版本；變更需新 protocol ID。主要流程見 [實驗設計](NEXT_EXPERIMENTS.md)。

MLP 主模型的具體流程見 [T-v1 數學規格](T_V1_MLP_MODEL_SPEC.md)。首版 unknown 是拒絕狀態，已知類的條件機率與邊界posterior分開保存；不得虛構 unknown softmax 機率。

## 1. 三種時間目標不可混算

| target_kind | 定義 | 合法用途／限制 |
|---|---|---|
| isolated_device_event_us | GE 獨立算子 CUDA events 間 elapsed | O1/O3 獨立算子時間主標籤；記 single-call 或 batch-amortized |
| contextual_trace_span_us | 完整 P/PM/R 內，reference 可驗證片段的首 GPU start 到末 GPU end | 上下文逐算子 proxy，受 profiler 干擾；P/R/PM 必須分欄 |
| request_host_service_us | 無 profiler G0 的 host start 到末端 synchronize 完成 | O4 完整 request 主標籤；不含排隊、網路、載入及事前輸入搬移 |

另存 kernel busy union、GPU request span、host submit 時間等，禁止都命名為 `latency`。一個單位由多 kernel 構成時同時記 span 與 busy union，不能把 overlap 重複相加。GE event 時間不是純 ALU 時間。

**完整 DNN 中無干擾的逐算子時間沒有自動可得的真值。** 插入 event／同步本身會改變執行；R 也有標記干擾。因此：獨立算子 GE 的預測精度、上下文 trace proxy 的預測精度、無 profiler request G0 的精度，三張表分開。O4 不能把 target trace proxy 當無 profiler 每算子真值，或把 isolated GE 當實際上下文時間。若需要新上下文時間模型，僅以 training-context label 擬合，與 zero-shot isolated→context 結果分開。

## 2. 檔案邊界：predictor 與 evaluator

以下為建議目錄，實作可調整但須維持資訊隔離：

```text
<dataset>/
  protocol.json
  inventory.json
  hardware_profiles/<hardware_profile_id>.json
  predictor/<opaque_capture_id>/
    events.jsonl
    metrics.jsonl             # 可缺；缺值不能補成 0
    observable_context.json
  evaluator_only/
    workloads.jsonl
    runs.jsonl
    reference_operations.jsonl
    labels.jsonl
    rp_alignment.jsonl
    pairing.jsonl
  quality/<opaque_capture_id>.json
  splits/<split_id>.json
  provenance/checksums.json
```

原始 `.nsys-rep`／SQLite／profiler log 保留資料儲存處，完整 checksum 與不可變來源路徑記 inventory；不提交 Git。原始 SQLite 可能帶 NVTX／工作名稱，**predictor 不直接讀原始 SQLite**。由審核過的 exporter 產生 allowlist 視圖，評估程式經獨立 ID join 打分。

train labels 合法供訓練器監督；test labels 只由 evaluator 開啟。推論入口只接收 predictor view、硬體 profile 與模型 artifact；不接受 workload manifest。無法用目錄結構隔離時也必須有 schema allowlist、欄位拒絕與洩漏測試。

## 3. 欄位規格

### 3.1 protocol.json / inventory.json

| 欄位 | 要求 |
|---|---|
| protocol_id / taxonomy_version / collector_commit | 必填；dirty patch 有 checksum；不能把變更回填舊 run |
| timing_policy | warm-up、組數、呼叫數、GE single/batch、同步、輸入準備、timer 範圍 |
| execution_policy | precision/layout、autocast、grad、eager/compile/graphs、stream、allocator/out 策略、TF32/cuDNN flags |
| software_stack | OS、CPU arch、Python、torch build、CUDA/cuDNN、driver、Nsight、trace backend、compiler |
| input_provenance | generator version、seed、實際 bytes checksum；跨架構 checksum 一致 |
| devices / modes / repeats | 預期與找到項目，保留失敗與缺失 |
| feature_allowlist / kernel_name_policy | 名稱可用／去名稱組；明示高階 API 是否允許 |
| thresholds / split_hash | pilot 後凍結，測試前寫入；變更另建分析 |

硬體 profile 包含 E2 來源 commit、protocol、cell 尺寸、原始單位、median／spread、有效／待審狀態及 toolchain。P/B/L 的映射規則固定；不得依測試工作 target latency 挑選最合適的硬體錨點。

### 3.2 events.jsonl：來源可觀測事件

必要欄位：`event_id`、`event_kind`、`start_ns`、`end_ns`、`stream_id`、`context_id`、`process_id`、`correlation_id`、`api_event_id`、`launch_order`；kernel 可含 `kernel_name`、`grid_xyz`、`block_xyz`、`dynamic_shared_memory_bytes`（可取得才填）。memcpy 可含實際可取得的 bytes/direction。

以 capture 內整數 nanosecond 保存時間，再明確換算 µs，禁止大絕對 timestamp 浮點相減損精度。ID 只供關聯，不作數值／字串 embedding。PID/context/stream 需重映射成 capture-local opaque ID，不能洩漏測試配置。runtime/driver 同一 launch 不重複記。

可有 `capture_start/end` 的已知 request scope，但不得有 `op_id`、`op_type`、`shape`、`layer_name`、`nvtx_label`、`reference_boundary`、`true_op_count`。來源 event_kind/kernel name 是原觀測，與真實 op_type 區分。

### 3.3 metrics.jsonl 與 inventory

每筆含 `sample_timestamp_ns`、`window_start/end_ns`（工具有定義才填）、`metric_name`、`value`、`unit`、`device_id`、`sampling_period_ns`、`valid`、`missing_reason`。保存工具原始時間軸與換算方式；不在 schema 中虛構無法取得的採樣窗口。

每設備盤點 `metric_name / definition / unit / sample_rate / scope / supported_status / permission / collector_command / evidence_path`。沒有工具提供 sampling window 時，用已文件化的採樣點覆蓋規則，不能宣稱精確每算子積分。至少 5 samples、80% coverage 是候選 gate，須寫明分母與邊界判定。缺值不等於 idle／0。

### 3.4 evaluator_only

| 檔案 | 核心欄位 |
|---|---|
| workloads | canonical signature、family、參數、tensor shape/stride、precision/layout、input hash、template/topology、equivalence_group |
| runs | capture→workload/device/mode/repeat 的關係、全部失敗、環境與版本、correctness／穩態 |
| reference_operations | logical_call_index、op_type、參數、reference CPU scope、API IDs、GPU event IDs、zero_gpu_work、fusion_group |
| labels | target_kind、timing_mode、單位、原始 samples、run median、跨 run 聚合、品質與來源 |
| rp_alignment | R/P IDs、matching 規則版本、事件數、signature、唯一性、未配對原因、干擾統計、review 狀態 |
| pairing | 相同 logical workload 的 source/target ID；不假設 kernel 一對一 |

標籤 provenance 至少區分 `reference_direct`、`reference_join_unique`、`isolated_scope_known`、`unverifiable`。`reference_join_unique` 需重複路徑一致與完整序列證據；duration 相近不算證據。參照 P 的逐算子標籤不能由預測器輸出反推。

## 4. 預測器輸出

每 request 須回傳：

- `prediction_id / model_hash / split_id / source_capture_id / source_hw_id / target_hw_id`。
- 每片段 `predicted_event_ids`、起迄 event index、已知類條件機率（總和=1）、`segment_posterior`、`predicted_class`、`ambiguous_candidates`、`abstain_reason`；拒絕時class可為unknown，但不是把未知類訓練成已知類。
- `target_kind`、`predicted_time_us`、區間估計／方法及校正集來源；未做區間校準則填null及原因，不能把分類 softmax 或seed spread當時間信賴區間。
- request 時間、unassigned events、unknown fraction、confidence、是否部分預測、各步耗時。

禁止回傳未宣告的 fallback oracle 結果。unknown 可用通用時間模型，但明示分類未成功。預測片段不應互相重複覆蓋事件；保留 orphan/unassigned，不能為降低時間誤差直接忽略。

## 5. 評估配對與公平比較

先依 event-set overlap 最大化做一對一配對，再用 IoU≥0.8 接受；不根據類型或時間誤差挑選。多切一個真實算子會產生額外 FP；合併兩算子造成漏辨識；另以 event index exact boundary F1 顯示。

各工作配置對 source/target 的配對依 evaluator 的 logical call/config，不依 target kernel 名稱。測試 target trace 只能用來評分上下文 proxy／診斷，不能供 source predictor 選擇路徑。W-B/W-BC/W-BCS 與 T 使用同一 P trace 的事件（需可靠 reference join）；R 版本必須另報，不能 W 用 R、T 用 P 卻忽略干擾。

時間主表同時報：全部嘗試、有效 reference、可預測、正確辨識、matched conditional error、完整 request error。T/TM/W-BCS 等方法共同可比樣本的 paired difference 與各自 coverage 同時呈現；沒有 matched prediction 不能填 0 或只從分母消失。

## 6. 切分與未見設備

split manifest 保存 canonical groups、等價群、template 群、所有設備與 mode 的 sample IDs。同配置的所有 seeds/repeats/日期/裁切同群。預處理、名稱 token 字典、未知門檻、模型選擇及時間校正皆只 fit train/validation。訓練時間模型用 OOF segmentation/class/source-time；inner folds 也按配置分組。

已知設備六方向與 leave-one-device-out 是兩種 protocol。後者留出設備的任何工作負載資料不得參與 fit，硬體校正可用但要事先收集並記 provenance。source 是留出設備的測試另報，不能在該設備測試後用其名稱字典再訓練。

不把三設備×三 repeats 算九倍獨立工作配置。bootstrap 以配置或模板群為單位，方向分析保留相依結構。不同模式／時間 target 不可拼為同一 label。

## 7. 節點實作時必要的防洩漏與品質檢查

1. 改動／刪除 test oracle、config 名稱、檔案順序不應改 predictor 輸出；ID 僅供 join。
2. 孤立 test target 檔案後 predictor 仍可工作；evaluator 才能讀 target label。
3. runtime/driver 重複、跨 PID 同 correlation、空 trace、多 stream、漏 event、未知 SQLite schema、時鐘單位錯誤必須能檢出。
4. reference 的巢狀 scope 不雙重計數；無 GPU 工作及 fused 群保留明確分類。
5. R→P 唯一／多解／路徑變動的 fixtures 分別測；不接受以時間最近猜標籤。
6. 同 configuration／equivalence／template 跨 split 應失敗；所有模型 artifact 保存訓練群 hash。
7. missing metrics、採樣不足、CPU/GPU 時間軸不一致應失敗或降級，不補 0。
8. 合成 fixture 僅測 parser／演算法，不寫進研究成績；GPU correctness 必須節點實測。

這些是實作後的風險導向測試清單；本次文件整合未執行或宣稱通過。
