# O4：由 operation 預測一次 GPU inference 迭代時間

更新：2026-11-06。狀態：**詳細設計／待節點實作與量測**。本次只整合文件，沒有新增 GPU 成績或可執行 O4 CLI。

本文件依使用者最新確認，將主目標由 host request 服務時間收斂為：**固定模型、輸入與執行條件，利用來源設備的 dry-run trace，自行辨識運算片段並預測其在目標設備的時間，最後預測一次完整 forward 的 GPU 迭代時間。** 主設備為 GB10／RTX 3090／RTX 4090；任務為 ResNet-50、MobileNetV3、Swin-T、U-Net、YOLO26、Qwen3-0.6B。

**使用者補充的主要部署方向：RTX 4090 → GB10、RTX 4090 → RTX 3090。** RTX 4090 是工作負載 dry run 的來源；另兩台是預測目標。其餘四方向僅作選配泛化檢查，不列本輪完成要求。三台配對收集仍必要，但 test 目標 trace／GI 僅供 evaluator。歷史 O3 六方向統計只作背景，不能冒充這兩方向的 O4 結果。

閱讀順序：本文 → [數學與 MLP 規格](O4_GPU_ITERATION_MATH.md) → [工作負載與配置](O4_WORKLOAD_MATRIX.md) → [節點交接](O4_GPU_ITERATION_HANDOFF.md)。目前證據見 [O3 q 消融摘要](O3_Q_ABLATION_STATUS_20261106.md)。

## 1. 成功時，系統應做到什麼

```text
離線：三設備獨立硬體校正＋受控算子／上下文資料＋reference 標籤
                              ↓ 分組訓練、驗證與凍結
推論：來源設備固定 forward 的 P trace＋兩端硬體校正
                              ↓ 固定辨識器：片段邊界與類型
                              ↓ N0 MLP：每片段目標 CTX 時間
                              ↓ 加總＋有限的整次迭代校正
輸出：目標設備一次 forward 的 GPU event elapsed time、運算清單及完整性狀態
```

已知整次 iteration 的 capture 範圍是允許的；本研究不要求從任意連續服務 trace 自動找出 request。範圍內 operation 邊界、類型與 shape 不能由 wrapper 提供給主 predictor。

「固定週期」指固定工作定義與狀態，不保證每次 kernel 序列或時間完全相同。主要預測量是暖機後固定協定下的典型迭代時間，不是任意負載下的 deadline 保證。LLM prefill／固定 cache 長度的單次 decode 是兩種迭代；完整生成另列擴充。

### 1.1 現況與缺口

| 項目 | 目前證據 | O4 要補什麼 |
|---|---|---|
| 三類辨識 | 節點 O3 有 GEMM/BMM/Conv 受控序列結果 | 真實 DNN、新類別、融合及未知工作 |
| 時間模型 | N0 不使用 q，真實片段 MdAPE 平均 10.49% | DNN 內上下文轉移，較長片段與 Conv 尾端 |
| 完整語意輸出 | N0 typed F1 90.90%，完整算子 success20 66.66% | 不是整個 DNN 迭代成功率；新增 request 評分 |
| reference | warning-accepted，strict verified=0 | 修復／驗證新 P/R 對齊，不能靠重複穩定解除警告 |
| 迭代時間 | 舊資料含 CTX／G0 等不同定義 | 新 GI event 標籤與整次預測校正 |
| Git 與節點 | 主線有 E1～E3/O1；節點有後續 O2/O3 | 盤點並接回節點程式，不覆蓋既有工作 |

最新結果支持 **固定辨識器＋獨立 N0 scalar MLP** 作開發候選。T-v2 joint、NQ、T-v1 F/V 留作對照，不是強制前向步驟。這是看過舊 test 後的設計選擇，需新資料確認。

## 2. 時間標籤與模式：不改名回填舊資料

新 protocol 草案 ID：`o4_gpu_iteration_v1_draft_20261106`。節點完成 pilot 後另保存 frozen manifest/hash；任何修改使用新版本。

| 模式／target_kind | 定義 | 地位 |
|---|---|---|
| **GI / iteration_device_event_us** | 無 profiler、無逐算子標記，CUDA events 包住一次完整 forward 的 elapsed，µs | **O4 主標籤** |
| P / contextual_trace_span_us | 無逐算子 marker 的 profiler trace 中，每個可驗證片段首 GPU start 到末 end | 算子模型 CTX 標籤，仍是 proxy |
| P / iteration_trace_span_us | 完整 iteration 第一個實際 GPU 工作到最後完成的 span | 加總閉合診斷；包含明示納入的 kernel／copy／memset |
| R | 與 P 相同 forward＋reference instrumentation | 訓練與 evaluator 的運算對應，不能當無干擾真值 |
| G0 / request_host_service_us | 無 profiler host start 到同步完成 | 可選輔助量測，已非這次主目標 |
| GE / isolated_device_event_us | 舊 O1 的獨立算子 CUDA event 時間 | 舊基準，不等於 GI 或 DNN 內 CTX |

GI 的 event interval 可能包含 GPU 等待 host 提交的空檔；它不是純 ALU 時間，也不是 kernel busy time。排除等待網路、排隊、模型載入、輸入準備與計時外搬移，但 forward 內實際發生的配置、copy 與等待不能事後刪除。標籤只代表明定 harness／平台條件。

### 2.1 GI 計時原理與偽碼

以下是待實作偽碼，不是已提供的收集 CLI。以單一已驗證 stream 為第一版邊界；若框架啟動其他 stream，必須證明 end event 前已完成正確 join，否則標記 unsupported，不只量 current stream 冒稱完整 forward。

```python
# 計時外：載入權重／固定輸入，設定 eval + inference_mode，預建並初始化 events
# 計時外：warmup，完成所有先前工作，準備固定 LLM cache 狀態
start.record(stream)
output = forward(fixed_input, fixed_state)
end.record(stream)
end.synchronize()                 # 等待不在兩個 event timestamp 之間
elapsed_us = start.elapsed_time(end) * 1000.0
# 計時外：驗證輸出／cache 狀態、保存資料、釋放上一筆 output
```

不得在每個 operation 中間插 event/synchronize；不得將多次 forward 的一對 event 平均值與 single-iteration GI 混用。若未來測連續吞吐，另立協定。CUDA 非同步語意與 events 見 [PyTorch 官方說明](https://docs.pytorch.org/docs/2.15/notes/cuda.html#asynchronous-execution)。

P/R 沿用與 GI 相同的 iteration 級完成政策；可用已知 iteration 窗口與完成事件做 scope 對應，但不注入逐算子同步。Profiler 下 event elapsed 另存為 `profiled_iteration_event_us`，不能代替 GI。相鄰 iteration 之間的 barrier 不計入單次 event interval，會影響暖狀態，必須一致。

### 2.2 建議先固定的 pilot 設定

- 每配置／設備／模式 3 個獨立程序 repeats；順序以保存的 schedule seed 打亂。
- 每程序至少 50 次 warmup，檢查連續三個各 20 次視窗的相鄰中位數差 ≤5%；上限 500 次。沒穩定就保留失敗，不無限等待或刪掉慢值。
- warmup 後固定 5 次 transition，再記錄 40 次單迭代量測。P/R 同樣保留 transition 與 iteration 計數。這是新協定建議值，不改動舊 E/O 資料。
- GI 主標籤：每 process 40 次中位數，再取 3 個 process 中位數。保存全樣本、P10/P90、spread，不只保留聚合值。穩定門檻候選 `(max(run_medians)-min)/median ≤5%`；不穩定案例另列，全部嘗試分母保留。
- 正確性在計時外檢查，FP16 對 FP32／可信實作的 tolerance 由模型 smoke 實測後凍結；不得只用輸出 finite 當完整正確性。
- 精度 FP16、eval/inference_mode、eager、autocast/compile/CUDA Graph 關閉、無其他 GPU 作業；Conv 輸入 NCHW contiguous。原生模型內部 layout 轉換保留。TF32、cuDNN benchmark/deterministic 等明確記錄並在 development 凍結，不為匹配結果偷偷改旗標。
- 同一配置三設備使用同權重與輸入 bytes/token IDs checksum。OOM、精度不支援、輸出不符都回報；不得自動縮 batch 或更換模型。

暖機與穩定門檻不是正確性的保證；不同 session 的 cache／時脈／軟體差異仍需記錄。Nsight 收集以當地版本支援為準，參考 [官方指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

## 3. 來源 dry run 與允許資訊

### 3.1 零調整轉移和新主線要分開

- **現有 artifact 零調整轉移**：保持該模型原本的 27 維特徵、normalizer、S1/S2 來源 view、候選上限與硬體錨點；若不相容就標不能直接套用，不能改特徵後仍稱凍結模型。
- **新 O4 主線**：在開發 validation 上一次選定來源 view，再凍結。候選為單一預定 iteration（例如第 8 次）或第 2～8 次的多觀測統計；後者僅在 P-only 可驗證事件對應時成立。對應失敗不得靠 R 邊界拼接成 predictor 特徵。
- 同一對照所有方法用相同 P；不得逐測試配置選最快 request。原 40-request capture 裁切七次不代表真實只執行七次的成本；記錄完整 warmup/capture/export/parse 費用。短 capture 最佳化留 O5。

### 3.2 主 predictor allowlist

允許數值事件時間、grid/block 各軸、已可取得的 launch/copy 資訊、capture-local stream/API correlation、已知 iteration scope、來源觀測統計及獨立硬體校正。沿用現有兩維 E2 可先作基線；host/launch 校正只有在新增維度和訓練資料一併凍結後才加入。

不允許 test 模型名、層名、真實 op/shape、config ID embedding、reference 邊界、真實片段數、目標 test trace/GI。ID 僅 join。主線不使用 kernel 名稱分類；若要使用原始可觀測名稱，另列 name-feature 消融。版本／架構等 metadata 供分層與相容性檢查；未宣告的 metadata 不進模型。

訓練可讀 train reference/target；validation 可評分與選模；test target 只由 evaluator 讀取。測試目標設備的獨立 E2 microbenchmark 校正合法，但不能以該 test workload 挑最佳錨點。

## 4. 從目前進度到完成目標的工作包

| 階段 | 具體工作 | 主要輸出 | 可前進的條件 |
|---|---|---|---|
| **I0 狀態凍結** | 保護節點 O2/O3、固定目前 N0/辨識器、版本及 hash；列新主標籤 | inventory、artifact index、protocol draft | 能追溯每個來源模型與資料；未收錄 Git 的程式不可假裝已有 CLI |
| **I1 量測與 reference gate** | 三設備少量 GI/P/R，核對完整窗口、事件、單 stream、對齊與 profiler 干擾 | GI samples、quality、R→P audit、失敗分類 | GI 可驗證；逐算子正式評分另須 reference 可驗證，warning-accepted 只能 pilot |
| **I2 六任務盤點** | 每模型先一個小配置，列 atomic/fused 執行單位、零 GPU 操作與未支援類 | operation inventory、事件／時間涵蓋率、候選長度分布 | 不漏 unknown；能決定下一批需補算子，不能只看 op 數 |
| **I3 擴充受控資料** | 新家族與實際 shape 範圍、依賴子圖、正確性、balanced 訓練 | 新 taxonomy、配置群、模型版本、分類／時間對照 | 每類有訓練與留出配置；不強制將新類判入舊三類 |
| **I4 子圖轉移** | 已見 shape＋新上下文；未見 shape＋新上下文分開；先凍結模型 | 零調整／上下文校正兩張表 | reference 與候選覆蓋有效；兩種轉移不能混稱一種泛化 |
| **I5 迭代組合** | SUM、SPAN 診斷、ITER MLP、直接 request 基線與 wrapper 對照 | 4090→GB10／3090 的 GI 預測、OOF provenance、gap/校正診斷 | 所有方法同 GI、資料切分及失敗分母；未識別不補零 |
| **I6 完整模型** | ResNet-50 block→stage→完整；其餘逐類擴充 | 兩個主方向／模型／配置結果與 coverage | 僅對實際完成且明定的任務範圍宣稱成果 |
| **I7 新確認集** | 方法凍結後新配置／日期；模型內與模型留出分開 | 完整報告、配對區間、limitations、可重現腳本 | 未用 test 調參；是否達標依預註冊門檻，失敗如實報告 |

I1 與 I2 可交錯做小型 pilot；reference 未解時可做整體 GI 工程檢查，但不能發表正式逐算子辨識與時間結論。不需先把所有 q／joint 變體做完才開始 I4。

### 4.1 Reference 具體關卡

1. 保存完整命令、環境、trace/export/log 與所有 warning，三台分開審查。解析成功、移除 warning、跨 repeat 穩定都不能單獨證明無遺失。
2. R instrumentation 覆蓋 functional 與 module 內的實際運算；父 module 和子 operation 不重複計數。reshape/view 無 GPU 工作單獨記錄。
3. 由 CPU API/correlation 關聯到 GPU 工作，再對 P 做可驗證的唯一映射；不能只依 duration 接近或標記 CPU 時間重疊猜測。
4. 融合／拆分導致來源目標粒度不同時，改用兩端皆可定義的 logical/fused group，或記無法逐片段配對；不要要求 kernel 一一相同，也不要硬貼錯標籤。
5. scope/事件遺失、非連續片段、跨 stream、R/P 多解與候選漏覆蓋各有獨立失敗狀態。純低 softmax 不是可靠 unknown detector。

### 4.2 新類別與未知工作的處理

候選四事件上限只在舊三類受控資料驗證過。新 DNN 先由 development reference 估可覆蓋長度與成本，再凍結 L；不可用 test gold 調 L。超界或未指派事件保留，不能裁切到容易部分。

建議 taxonomy：一般卷積；group/depthwise 卷積；轉置卷積；GEMM/BMM/Linear；elementwise/activation；normalization；reduction/pooling/softmax；layout/copy/concat/resize；fused attention。每家族保留可評估子類，不能把不同語意全算答對。資料中尚無 unknown 訓練／校準時，至少做 held-out-class 挑戰，不宣稱已能自動拒絕所有新類。

主輸出包含兩個獨立狀態：`time_coverage_status` 與 `semantic_coverage_status`。若只以通用 residual 預測未知工作，時間可能完整但語意不完整，另列 exploratory fallback；主 operation-aware 結果不能混入。第一版不能可靠覆蓋全部 GPU 工作時，輸出 partial/incomplete，不把缺失當 0。

## 5. 資料切分與公平比較

### 5.1 三種問題分開

| 問題 | 允許已見的資料 | 必須留出 |
|---|---|---|
| 上下文轉移 A | 相同單算子 signature | 新的依賴上下文；明示 shape 已見 |
| 尺寸／組合轉移 B | 同類家族 | test 的等價 signature、其跨設備 repeats 與衍生子圖 |
| 整個模型留出 C | 其他模型及事先宣告的受控算子訓練 | 留出模型的 train/validation GI 與 CTX；若單算子 shape 重疊另揭露 |

相同 workload/config 的所有設備、方向、日期、repetition、來源 view、裁切與 input seed 同群。不要將整個 DNN 因共享少數常見算子而強制連成唯一巨大群，再聲稱有五折：依研究問題 A/B/C 建不同 split manifest；B 做 signature 重疊稽核，C 留模型，A 明示允許 signature 重疊。

三組展示配置只是 inventory/pilot，不足以訓練迭代 MLP 或宣稱泛化。先以 development 小矩陣估群間差異與 learning curve，再擴配置；確認集數量依預期差距與區間寬度規劃，不由 repeats 數量替代。形狀與 topology 的 config pool 在測試前凍結。

### 5.2 建模／選模規則

- 新主結果的 train／validation／test 來源固定 RTX 4090，目標為 GB10 與 RTX 3090；兩個目標使用相同配置分組。可以先用歷史多來源 artifact 作零調整／預訓練對照，但必須標示其訓練來源，不能稱只用4090訓練。兩目標的 training CTX/GI 標籤可用，因此主線是已知目標設備的未見配置預測，不是零目標資料泛化。
- 三 seed 17/42/73、同資料與相同候選搜尋預算；逐 seed 評估再平均，不是 ensemble。
- operation 層先保持既有 N0 43→64→32→1 及 loss。增加 taxonomy／特徵後是新模型，完整重訓與消融。
- 完整迭代校正使用 OOF 上游特徵；若分組數不足以產生有效 OOF，就用獨立上下游訓練／校正集，標記資料不足，不借 test。
- 完整迭代 MLP 的候選訓練設定與選模順序見數學規格；只在 development validation 選擇。不得 test 逐方向挑 SUM/ITER/W。
- 舊測試資料全列 development/exploratory。同一資料換 seed、重訓或換文件日期不會恢復成未見確認集。

### 5.3 評分要同時包含 operation 與 iteration

1. 算子：exact boundary F1、typed F1、type macro-F1、typed coverage、真實片段時間診斷及 matched conditional 時間，全部嘗試／拒絕另列。
2. 迭代：GI MdAPE、P90 APE、MAE µs、signed error、低估尾端、`iteration_success20`、完整輸出 coverage。`iteration_success20` 分母為具有有效真實 GI 的全部配置－方向案例，無完整預測算失敗，不得只取成功輸出子集；沒有有效真實 GI 的記不可評分，另以全部嘗試為分母報量測成功率及缺失原因。條件 MdAPE／P90 的有效案例數也須同時列出。
3. 事件／時間涵蓋率：由 evaluator 計算 reference 匹配與 GPU interval union；不把 kernel 數當全部工作。predictor 的自評完整性與 evaluator 真實完整性分開。
4. 4090→GB10、4090→3090 分別報告，再作兩方向等權平均；其餘方向另表。每一配置先聚合 repeats，不能把三 GPU×三 seed×40 iterations 當獨立樣本。
5. 依配置／template 相關群作配對 bootstrap，保留方向／seed 的配對。模型留出只有六個模型，單一平均與狹窄區間不足以支持任意 DNN 泛化。

### 5.4 完成與達標是不同的事

**實驗完成**：六類指定配置與三設備的收集／失敗均有紀錄、無洩漏的整條預測、基準、coverage 與可重現評分已交付；允許結果顯示方法不足。

**性能候選目標**：沿用歷史設計作討論起點，GI MdAPE≤15%、P90 APE≤30%、完整時間輸出 coverage≥90%；相對 W-BCS 的 MdAPE 差≤5pp、P90 差≤10pp。這些不是已驗證的服務需求，也不是使用者已承諾的正式標準；在新確認 test 前以 `criteria_frozen.json` 定案。各方向均報，只有明確達標子範圍可主張成功。對「逼近」須用相同案例的配對差及單側上界，不用 1−APE 當能力保留率。

reference 正確性、事件覆蓋與資料隔離是獨立品質條件，不能用低 GI 誤差抵銷。舊 O3 typed+time20 66.66% 不是此 GI 完成目標的門檻。

## 6. 最先做的可交付工作

1. 節點盤點最新 O2/O3 collector／parser／N0 artifact，保存 commit／patch hash；先不要重寫現有管線。
2. 實作 GI 計時模式及明確 target_kind；選一個小 GEMM、Conv、BMM 在三台做 GI/P/R smoke。
3. 修復 reference 或交付受限原因；在 ResNet-50 B1/224 做完整運算盤點，不只 six phases。
4. 六任務各一個小配置做 inventory；依 trace 補必要算子。以 ResNet block→stage→完整為第一條完成管線。
5. 用明確相容的 N0 artifact 做零調整子圖轉移；必要時另訓上下文校正，不能覆寫前者。
6. GI 組合實驗與新確認集；再依矩陣拓展其餘任務。每階段的交付、偽碼與待實作介面見 [節點交接](O4_GPU_ITERATION_HANDOFF.md)。

## 7. 範圍外與可行性限制

目前不能保證從 trace 唯一還原任意 fused DNN 的原始 Python operation；能預測的粒度須由執行可觀測性決定。若原始語意不可辨識，可改為明示的融合執行群，不能偷換後仍宣稱逐原始 op 恢復。

初版不含多任務共置、動態 batching、多 GPU、網路、排隊、cold start、任意序列長度生成或 deadline 保證。metrics、真實短 capture、未見設備是 O5，不能成為完成 trace 主線的隱性依賴。GB10 與 RTX 若軟體／host 不同，結論屬設備＋執行堆疊轉移。

成本先從 smoke 測 CPU/GPU 工時、capture 大小與解析速度估計，再安排大矩陣；本規劃不虛構總工時或六模型已支援。若 reference／新類／OOF 資料不足，就交付對應限制與已完成範圍。
