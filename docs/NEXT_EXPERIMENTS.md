# O 系列設計：由 trace／metrics 建立 operation-aware 跨設備時間模型

更新：2026-10-21。**狀態：O1 九配置 pilot 已在 GB10 實作並部分實測；O2～O5 仍是設計規格。** RTX 3090、RTX 4090 待用同版程式進行當地量測與品質審查。GB10 P/R 有事件完整性警告，逐算子 P 標籤未驗收。程式與命令見 [O1 節點手冊](O1_NODE_RUNBOOK.md)，實測摘要見 [GB10 O1 狀態](O1_GB10_PILOT_STATUS_20261021.md)。原 E1～E3 工具保留；E4 為節點已回報、本次 Git 尚未收錄。另見 [一致性審查](GOAL_ALIGNMENT_REVIEW_20261021.md)、[資料契約](OPERATOR_DATA_CONTRACT.md)、[節點交接](CODEX_HANDOFF.md)。舊方案見 [歷史文件](archive/WHOLE_MODEL_PLAN_20261020.md)。

使用者追加指定 **MLP 為主要模型**。O2/O3 的具體數學、loss、模型大小、分段DP、OOF及實作順序以 [T-v1 MLP 規格](T_V1_MLP_MODEL_SPEC.md) 為準；本文保留整體實驗流程。線性／倍率模型僅作基線。

## 1. 研究目標與成功的定義

在來源 GPU 對 inference 做短 dry run，取得 CUDA 執行軌跡及實際可用 metrics。模型須自行找出算子片段、辨識類型，再依來源／目標硬體特性預測各算子在另一設備的時間，最後估計整次 inference 服務時間。希望在不提供測試工作的真實邊界、類型、shape 的條件下，接近 wrapper 取得這些資訊時的預測效果。

```text
離線：獨立硬體校正 + 受控算子／序列 + reference 標籤
              ↓ 分組切分、訓練與校正
部署：來源 request 的 trace／可用 metrics
              ↓ 自動分段 → 類型機率／未知類別
              ↓ 各片段的跨設備時間預測
              ↓ 請求時間組合／校正 → 目標服務時間與不確定性
```

必須分別回答「能否找出算子」「能否辨識類型」「時間預測是否接近 wrapper」。只有低時間誤差而無算子辨識，尚未滿足完整目標。最終應用是 AI RAN 的 Near-RT RIC／edge 任務配置；本輪先建立預測模型，排程器與 deadline 保證另驗證。

第一版允許服務端提供**整個 request 的 capture 範圍**，不提供 request 內算子邊界。報告須寫「已知請求窗口、未知算子邊界」。全連續 trace 的 request 自動偵測另列後續實驗。

## 2. 與論文的關係

- [Habitat，ATC 2021](https://www.usenix.org/system/files/atc21-yu.pdf)：§4.1 以框架攔截取得 operation 資訊並個別重播量測，§4.3 用不同 operation 配置建立模型。它會分析完整工作，也會獨立執行算子。可借鏡逐算子建模／組合；本研究額外要從觀測資料恢復 wrapper 直接提供的資訊。訓練中的 backward 時間不能當 inference 標籤。
- [NeuSight，ASPLOS 2025](https://arxiv.org/html/2407.13853v2)：利用 tile、wave、利用率與硬體特徵估計時間，使用 operation metadata 與獨立 operation 資料。可借鏡分層建模；不能簡化成只有算子名稱的回歸，或假定原實作涵蓋全部 Conv2D。
- [Nsight Systems 文件](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics)：GPU metrics 為採樣資料，設備級統計與時間解析度不等同每個 kernel 的 counter，須先盤點實際支援。

本文 wrapper 基線是 **Habitat-inspired 的資訊條件對照**，不是已重現兩篇論文的完整成績。要宣稱超越原方法，需另做相同工作負載與設定的復現。

## 3. 階段、產物與停止條件

| 階段 | 工作 | 必要產物／通過條件 | 未通過時 |
|---|---|---|---|
| O0 可觀測性 | 三節點環境、trace 品質、correlation、metrics 盤點；接回 E4 | inventory、allowlist、可稽核事件映射 | 遺失疑慮資料只列探索；metrics 不可用則保留 trace 主線 |
| O1 受控算子 | 9 配置 pilot，再擴矩陣；已知邊界時間模型 | 正確性、穩定性、G0/GE/P/R 配對 | 先修量測，不擴量 |
| O2 自動辨識 | 無逐算子 marker 的混合序列分段／分類 | 真正 P 的可驗證標籤、邊界／類型／未知類成績 | 只能驗證 R 去標記版時，明列受限證據 |
| O3 整合預測 | 推估類型驅動時間模型，分級 wrapper 對照 | 六方向、配置留出、OOF 全管線、coverage | 定位分段／分類／計時誤差來源 |
| O4 放回 DNN | 獨立算子 → 子圖 → ResNet50 → request | 上下文轉移、算子涵蓋率、服務時間校正 | 部分涵蓋就回報部分 |
| O5 實用性 | 真實短 capture、metrics、跨日期及設備留出 | 成本、穩健性、不確定性 | 裁切長 trace 不當線上低開銷證據 |

O0＋O1 pilot 通過後才收全矩陣。O1 可先建立時間基線，但 O2/O3 才回答新的核心問題。

## 4. O0：協定與硬體資料

固定 FP16、eager、單 stream、無 CUDA Graph、無 torch.compile、無 autocast、無梯度；Conv 使用 contiguous NCHW。三節點共用 manifest、輸入 bytes/checksum、參數與 warm-up 政策。框架／CUDA／cuDNN／Nsight 儘量一致；無法一致時明列「設備＋軟體堆疊」轉移，不能宣稱純架構因果效果。

記錄 GPU UUID、SM 數、記憶體類型／容量、功率與時脈政策、CPU 架構、驅動、框架 build、CUDA/cuDNN、Nsight、trace backend、E2 nvcc／編譯選項及背景工作。GB10 與兩張 RTX 的 host 差異也會影響服務時間。

| 資料類別 | 候選資料 | 限制 |
|---|---|---|
| 靜態硬體 | SM、記憶體容量、可取得的時脈 | 環境盤點；缺值保留缺值 |
| E2 有效校正 | 多尺寸 GEMM、memory read/write/copy、launch、同步、transfer | 保存尺寸曲線／穩定性，不只 peak；與測試工作獨立 |
| CUDA trace | kernel、API、grid/block、stream、記憶體事件 | 事件統計不是實測 FLOPs／occupancy |
| Nsys metrics | SM／Tensor／記憶體活動等實際支援欄位 | 先確認精確名稱、單位、頻率、權限，再取可比交集 |
| 低頻 telemetry | 可用的功率、溫度、時脈 | 作漂移資訊，不是微秒算子的專屬 counter |

此表**不是三台已確認可用的 counter 表**。O0 須輸出每設備 `supported / denied / unavailable / unverified` 清單；同名也核對定義。不默默用 Nsight Compute replay 替代 Nsys；加入時另列成本與觀測條件。

檢查 capture 截斷、跨 PID/context 誤配、runtime/driver 重複計數、warm-up 混入、事件順序及單 stream 假設。已有 CUDA/NVTX 可能未收齊警告，不能以 parser passed 消除。無警告仍需結構檢查；未解決的資料不可支撐主要算子辨識主張。

## 5. O1：算子與資料矩陣

### 5.1 單位及執行政策

一筆 operation 是預先定義的 functional 呼叫及其**全部 GPU 工作**，不是一個 kernel。兩 GPU 的 kernel 數可不同。父 nn.Module 與子 functional 不重複計數；view 等無 GPU 工作操作另列。

GEMM=`torch.mm`，BMM=`torch.bmm` 無 broadcast；Conv2D 無 bias、groups=1。先用 out-of-place、預熱 allocator，輸入準備在計時外；不得混用 E2 `mm(..., out=...)` 時間。Add 無 broadcast；LayerNorm 無 affine、eps=1e-5。禁用 autograd，正確性檢查在計時外。

### 5.2 九配置 pilot

| 類型 | 三種配置 |
|---|---|
| GEMM (M,N,K) | (64,64,64)、(256,1024,256)、(1024,1024,1024) |
| BMM (B,M,N,K) | (2,32,64,64)、(8,64,128,64)、(4,256,256,256) |
| Conv (B,Cin,Cout,H=W,k,stride,padding) | (1,16,32,32,3,1,1)、(4,64,64,56,1,1,0)、(1,64,128,56,3,2,1) |

pilot 驗證工具，不宣稱泛化精度。用跨架構一致、非全 1 的輸入 bytes，檢查有限值與 FP32 reference。候選 FP16 allclose 為 rtol=1e-2、atol=1e-2，同時保存最大誤差及相對 L2 誤差；近零值用絕對容差。pilot 後、正式測試前凍結規則，不能看完 test 才放寬。

### 5.3 正式候選矩陣

| 類型 | 配置生成規則 | 數量 |
|---|---|---:|
| GEMM | M∈{16,128,512}，N,K∈{64,256,1024} | 27 |
| BMM | B∈{2,8,32}，M∈{16,64}，N,K∈{64,256} | 24 |
| Conv2D | B∈{1,4}，Cin∈{16,64}，Cout∈{32,128}，H=W∈{32,64}，k∈{1,3}，stride∈{1,2}；padding=floor(k/2) | 64 |
| Add | 元素數 2^10～2^21，每個整數次方 | 12 |
| LayerNorm | rows∈{1,32,256}，D∈{64,256,1024,4096} | 12 |

核心三類 115 配置，五類 139。檢查相近計算量但形狀不同、長短工作是否足夠，再以新 manifest 擴充。任一設備 OOM／不支援即記錄；不得偷偷縮小。配對用可比交集，另報全部嘗試與失敗分布。

另設不計入 139 的挑戰集：BMM B=1 與相同矩陣 GEMM、可能採 GEMM 路徑的 1×1 Conv、ReLU、softmax、transpose/view。等價呼叫同 split 群；未知類 validation/test 用獨立預定集合。精確語意可能不可識別，允許 ambiguous/unknown，精確類型與粗計算家族分開報。

### 5.4 新量測模式

| 模式 | 用途 | 主 predictor 可見資料 |
|---|---|---|
| G0 | 無 profiler／逐算子標記的 host 完成時間，末端同步 | 預設 evaluator label；來源實測輔助組需各方法同享 |
| GE | 獨立 CUDA event 單算子／整 request 的 device elapsed | 預設 label；含 stream 間隙，不稱純計算時間 |
| P | 無逐算子 marker/event 的 CUDA trace | 正式 trace 輸入 |
| R | wrapper/NVTX reference trace 及 API correlation | 訓練／評估標籤；剝離後只用於明示的研究組 |
| PM | P 加可用 metrics | 同一次 capture 的 trace＋metrics |

P/PM 不插逐算子同步、等待、dummy kernel 或專屬 memcpy 暴露邊界。O1 已知單呼叫範圍，不是 O2 自動分段證據。混合序列 P 不插逐算子 GE；event 計時干擾另審查。

每模式 3 個獨立程序重複，worker 可掃多配置。候選 warm-up 至少 100 次且累計 100 ms、設超時上限；量測至少 5 組，每組至少 100 次，短算子增次數至約 100 ms／組並設上限。精確政策由 pilot 凍結，不用單次 call 冒充獨立配置。單呼叫 GE 與批次平均 GE 分欄，主 label 選定後不混用；event overhead 過大時回報可辨識時間限制。

先取 run 內中位數，再取 3 run 中位數；保留分布。候選 run spread=(max−min)/median≤5%，失敗列不穩定，不無限重跑挑最快。資料生成、載入、輸入搬移、warm-up 不在穩態計時內；有傳輸需求另定義服務目標。

## 6. O2：無標記分段與類型辨識

建立 60 個混合序列模板，每序列 16～48 次呼叫，包含連續同類、同類不同 shape、多 kernel、相似時間不同類。單 stream、無逐算子同步、輸入預配置；先連續呼叫，再有 tensor 依賴的子圖，分開報告。

保留 API launch 順序、GPU correlation 及事件粒度，不只用 E3 聚合統計。T-v1 採 MLP 評分變長候選片段，以 segmental conditional loss 訓練，再用動態規劃選完整分段；MLP 同時提供類型分數。最大片段長度、字典、縮放都只由 train/validation 設定，詳細公式見專門規格。

特徵候選：kernel 數、duration／gap 分布、grid/block、可用 shared memory、kernel 名稱 token、API 關聯與相鄰窗口。主模型允許部署可見 kernel 名稱，並做**去名稱消融**；名稱可能透露語意／尺寸，因此主張是「無 wrapper 語意輸入」，不是完全無語意線索。禁止 NVTX、Python stack、模型／層名、true op count、config ID。高階 cuDNN 等 API tracing 另列額外資訊組，不能混入 CUDA-runtime-only 主組。

### R → 真正 P 的標籤驗證

1. R 依 wrapper scope → CPU CUDA API → GPU correlation 建立標籤；不能以 CPU NVTX 與 GPU 時間相交直接歸屬。
2. R 刪掉標籤後，marker 引入的間隔仍在；稱 sanitized-reference，可先訓練／診斷，不能當真正 P 的部署證據。
3. 獨立 R/P 不能按絕對 timestamp 對齊。Evaluator 在穩定路徑下以完整事件順序、kernel signature、launch geometry、stream 驗證**唯一映射**；不得用模型預測或相近 duration 推定標籤。
4. 重複 signature、多解、事件缺失或路徑改變時，逐算子標籤記為不可驗證；保留在 coverage／失敗統計與可行的整 request 評估。
5. 量 R/P 時間、路徑與 gap 分布差異。若標記改變行為且無可靠對應，真實無標記逐算子主張尚未成立；這是可行性 gate。

輸出類型機率與 unknown/ambiguous。同底層實作可能來自不同 Python 呼叫，不能靠增加模型容量恢復不存在的資訊。

## 7. O3：數學模型與公平對照

### 7.1 分層時間模型

令 x_s 為來源事件序列，h_s/h_t 為獨立硬體描述。辨識器輸出片段 B_hat 與類型機率 q_i(c)，片段特徵為 z_i：

```text
(B_hat, q) = S(x_s, optional_metrics_s)
t_hat_s,i = exp(f(z_i, h_s))
t_hat_t,i,c = t_hat_s,i × exp(g_c(z_i, h_s, h_t))
t_hat_t,i = Σ_c q_i(c) × t_hat_t,i,c
```

f 由訓練配置 GE 校正 trace-span 到指定 device elapsed；g_c 學 log(t_t/t_s)。主方法使用小型 MLP；T-v1 以共享 MLP 在 h_t/h_s 的輸出差分實作 g_c，並對 unknown 採 abstain。CTX 模式以 source trace span 為時間目標，固定 f=0；GE/CTX 不混訓。辨識與來源校正對時間模型的訓練輸入採 OOF，避免 train 用完美標籤而 test 用錯誤推估。

可解釋硬體基線：

```text
t_hat_t ≈ a_C(z)/P_t + a_M(z)/B_t + a_L(z)×L_t
```

P/B/L 由有效 compute/memory/launch 校正取得，a 非負；另比 compute/memory 取 max 的重疊形式。未實測 FLOPs/bytes 時 a 是需求代理，不能當真實 FLOPs／DRAM bytes。三設備不足以保證係數唯一可識別；看留出誤差與穩定性，不因公式像 roofline 就宣稱完成因果分解。

W-BCS 可用 shape 計算 GEMM=2MNK、BMM=2BMNK、Conv≈2BHoutWoutCout(Cin/groups)k² 等算法計算量；理論最少 bytes 不等於 cache/DRAM 流量。這些僅給明示的 shape-aware 組，不當硬體 counter。

### 7.2 資訊條件對照

同工作配置、split、時間 label、硬體資料與來源計時政策；主組只從 trace 校正來源時間。真實來源 GE 輔助須另列 oracle-time 並對等提供。

| 組 | 額外已知資訊 | 分離的效應 |
|---|---|---|
| B0 | 整 request 來源時間代理＋訓練取得的設備倍率；O1 可另做已知邊界單算子版 | 簡單倍率，無邊界辨識能力 |
| W-B | 真實邊界＋相同 trace 特徵 | 邊界資訊 |
| W-BC | 真實邊界、類型＋相同 trace 特徵 | 類型資訊 |
| W-BCS | 真實邊界、類型、shape／參數 | 主要 wrapper operation-aware 參考 |
| T | 邊界／類型全推估，僅來源 trace | 主要研究方法 |
| TM | T＋實際可用 metrics | metrics 增益 |
| W-BCS-M（可選） | W-BCS 同樣得到 metrics | 更充分資訊的對照 |

O1 已知邊界只測時間上限；O3 的 T 必須用推估邊界與類型。B0 不提供 segmentation 成績；單算子 B0 若用了真實邊界，僅與 O1 已知邊界組比較，不能當無邊界主模型。各組調參預算相同，不刻意弱化 wrapper。舊 E4 作整 request 基線，僅在標籤／切分／品質相容時並表。

### 7.3 三設備與留一設備

已知設備測試涵蓋 3090→4090、4090→3090、3090→GB10、GB10→3090、4090→GB10、GB10→4090。可使用各設備訓練配置時間，測試配置跨設備共同留出。

另做三輪 leave-one-device-out。例如留 GB10，所有 GB10 工作負載 trace、時間、類型字典不參與 fit／調參；只允許預先獨立的 GB10 hardware profile。由另外兩台訓練，從非留出設備測試 trace 預測 GB10；留出 GPU 作來源的辨識另報。超參數只在非留出設備選擇。需要 target pair 訓練倍率的 B0 不適用此組，改用明示硬體比例基線。

三種 GPU 只支持有限探索；不能宣稱任意 GPU 泛化。

## 8. 切分、指標與候選驗收標準

canonical signature 包含類型、shape、參數、精度、layout、模式；其 repetition、seed、日期、設備方向、裁切同群。等價挑戰對同群。不得把同 shape 的 3090 放 train、4090 放 test。

候選 5 folds，按類型分層、seed=42、canonical ID 排序後凍結；每輪 train 3 folds、validation 1、test 1。pilot 不算正式 test；與正式矩陣重疊的 pilot signature 只能作固定的開發／訓練補充，不進任一正式 test fold，實際各 fold 配置數另列。混合序列只用同 fold 配置，60 模板候選每 fold 12；同模板排列／重複同群。未見 shape 與未見拓撲是不同測試。

DNN 區分 A「shape 看過、上下文未見」與 B「shape 也未見」。B 須訓練前剔除相同 benchmark signature，不能測後補訓再稱未見。

| 指標 | 規則 |
|---|---|
| Boundary F1 | event/launch index 切點 exact match；容忍窗口成績另列 |
| Segment matching | 一對一、event-set IoU≥0.8；依重疊配對，不用類型挑容易樣本 |
| 類型 Macro-F1 | matched 分類；另報含漏／多分段的 joint typed-segment F1 |
| unknown／拒絕 | 未見類檢出、誤拒、coverage；精確語意與粗家族分開 |
| 時間 | matched 算子 MdAPE、P90 APE、MAE；另報完整 request 及全部失敗 |
| 成本 | capture/export/parse/辨識/predict/初始化分別量 |

零 GPU 工作不用 epsilon 算 APE，另報絕對誤差。不能只報辨識成功子集的低誤差。六方向逐一報，再等權平均；不混用方向平均與 pooled 指標。

候選預註冊標準（正式收集前凍結，非已達標結果）：boundary F1≥0.90、matched type Macro-F1≥0.90、joint typed F1≥0.85、可驗證且可預測 coverage≥90%。時間目標 MdAPE≤15%、P90 APE≤30%；「逼近 W-BCS」定義 T 的 MdAPE 多≤5 百分點、P90 多≤10 百分點。各方向／類別都呈現，總平均不掩蓋失敗。

以配置／模板群 bootstrap，同樣本計配對誤差差與單側 95% 上界；上界低於門檻才支持對應範圍的接近。區間寬就說證據不足；1−APE 不是「保留 operation-aware 能力」。

## 9. O4：獨立算子到 DNN 與服務時間

先用已涵蓋類型組真實依賴子圖：維度相容的連續 GEMM、兩層 Conv、BMM 序列。比較獨立／上下文內相同參數的算子，檢查 cache、資源重用與 host submission gap。含 ReLU/softmax 的 MLP/attention 須先擴 taxonomy，不藏進已知類。

ResNet50 先做完整 inference；它包含 BatchNorm、ReLU、pooling、add、linear/bias 等，三類／五類不能完整涵蓋。先盤點 eager 路徑，擴訓必要類別，或明列「已知部分＋unknown residual」。四項長期模型為 ResNet50、YOLO26、Neural Receiver CNN、Qwen3-0.6B；後三者需先凍結實作、權重、輸入／前後處理，再另定三配置，不以未定模型湊驗證。

Qwen prefill/decode 分開；decode step 不等於整請求，多 token 時間受長度與 KV cache 狀態影響。TensorRT／torch.compile／融合另定「融合執行群」，不保證恢復已消失原始 op。

串行算子時間和是第一個基線，不直接等於 host 服務時間。另以 training request G0 學：

```text
T_hat_service,t = A(sum(t_hat_t,i), predicted_gaps, host/launch_profile_t,
                  request_trace_features_s, unknown_fraction)
```

A 的訓練特徵用 OOF；測試 target G0 僅評分。CPU API 與 GPU 時間可重疊，不能直接相加；獨立量測總和也不保證是下界。多 stream 需依賴圖／critical path，另立實驗。服務時間不含排隊與網路，排程價值最後用到達負載與 deadline 模擬／實測驗證。

## 10. O5：metrics 與短 dry run

PM 的 metrics 對齊同一 capture。候選有效窗口至少 5 samples、覆蓋≥80%，精確定義由 inventory 凍結；不足標 missing/mask。不能把一次粗採樣複製給多個短算子。重複單算子延長窗口只能稱重複工作條件，不能當原 DNN 單次 counter。

per-op metrics 不可用時，可測長窗口上下文，但 train/deploy 相同且不能偷用真實邊界。T/TM 同樣本配對，另報 coverage 與 PM 干擾，不能只留容易長算子。無可比 metrics 就如實記未完成，trace 主線可繼續。

pilot 穩定後對候選 1/4/16 次 request 做真實獨立 capture，量完整成本；長 trace 裁切只算離線敏感度。跨日期、功率／時脈、背景負載先作 robustness，不混入無干擾主組。初版不承諾 concurrent 任務時間。

## 11. 資料量與節點安排

| 範圍 | 配置量測單元（不是程序數） |
|---|---:|
| pilot：9×3 GPU×G0/GE/P/R 四模式×3 repeats | 324 |
| 核心：115×3×4×3 | 4,140 |
| 五類：139×3×4×3 | 5,004 |
| 若全加 PM：139×3×3；實際只做可用子集 | 額外 1,251 |
| 60 模板×3 GPU×G0/P/R 三模式×3 repeats | 1,620 |

每 worker 掃全部 pilot 時，基本 3×4×3=36 程序 session，各配置仍獨立記錄。混合序列沒有逐算子 GE；其逐片段 trace-span 是另一 label，不能與 isolated GE 混成同一目標。pilot 與正式矩陣重疊處按 canonical ID 去重、保留用途。挑戰集、correctness audit、DNN、重測不含表內。

先量 pilot 的 cell warm-up／measure／export 成本與檔案大小，再估總成本≈Σcell 成本＋程序啟動＋解析；目前不承諾工時。三節點用同 manifest、協定與凍結 split。第一個交付是 **O0 inventory＋9 配置 pilot＋R/P 對應報告**，不是直接收 5,004 單元。

## 12. 本輪交付範圍

本次已另加入 [O1 pilot 程式](../o1_operator/README.md)、[節點操作](O1_NODE_RUNBOOK.md) 與 [GB10 受限結果](O1_GB10_PILOT_STATUS_20261021.md)；原本的 [資料契約](OPERATOR_DATA_CONTRACT.md)、[節點清單](CODEX_HANDOFF.md)、[報告範本](OPERATOR_REPORT_TEMPLATE.md)仍適用。E1～E3 凍結量測程式未修改。無法可靠恢復邊界／類型時，應報受限辨識或只有時間預測，不能宣稱完成全部研究目標。
