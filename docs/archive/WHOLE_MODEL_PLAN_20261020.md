> 歷史文件：保存 2026-10-20 的 E 系列設計與交接背景。2026-10-21 起的主要目標、優先順序與執行範圍以 [現行設計](../NEXT_EXPERIMENTS.md) 與 [現行交接](../CODEX_HANDOFF.md) 為準；本檔內的「目前／下一步」均指當時狀態。

# 後續實驗計畫：從可信量測到跨設備推論時間預測

更新：2026-10-20。狀態：GB10／3090／4090 的固定輸入 E1 各有 36 個正式 run；三設備 E2 各完成 100 cell／1,500 群，品質與限制見本機 `analysis_runs/three_gpu_e2_20261020/REPORT.md`。E3 的跨解析度固定輸入、量測入口與 C0/C1 pilot 已實作，三設備新配置待收齊；5090 暫不可用。正式研究設備為 GB10、RTX 3090、RTX 4090、RTX 5090；場景為 AI RAN 平台的 Near-RT RIC／edge 應用。本文提出的候選設定不是實測結果或精度保證。

目前可執行的功能請以 [README](../../README.md)、[E3 節點手冊](../E3_NODE_RUNBOOK.md)與程式為準。E2 microbenchmark 與 E3 C0/C1 pilot 已有工具；C2 多特徵模型、E4 跨設備模型、其他任務與排程器尚未實作。新 E3 resolution grid 已由 `inference_pilot` 支援，舊 E1 session 不會回填為新協定。

## 1. 研究要回答什麼

對一個新推論工作，只在來源設備執行短 dry run，從 Nsight Systems 取得 kernel／CUDA API 軌跡，再結合預先取得的來源與目標設備描述，預測該工作在另一台設備的服務時間。最終希望用這個預測協助任務放置與截止時間管理。

四類應用全部是 inference；「訓練模型」在本文只表示訓練效能預測器。先前討論過的訓練 step、optimizer、全參數微調或 LoRA 不屬於現在的工作負載。

主時間指標 S0 為：模型已載入、輸入已駐留設備、完成預熱、queue depth=1 時，從呼叫 forward 到 GPU 工作完成的 host 服務時間。一個 iteration 是一個批次請求。模型載入、外部前後處理、網路、等待佇列與湊批時間均另外處理。跨 GB10 與桌機時，S0 同時含主機提交影響，因此稱設備／平台服務時間較準確。

第一個研究成果不要求預測每個 kernel。先證明「來源 trace 是否能預測完整請求」，再檢查分階段或序列表示是否改善誤差。這能避免跨架構 kernel 數量或實作改變時，強迫做一對一對應。

## 2. 固定資料可見範圍

| 設定 | 預測時可讀 | 不可讀 | 能支持的主張 |
|---|---|---|---|
| 主實驗：已校正設備、新工作配置 | 測試工作的來源 P0 trace、來源／目標設備描述、訓練資料學到的參數 | 該測試工作的目標 G 時間與目標 trace | 四台已知設備間的新配置預測 |
| 進階：留一目標設備 | 已見來源設備的測試 trace、留出設備的一次性 microbenchmark 與規格 | 留出設備全部 AI trace／時間，包括把它當訓練 source | 硬體已表徵、AI 工作未量測的設備轉移 |
| 延伸：只有目標公開規格 | 來源 trace、目標公開規格 | 目標 microbenchmark 與 AI 時間／trace | 更嚴格的硬體外推；先不承諾效果 |

訓練階段需要跨設備的 G 時間標籤。測試時不提供目標標籤，和「永遠不量測任何目標資料」是不同設定。

模型名稱、batch、shape、prompt/output 長度、NVTX phase 名稱與人工邊界只供配對、切分與評估；主模型使用明確的 feature allowlist。加入服務設定或 operator features 的模型，單獨列為額外資訊比較組。GPU Metrics 也另列增強組。

## 3. 實驗順序與完成條件

| 階段 | 工作 | 產物與通過條件 | 程式狀態 |
|---|---|---|---|
| E1 | ResNet50 量測與標記稽核 | 正確性、穩定性、trace 對應與四模式成本報告 | GB10／3090／4090 固定輸入各 36 run 自動檢查通過，trace 警告待審；5090 待設備可用 |
| E2 | Compute／Memory／Launch／Host 校正 | 四設備有效效能曲線、重複量測與版本紀錄 | 前三台各有 100 cell／1,500 群；5090 待執行；波動 cell 與工具鏈差異見本機整合報告 |
| E3 | 單設備來源時間校正 | 用 P0 trace 預測同設備 G，保留配置測試 | C0/C1 pilot 與解析度 grid 量測入口已實作；三設備新矩陣與 trace 品質仍待收集／審查，C2 未實作 |
| E4 | 第一個跨設備模型 | 固定倍率、硬體倍率、trace 模型的公平比較 | 待實作 |
| E5 | 四任務、序列與短 dry run | 未見配置／任務／設備結果、error–coverage 與總成本 | 待實作 |
| E6 | 排程效用 | 相同到達序列下的 deadline miss、goodput 與 oracle 差距 | 待實作 |

先完成 E1 的一台 batch 1，再展開四台。E2 可以在 E1 流程穩定後進行。E3 與 E4 先使用 ResNet50 批次／解析度網格；資料量不足時以小型正則化模型為主。E5 才檢查對更多任務的泛化；E6 使用已凍結的預測器。

## 4. E1：確認現有資料能用

依 README 使用倉庫 [固定輸入](../../inputs/README.md)先執行 G、P1 的 batch 1 試跑，再執行四模式 G／A／P0／P1 各三個獨立程序。確認後擴為 batch={1,8,32}，四設備合計 144 runs／session。已有相同協定且品質合格的 runs 可以納入，但須保存是否重用，不把同一資料複製成新樣本。

固定驗收：輸出正確；每組三個 G run 中位數的 `(max-min)/median <= 5%`；P1 kernel 數與 duration 的 phase 覆蓋率皆至少 99%；同步／iteration 計數一致；人工確認沒有未處理的事件遺失警告。未通過先排查或重收，不能調寬門檻後沿用同一批結果宣稱通過。

報告 A/G、P0/G、P1/A、P1/G 的比例與微秒差，也檢查 kernel 數、執行序列是否有意外改變。P1 圖是被 profiler 與標記影響後的診斷，不能直接當成 G 的服務時間分解。

目前 P0 分段為 `synchronization_assisted_not_blind`，且 parser 讀取 manifest 計數做完整性檢查。這是研究用 harness 下的無 phase 標記模式；完全外部、無 harness 的部署還需要更換分段與完整性檢查介面。P1 wrapper 的六個 phase 是稽核工具，並不要求未來預測時提供 operator 類型或 tensor shape。

## 5. E2：一次性硬體校正

硬體校正應回答「這個大小與執行方式下，平台實際能多快」，而不只收型錄峰值。所有設備用相同工作量定義、程式版本、精度與同步政策。無 profiler 計時為主，另做少量 trace 確認真正的 kernel 路徑。GPU 事件時間與 host 完成時間分別保存。

| 類別 | 起始配置 | 算法與紀錄 | 重要限制 |
|---|---|---|---|
| Compute | FP16 GEMM 方陣 M=N=K={128,512,2048}；另 M={1,8,32}, N=K={1024,4096} | FLOPs=2MNK；保存 latency、有效 FLOP/s、layout、backend | 小矩陣效率與大矩陣不同；FP16 不等於每條指令或累加均 FP16 |
| Memory | device copy／read／write；工作集 {64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB} | copy 若採讀+寫則 bytes=2N；保存配置方式與有效 bytes/s | 有效流量不是實際 DRAM 交易量，不能藉此宣稱量到 cache hit rate |
| Launch | 可驗證的 empty kernel 與固定短 kernel；N={1,10,100,1000} | 分別量連續 enqueue 後同步、逐次同步；CPU 提交時間與完成時間分開 | 框架算子可能融合或消除，需 CUDA 實作與 trace 驗證；不假裝 torch.empty 是 GPU 運算 |
| Host／transfer | 同步呼叫、固定 Python 呼叫路徑；H2D/D2H 的 4KiB～64MiB 對數掃描 | 固定 pinned/pageable、方向、bytes、CPU threads 與駐留政策 | PCIe copy 與 GB10 的記憶體路徑不能只按型錄名稱類比 |

Memory 同時量「同一 buffer 重複使用」與「輪替足夠大的一組 buffer」。只有工作集及輪替總量確實超過當地已確認 cache 容量時，才稱為超過 cache 的測試；先保留這兩種存取政策名稱，不預設一定冷／熱。

每個校正 cell 先做三個獨立程序；每程序至少五個重複計時群組，每群組選足夠 repetition，使計時區間達約 100ms 並記錄 N。觀察延長群組是否改變時脈／溫度。對 Launch 類另保存有限 N 的原始總時間，不只報除以 N 後的數值。使用中位數及分布，短 kernel 不用單次 host timer 估 GPU duration。

現有 E2 工具已保存 `hardware_profile.json` 與 `microbench_samples.csv`，包含 profile_id、GPU UUID、軟體／程式 hash、類型、工作量、精度、記憶體配置、計時範圍、repeat 與 duration；環境快照見 `runNN/environment.json`。三設備來源、品質與限制見本機 E2 整合報告。

第一版以事前固定的幾個錨點構成硬體向量 h_d：大／小 GEMM、兩個工作集的有效頻寬、launch 完成成本與 host 成本。只用三種設備時，不適合同時擬合大量硬體係數。若無法取得同版本軟體，另立 protocol 分組；跨軟體轉移是額外實驗，不放寬既有比對規則來湊配對。

## 6. E3：先解決單設備來源時間校正

問題是：來源設備開了 Nsight Systems 後得到的 trace，能否估計同設備沒有 profiler 時的 S0？目前 `latency.csv` 的 P0/P1 host 時間可作評估參考，但嚴格 trace 主模型不直接讀它當輸入。

定義每配置的 G 標籤 `T_s` 為獨立 G runs 中位數的中位數。對每個 P0 run，把量測區間內各 iteration 的數值 trace 特徵取中位數及預先定義的變異摘要，得到 z_s。不要把數百個 iteration 當成數百種獨立工作。

第一版比較三種校正器，所有時間用同一單位 us：

```text
C0: T_source_hat = median(gpu_span_us)
C1: T_source_hat = exp(a_d + b_d * log(median(gpu_span_us)))
C2: T_source_hat = exp(a + b * log(median(gpu_span_us)) + w^T z'_s + v^T h_s)
```

C0 刻意作為量測範圍不完整的簡單基準；C1 是每設備的低容量經驗校正；C2 用 ridge 等正則化方法加入 kernel 數、短 kernel 比例、gap、launch 數及硬體描述。z' 排除已作錨點的相同欄位，特徵縮放只在訓練 fold 擬合。log 輸入必須為正；缺失值採訓練集決定的補值加 missing mask，或使用共同可用欄位子集。

P0 的兩個同步結束點之間可能包含上一請求結束後的 Python 記帳與下一請求準備，所以同步分段長度不等於 S0；`gpu_span_us` 也少了部分 host 成本。C2 是從可取得特徵學習 G 服務時間，不能把任何一個 trace 長度直接稱為無 profiler 標準答案。

對時間誤差分開報告：分段能否找到正確請求、P0/P1 與 G 的量測差異、校正器對保留配置 G 的誤差。四個模式是不同程序，不存在逐 iteration 完美配對；以同配置、協定與 session 的分布比較。

ResNet50 先擴成 B={1,4,8,32} × resolution={160,224,320} 共 12 配置。此解析度選項需先實作，現有 CLI 固定 224。主資料每配置每設備 G×3 + P0×3，即 216 runs／session。A／P1 稽核另計；改變解析度、模型或 backend 時，以涵蓋最小與最大工作量的配置重做四模式稽核。

至少做 grouped 配置留出，並另完整留出最大 batch 或解析度做外插。三個 batch 的現有 sanity check 只能驗證流程，不能支撐上述校正器的可靠泛化結論。

## 7. E4：跨設備數學模型

先使用相同 workload configuration 的來源、目標 G 標籤建立 supervision；來源 P0 建特徵。測試時只能取得來源 trace 與硬體描述。對四設備的十二個有向方向各自報告結果。

| 模型 | 定義 | 想確認的問題 |
|---|---|---|
| M0 | `T_target_hat = T_source_hat` | 完全不縮放有多差 |
| M1 | `T_target_hat = T_source_hat * r_pair`；r_pair 為訓練配置 G 倍率的中位數 | 是否真的需要複雜模型 |
| M2-C／M2-B | 分別乘 `C_source/C_target` 或 `B_source/B_target` | 純運算或純頻寬縮放的適用範圍 |
| M3 | 來源 trace 無序統計 + h_source/h_target 的小型倍率模型 | trace 是否比單一倍率有額外資訊 |
| M4 | M3 加入有序特徵，保持相同調參預算 | 序列是否有額外價值 |
| M5 | M4 加 GPU Metrics 的共同可用欄位 | metrics 的收益是否值得成本 |

M2 的錨點在看測試結果前固定；不能逐測試樣本挑誤差最低的 compute 或 memory 基準。M1 的設備 pair 常數只適用已知設備，不可用於尚無該 pair 訓練標籤的留一設備設定。

M3 的一般形式為：

```text
T_target_hat = T_source_hat * exp(g_theta(z_source, h_source, h_target))
```

需要較容易說明的硬體模型時，採低維正則化交互式：

```text
uC = log(C_source / C_target)
uB = log(B_source / B_target)
uL = log(L_target / L_source)
uH = log(H_target / H_source)

g_theta = a + beta(z_source)^T [uC, uB, uL, uH]
beta(z_source) = b + W * z_small
```

C、B 是有效吞吐，越大越快；L、H 是固定政策下的成本，越大越慢，因此比值方向不同。z_small 先取 3～5 個事前決定的 trace 特徵，與普通 ridge 模型比較；四設備不足以穩定辨識大型 W。係數是經驗縮放敏感度，不能稱為真實 compute／memory 時間比例。不要把 Compute、Memory、Launch 三個估計直接相加，因為提交、運算與記憶體存取可能重疊。

為保持校正與跨設備訓練的輸入品質一致：在外層訓練資料內，以配置群組做內層 cross-fitting，得到來源時間的 out-of-fold 預測 `T_source_OOF`。跨設備訓練標籤使用 `log(T_target_G / T_source_OOF)`；外層測試再使用只在外層訓練資料擬合的校正器。M1 保留 G/G 訓練倍率作直觀基線，也可另列以 OOF 來源錨點擬合的固定倍率版本，名稱分開。不能訓練時用完美來源 G、部署時才換成帶誤差的來源估計，卻忽略此差異。

另做 oracle-source-G 與 oracle-boundary 診斷，量化改進上限；oracle 不能混進主結果。若先做直接預測 log target time 的 ridge，也保留為比較，讓兩階段是否值得由實驗決定。

## 8. E5：擴到四類任務與短 trace

以下三組 pilot 是候選錨點。全設備共同可行、版本及數值檢查完成後才凍結正式矩陣。

| 任務 | C1 | C2 | C3 | 範圍 |
|---|---|---|---|---|
| ResNet50 | B1，224² | B8，224² | B32，224² | 已實作，固定 TorchVision 權重 |
| YOLO26n | B1，320² | B1，640² | B4，640² | 待固定實際權重 revision、套件及 forward 輸出範圍 |
| Neural Receiver CNN | B1，14×128 | B8，14×128 | B32，14×128 | 研究代理，無指定原始實作 |
| Qwen3-0.6B | B1，prompt128/output32 | B1，prompt1024/output128 | B4，prompt2048/output128 | 固定長度請求，待實作 |

Receiver 候選設計為 2 Rx 的實／虛訊號加噪聲特徵共 5 通道、128 hidden channels、4 residual blocks、2 輸出通道。卷積、padding、normalization、activation、權重與 seed 必須先寫入 architecture spec 並凍結；這是計算代理，不代表經通訊訓練驗證的接收器，不能報 BER／BLER 能力。

Qwen 每請求重置 KV cache、關閉 prefix cache，tokenization 在範圍外，固定 decoding/sampling 規則與輸出長度；不讓 EOS 改變工作量。模型載入不計入，cache 初始化／重置是否在 S0 內要明確紀錄並保持一致。先預測完整固定長度請求，再做 prefill 與逐 token decode。首 token 通常由 prefill 產生，N 個輸出 token 的後續 decode 次數需按實作核對。未知的未來輸出長度不能只靠一次短 trace 保證推知。

正式候選網格：ResNet50 12 配置；YOLO26n B={1,2,4}×resolution={320,480,640} 共 9；Receiver B={1,8,32}×FFT={128,256,512} 共 9；Qwen B={1,2,4}×prompt={128,1024,2048}×output={32,128} 共 18。總共 48 種工作配置、192 個配置—設備單元。

主資料每單元 G×3 + P0×3，共 1152 runs／session；若做三個時段／日期共 3456 runs。四任務三錨點的完整四模式稽核是 12×4×4×3=576 runs／session。兩種設計有交集，可以重用同協定且合格的 runs，實際去重後另報總量。不要把這些數字直接相加當最低必要量。48 配置仍是小型資料集；十二方向產生的 576 個配置配對並不等於 576 個獨立 workload。

OOM 或不支援的 cell 保存為 infeasible，不在個別設備默默改 batch／精度。主精度分析使用共同可行子集，另報不可行率。正式測試配置應在調參前保留；若 48 配置全用於探索，再宣稱確認性成果前需另收未看過的配置。

序列特徵先嘗試 duration 轉移、相鄰 token 頻率、固定 bin 波形與重複片段。比較完全無 kernel 名稱、規則正規化名稱、原名三組；名稱可能攜帶框架與形狀線索。字典、PCA、標準化、特徵選擇皆只 fit 訓練 fold。把 sequence 在 train/test 以相同方式打亂後重新訓練，才能較公平檢驗順序的貢獻。

短 trace 先從長 capture 裁切 0.1／0.5／1／2／5 秒研究資訊量；Qwen 長請求另選足夠覆蓋完整請求的區間。不足完整單位時允許 abstain 並報 coverage。選定長度後必須真正重新執行短 capture，計入啟停、export、parser、校正與預測總成本。warm-up 與模型載入成本另列；不可只用離線裁切時間宣稱線上成本很低。

純週期分段另立實驗：在預測器中移除 NVTX、harness 計數與同步邊界捷徑，對候選週期做估測；reference 僅 evaluator 讀取。特別檢查重複 layer、半週期／雙週期與 decode 重複。成功前維持「同步輔助」的主張。

## 9. 評估與防洩漏

1. 主切分以 workload configuration 分組；同配置所有設備、日期、run、方向與裁切一起進同一外層 fold。來源測試 trace 是合法測試輸入；目標標籤只能在輸出預測後交給 evaluator。
2. 另做整個最大 batch／解析度／context 留出；內插與外插分開報。
3. 四任務 leave-one-task-out：整個任務的資料不參與校正、字典建立或調參。
4. 四設備 leave-one-device-out：留出設備不能當訓練 target 或 source；硬體校正資料是否允許，在設定名稱標清楚。
5. 跨 session 穩健性另列；同配置跨日期留出代表時間漂移測試，不冒充未見配置泛化。

在每個外層 fold 內完成校正、特徵處理、模型選擇與不確定度門檻。保存 split manifest、訓練配置 ID、原始資料 checksum、程式 commit、seed、特徵清單與預測檔。不要將 target 路徑或可推回 target 時間的欄位交給 predictor。

報 MdAPE、MAPE、P90 APE、MAE(ms)、coverage 與失敗數；P90 APE 不是服務延遲 p90。先對配置／方向彙整再 macro-average，並列十二方向、四任務及最差配置。bootstrap 以配置為抽樣群組；只有少數群組時直接承認估計不穩定，不用 iteration 數量製造狹窄信賴區間。

探索目標可先用 MdAPE≤10%、P90 APE≤25%，但是否足夠需由排程 deadline slack 決定；不能達標就自動宣稱可部署。序列與硬體曲線需在相同切分、模型容量／調參預算下持續改善，才支持增加複雜度。所有失敗與 abstention 都要計入覆蓋率分母。

## 10. E6：證明對 AI RAN edge 排程有幫助

先做離線事件回放：模型常駐、每台設備一次服務一個請求、無並行干擾、到達與 deadline 固定，使用獨立保留資料的 G 時間模擬實際完成。排程器只看預測值與已知佇列狀態，事件模擬器才看真實服務時間。

固定 dispatch 規則（例如選最早預測完成且能滿足 deadline 的可行設備），分別換入 M1、M3／M4、oracle，另列 round-robin／最短佇列基線。相同 arrival seed、負載與 deadline 設定下比較，先隔離「預測改善」的效果。

負載從低到高掃描，deadline slack 的範圍在 validation 設定後凍結。報成功在期限內完成的請求率、deadline miss、拒絕率、goodput，以及選錯設備的代價；不只報平均預測誤差。把 profiling 成本計入首次工作登錄或攤提成本，另做成本敏感度。

S0 不含傳輸、模型載入或排隊。若擴到端到端回應時間，使用 `R = queue + batching + transfer + load + host_prepost + service`，只在範圍互斥時相加；若有重疊需改事件模型。Near-RT RIC 的平台控制時標不能直接當成每個 AI 任務的 deadline，deadline 由具體應用定義。

通過回放後才做實機排程、多租戶、背景記憶體壓力及版本漂移。這些結果屬額外研究階段，不用隔離推論資料宣稱已解決干擾或真實 RAN 部署。

## 11. 與文獻的關係

- [Habitat，USENIX ATC 2021](https://www.usenix.org/conference/atc21/presentation/yu)：按 operation 使用 wave scaling 或 MLP 預測跨 GPU 訓練時間。借鏡來源執行錨點與分解預測；本研究改問缺少 operator semantics 時，trace 與硬體校正能保留多少預測能力。上述 C/M 模型是本研究提案，並非 Habitat 原式或其完整重現。
- [NeuSight，Forecasting GPU Performance for Deep Learning Training and Inference](https://arxiv.org/abs/2407.13853)：參考 tile 層級與效能約束的建模觀點。只拿到 Nsys 軌跡時，不能假定已知道其所需的工作量與硬體利用資訊。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CUDA trace、NVTX 與 GPU Metrics 的工具依據。Metrics 是取樣資訊；本計畫沒有宣稱四設備已有共同 per-kernel counter 表，實際支援需在節點盤點。
- [Clockwork，OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/gujarati)：參考受控執行條件下，以時間可預測性支援 serving 的系統設計。
- [Vidur，MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html)：作為後續 LLM 請求、prefill/decode 與模擬的參考。

Sia、CoreTuner 及訓練偵測 telemetry 論文保留為異質資源／監測背景，不作為目前 inference 預測能力的直接證據。研究不同點須靠受限資料條件下的精度、泛化、整合成本與排程效果實驗證明，不能只靠「把 wrapper 換成 trace」宣稱新穎性。
