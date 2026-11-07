> 本文件保留規劃當時的工程待辦。後續 A0 Smoke 工具及固定資產已交付；新配置與正式 40×3 仍待實作，見 [目前節點手冊](../O4_I3_NODE_RUNBOOK_20261107.md)。

# 下一個執行回合與 RTX 節點交接

本文件是implementation backlog；除原A0工具已存在外，新增項目均待實作。現在clone main尚拿不到這套新I3 collector／assets。不要把本目錄JSON交給舊A0 frozen validator，也不要改其白名單來假裝支援新配置。

## GB10 下一步：先交 S0＋S1

| 次序 | 要改／新增什麼 | 必須看到的證據 |
|---|---|---|
| 1 | 從上一輪source獨立複製新工作版本；核心parser先不改 | base SHA、diff、run_id；舊raw／模型／snapshot不動 |
| 2 | 新受控asset schema、stride/layout建構、完整kwargs及12家族adapter | 實際shape/stride/values、bias與broadcast、group/dilation/padding、CPU parity；unknown明確拒絕 |
| 3 | 新I3 formal worker／scheduler | 讀取版本化protocol而非硬編碼3；40×3、45scopes、warmup上限、獨立PID、mode排程、failure/resume |
| 4 | source snapshot執行與不可變asset驗證 | 每個cell實際執行檔案的SHA；禁止收集期間修改live module |
| 5 | development用途overlay、每mode穩定性與coverage報告 | 不覆寫strict gate；可逐target拒絕；raw／warnings／unmatched全保存 |
| 6 | offline tests與raw regression | 新測試、原O4、legacy全通過；parser未改可重播代表性保存SQLite，若改映射須擴完整受影響案例 |
| 7 | GB10 GPU idle與disk preflight；三個sentinel smoke | BN/GEMM/Resize各1process GI/P/R，共9cells／27measured；錯誤記錄不刪 |
| 8 | 三sentinel的新40×3 | 27cells／1,080measured，6captures/work共18；品質、成本與所有失敗有報告 |
| 9 | 基於S1工程結果凍結release | 完整source/tests/protocol/asset manifest與二進位資產分發、portable指令、known limitations |

S1對應ID：`I3_batchnorm_eval_00_base`、`I3_gemm_00_base`、`I3_bilinear_resize_00_base`。與舊`A0_*`不同配置，不覆寫舊ID。新能力可以在同一個新工程run中依序驗收，不要求先重訓或等待RTX。

新adapter至少測：Conv groups整除與輸出幾何、bias/dilation、channels_last實際stride、GEMM/BMM轉置view非連續、Linear bias-off、Add broadcast、BN state不變、Resize align_corners、Concat軸與shape、in-place拒絕。DAG另外測多輸入skip／gate資料依賴、節點shape與weight shape、R parity、I/S中間值hash及計時外準備。

Collector測：缺asset/hash錯／unsupported、wrong GPU、外部owner忙碌、nonzero exit、只完成部分iterations、warmup500未穩、source snapshot被改、resume不重複成功cell、重試保留attempt、subset選取與未跑項目不填PASS。Report測：spreads分mode、40迭代分母、0B未知不升級、NVTX warning不一刀切、GPU missing拒絕受影響label、P-only禁止reference欄位、negative GPU interval、overlap union及跨split同signature。

目前`build_plan.py`只驗證設計與數量；不是這些runtime測試的替代，也不是原155項測試這輪重跑。實作GPU部分前要再查空閒狀態，本文未做GPU preflight，不能宣稱現在GPU沒有外部作業。

## 發布與對齊哪些資訊

先以獨立feature分支提交可review的source、tests、schema、protocol、矩陣、runbook、sample report及小型測試fixtures。不要直接改main或合併PR #7/#8。PR #8只文件且base依賴PR #7，後續整合需明示依賴；不要讓三台各clone不明確的latest。

完成S1後提供唯一release commit／tag及三項hash：source bundle、protocol/config manifest、asset manifest。大binary資產可以版本化release asset／受控傳輸，不把所有原始trace塞進Git；Git保存SHA索引、可取得位置、命令與摘要。若只有本地檔案還沒發布，就明說需要傳輸，不能提供不存在的下載指令。所有必要資產可校驗後才稱可攜包可用。

必須一致的是工作語意、外部input／weight bytes、精度、scope、協定與標籤定義。保存但不強迫一致的是CPU架構、driver、Torch/CUDA/Nsys版本、實際backend、kernel實作與事件數。不同設備無需逐模型全收齊；缺項不阻塞其他有效工作，也不偷偷換配置。

## RTX4090 與 RTX3090 操作順序

1. 在各自新目錄取得**已發布的明確release ref**，驗source、protocol、asset SHA；不在RTX重新seed生成「同一份」資產。先執行offline tests／plan-only。
2. 原A0 BN可以作舊包連線smoke，但不算新I3 40×3。新版本首先重現GB10同三sentinels的smoke，再各自40×3；actual backend、warnings及schema逐台記錄。不要求相同kernel名字。
3. 優先接同36個core_first配置；若某配置unsupported/OOM/忙碌，留下reason並繼續其餘。batch改小是新variant，不覆蓋ID。
4. 後續拿GB10 release中同一矩陣與DAG，依資源收train→validation→sealed test；不能因GB10哪項時間好看才選RTX測試集。Context D最優先；I/S為same-node對照，跨GPUbytes限制見CONTEXTS文件。
5. 交回source/environment/asset驗證、attempts、GI/P/R原始證據、quality/stability、錯誤、summary與完整SHA manifest。沒有連到RTX或收到證據，不能填遠端completed。

RTX4090最優先提供主predictor P及其R訓練／評分reference；GB10／3090提供目標GI與在允許用途內的P/R CTX supervision。第一版方便審查仍建議三mode全收；資源有限可分批收，不把缺R的GI整筆判無用。test target P/GI/R只給evaluator，收集完成後建立隔離路徑與封存預測流程。

**本輪沒有GitHub提交／新release、沒有RTX命令實跑，也沒有正式collector CLI。** 等S0程式完成且S1測過才寫可直接執行的實際命令，避免現在提供不存在的`--formal`參數。下列唯一現在可執行的命令只是重新產生計畫，不會啟動GPU：

```bash
python3 docs/o4_i3_collection_plan_20261107/build_plan.py
```

這會更新本計畫內的衍生JSON/CSV；發布後應在副本重現並比較hash，不修改已封存manifest。新收集使用另外的run目錄。
