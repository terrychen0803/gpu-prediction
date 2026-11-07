# 新 I3 40×3 協定草案與驗收

草案 ID：`o4_i3_config_context_40x3_v1_draft_20261107`。**尚未實作／執行，不能拿來標記既有 A0 或 seven-A 資料。** 第一次驗收前凍結非 draft ID、完整設定與 hash；任何量測語意變更另升版。規劃沿用已實測的 seven-A 穩定性邏輯，但受控 adapters、R 完整參數和 source snapshot 另接新 I3 collector。

## 一個 cell 與一個工作

cell = 節點 × configuration/context × mode × process_repeat。一個工作 GI/P/R 各啟動3個新 OS process／CUDA context，共9cells。不是同程序裡重設迴圈三次；各process記錄PID、UTC起迄、monotonic時間及exit code。

每個 process：讀取並校驗 frozen assets/source → correctness → 初始化 timing events → adaptive warmup → profiler start（P/R）→5 transition →40 measured →完成自己的 trace finalization →export／parse／audit。完整warmup與transition都保存，只有指定40筆進主要彙總。

warmup最低50、最高500，按20筆非重疊視窗檢查連續三個中位數，相鄰變動 `abs(m[j]-m[j-1])/m[j-1]≤5%`。因此最早實際通過點為60次。GI/P/R各自暖機；R暖機需另確認 dispatch 開啟狀態與measured一致，不能預熱純forward後假設R overhead已穩。到500未穩定即結束該cell，保留warmup和失敗，不偷偷降低門檻；可另建立明示未穩定的diagnostic run，不混入這個協定。

## 計時邊界與資產

- FP16、eval/inference_mode、eager；autocast/compile/CUDA Graph關閉。TF32、cuDNN benchmark/deterministic等旗標記錄並在版本凍結；不為匹配別台結果升級環境或暗改旗標。
- external tensors／weights／bias／BN stats在計時外載入並完成傳輸。每次prepare在timer外，同步清除前一筆尚未完成的工作；固定prepare政策與input layout。
- 預先建立並初始化CUDA start/end events。一對event包住**一次完整指定forward**，end.synchronize後取得 elapsed×1000 µs。沒有逐op event／synchronize，也不用40次總時間除40。
- GI無profiler、無逐op marker。P只有iteration/prepare層scope；R才有ATen leaf reference及完整bound arguments。`profiled_iteration_event_us`另存，不能覆蓋GI。
- 新controlled單算子的GI是這個「一算子工作」的event elapsed；它不是舊O1 GE，也不是DNN內CTX。新子圖GI只代表子圖，不是六任務完整forward GI。
- 只支援已驗證單stream／完成join的範圍。current stream不變不能證明其他stream不存在；audit需列actual stream分布與依賴。無法證明end前join則拒絕該GI完整性主張，保留診斷。
- P CTX=`max(end)-min(start)`（該op已映射GPU工作），另存sum durations、interval union、gaps、跨op overlap。event-index分段與時間重疊分開。禁止裁切、平移、補timestamp或移除異常。
- GB10明確指定並核對actual `cuda-sw`。Hardware問題0/80已有重現，不追加無新假說的flush/drain試驗。finalization以既有A0自己的context清理路線為候選，不直接繼承seven-A的sleep1s；需最小capture驗收。絕不做整卡reset／系統服務變更。

## Correctness（先凍結再收集）

單算子用同一組FP16 operand值轉CPU FP32 reference，shape完全相同、finite、relative L2≤0.02且 `max_abs_error/max(max_abs_ref,1)≤0.02`。不能只看finite。layout view需logical values一致、storage/stride符合清單且無意外mutation；BN running stats前後hash相同。

子圖另建立獨立CPU FP32 reference及逐節點round-to-FP16 reference，保存兩種誤差；第一版同樣要求上述兩個2%門檻對FP32 reference均通過。若累積數值誤差失敗，先定位節點與輸入分布，不自動放寬；若確有理由換門檻，建立新criteria版本並保留舊失敗。R開關須得到相同輸出；如backend非確定使exact不可能，先明列原因，另協定規定tolerance，不在原run中悄改。

CPU reference與輸出hash在capture／timer外。資產生成、prepare、correctness亦不混入被測forward；forward本來產生的copy／allocation仍保留，不為了讓scope變乾淨而移走。

## 排程、穩定性與失敗

同phase以repeat round執行，每round先用 `SHA256('61107|phase|repeat|work_id')` 排work，再依repeat輪換mode順序：GI/P/R、P/R/GI、R/GI/P。先寫完整schedule再執行。這減少固定模式順序偏差，不聲稱隨機化消除溫度／DVFS影響。不可看結果後重排挑快值。

每cell保存40筆、median、P10/P90（明定線性分位插值）、min/max、MAD、時間序列。每mode再取3個process median的median為配置摘要；spread=`(max-min)/median`，≤5%標stable，>5%標review_required。另列每process `(P90-P10)/median`，>20%作tail-review旗標。5%／20%是本輪工程診斷門檻，不是可信標籤或服務deadline保證。

spread不合格仍保存可用GI／P／R，不整筆刪除；三mode各自判定。先查GPU其他工作、warmup、trace overhead、錯誤、輸入／scope；必要時額外收**完整三process診斷批**，保留原批，不用較快補跑替換。失敗cell只有明確環境／工具錯誤修復後允許一次新attempt，連續相同錯誤停止該工作後續cells並繼續其他獨立工作；全部attempt分母保留。未完成3process，不假裝已40×3。

執行前／每cell前唯讀檢查GPU工作、utilization、可用記憶體；有外部compute owner則排隊／暫停，不終止它。保留guard開始結束與執行期間低頻host側活動記錄，監測開銷版本固定；不停止K3s／Pod／Netdata／RustDesk。每批最多12工作，磁碟先估算。正式排程不得自動擴到全部294工作。

## Reference與用途驗收

對每一個iteration分開核對45個scope（5transition＋40measured）、輸入與op序列、所有GPU activity→API、API→GPU reverse coverage、correlation scoped uniqueness、R→P映射與ATen ownership。Correlation ID須帶process/context/API domain，不能假設整份檔案全域唯一；有多對一／合法事件種類必須顯式處理。

P/R同一node、同一工作與repeat的事件結構應有唯一可解釋對應，**不要求P/R timestamp相同**，也不要求不同GPU kernel名稱／數目相同。若不一致不能拿R邊界硬套P；可以保留GI與inventory，逐iteration標記不能配對。matched coverage分母保留40，不只挑exact視窗。

`recorded_event_integrity`、`submitted_work_coverage`、`reference_semantic_alignment`、`timestamp_consistency`、`event_index_partition_supported`、`overlap_aware_closure_supported`、`strict_reference_verified`各自有證據。API成功無activity但沒有直接byte/no-op證據時，submitted仍unverified；不要用其他native實驗反推本capture傳入值。

**已知NVTX warning可以warning-accepted進探索；不等於解除strict gate。** GPU start早於CPU API entry只先否定CPU/GPU對時用途；若GPU域interval與API/ownership仍有效，可按項目保留分類／CTX的development用途。若GPU域duration為負、關聯多解、必要activity確定缺失，受影響CTX必須排除。Temporal overlap本身不是拒絕原因。用途詳見QUALITY_POLICY.json。

## 每批要交付

`protocol.json`、`criteria.json`、frozen config/DAG/asset hashes、`schedule.json`、`attempts.csv`、source snapshot逐檔SHA、environment、commands、GPU guard、warmup/iterations CSV、raw nsys-rep/SQLite/logs、R bound args、P allowlist、quality overlays、measured coverage、每mode stability、component equivalence與split audit、輸出manifest及繁中報告。

原始與derived分資料夾；run_id唯一；執行中不修改snapshot。不含credentials。Collector/parser改動要跑新測試＋原O4＋legacy，再用保存SQLite regression；舊141檔分發副本與raw不被覆寫。工程測試通過、GPU測得、development可用、strict verified四者分開。
