# O4 首版資料語意

- `iteration_device_event_us`：GI無profiler單次native forward CUDA event interval，含區間內等待。不是純GPU busy或完整服務時間。
- `profiled_iteration_event_us`：P/R中的同一event計時；profiler與R instrumentation可能改變它，不能併入GI聚合。
- `contextual_trace_span_us`：R可觀測運算的first-start到last-end；`P_contextual_trace_span_us`是條件R→P映射後的P proxy。未解reference警告。
- `iteration_trace_span_us`：整個iteration的實際GPU工作first-start到last-end；與event邊界不同，不互換GI。
- `gpu_busy_union_us`：已記錄kernel/copy/memset的interval union。不是全設備利用率。
- `host_forward_plus_wait_us`：另存perf_counter host區間，含Python提交與等待，不能當GI。

所有duration為µs，原SQLite/allowlist timestamps為ns。UTC事件用ISO8601＋offset；排序與duration另存monotonic perf_counter ns。transition與warmup樣本分開，不計主要40筆聚合。

`quality.status=warning_accepted`表示已記錄結構可解析且保留警告；不代表無事件遺失或reference正式可用。無警告時仍是`structural_only_pending_review`。本版從不自動設定`formal_reference_verified=true`。

`composition_supported=false`表示觀察到重疊GPU工作，不能直接套非重疊SUM／舊模型假設。`reference_contiguous=false`表示部分ATen對應事件不連續，不能當舊semi-Markov真實片段。完整CPU reference與數值GI正確性另列，這些狀態互不替代。

`reference_transferred_to_P.jsonl`的類型、shape、運算數及邊界屬oracle。P-only匯出只有數值事件、單stream ID、已知iteration範圍與來源hash。任何後續predictor均不得掃raw或整個quality目錄取得額外資訊。

`old_three_family_*`僅為GEMM/BMM/Conv family關聯比例。RN-A的Conv shape不一定在O3訓練範圍內；addmm/Linear帶bias不默認GEMM。實際適用性需另做signature稽核。本版沒有將任何RN-A片段送入舊N0後宣稱支援。

所有inventory與family百分比以成功R→P匹配的iteration為條件；`pair_summary.json`與attempt inventory同時揭露未匹配／失敗分母。跨類別busy union可能重疊，各類比例和可超過100%。一個configuration×direction仍是一個研究案例，不把40樣本、三process或GPU事件數當獨立配置。

此處ATen「leaf」指Python dispatch攔截層的非巢狀單位；實測有`aten.conv2d.default`、`aten.batch_norm.default`、`aten.linear.default`等複合呼叫，C++內部不再重複拆成所有primitive。它是明定可觀測運算單位，不是任意Python原始層完整還原。`static_module_inventory.json`另存CPU建立模型的Conv stride/padding/dilation/groups/bias等參數及TorchVision source hash，只供evaluator；module registration順序不是動態執行順序，functional add仍以R清單為準。
