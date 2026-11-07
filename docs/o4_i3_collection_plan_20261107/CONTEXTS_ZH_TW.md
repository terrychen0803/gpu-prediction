# Context 的具體控制與後續擴充

本文件及CONTEXT_PLAN.json全為planned_only，沒有新增GPU實測。張量依賴必須由forward實際傳遞，不只按固定順序執行幾個互不相干函式。

## 核心六模板

共同三尺度 `(B,C,H,W)` 為 `(1,20,36,36)`、`(2,40,24,24)`、`(1,80,12,12)`，分配train／validation／exploratory test。這些是新受控尺寸，刻意與核心isolated矩陣分開；不是擷取的DNN原層。每個模板保留完整JSON DAG，不將parent module與leaf ATen重複算標籤。

| 模板 | 實際工作與輸出 | 每圖 isolated 位置數 | 回答問題 |
|---|---|---:|---|
| conv_bn_relu | C→C 3×3/s1/p1 Conv(no bias)；BN eval eps1e-5；out-of-place ReLU；輸出[B,C,H,W] | 3 | 加BN／activation後事件分段與CTX怎樣變 |
| residual | Conv→BN→ReLU→Conv→BN→Add(x)→ReLU；兩Conv均C→C 3×3/s1/p1，skip不投影 | 7 | 真正skip資料依賴與末端Add是否影響span／overlap |
| depthwise_pointwise | DW3×3/groups=C→BN→Hardswish→1×1 C→2C Conv；輸出[B,2C,H,W] | 4 | depthwise、hard activation、pointwise的不同實作與長度 |
| se_gate | pool到1×1；C→C/4 bias Conv1×1→ReLU→C bias Conv1×1→Hardsigmoid；乘回原x | 6 | reduction後小算子、broadcast mul與SE型context |
| resize_concat_conv | bilinear到2H×2W；concat外部skip[B,C,2H,2W]；2C→C Conv3×3→ReLU | 4 | resize分類修復、concat/copy與後繼Conv |
| attention_eager | batch=B×4 heads，D32，L=32/64/128；Q[B×4,L,32]×Kᵀ[B×4,32,L]→乘1/√32→softmax(-1)→乘V | 4 | BMM與softmax/scale互相依賴；不是fused attention，也不含mask／cache |

每尺度共28個node positions，三尺度84。Graph 18＋independent sequence18＋isolated84，共120工作。部分位置的static signature相同，不能以84宣稱84個獨立config。BN weight/bias/mean/variance shape均[C]；SE各bias依輸出channel；所有參數與外部tensor必須凍結為資產，JSON中的weight_shape不是已存在的weight bytes。

## 三個條件怎樣公平比較

**D（dependent）**：真實中間tensor直接傳入下游；Residual兩條input、SE gate及skip都按DAG。禁止detach到CPU、clone中間tensor或加逐op同步。R只在獨立模式取得參數與ownership；計時外的correctness執行可保存中間輸出，不能把記錄tensor的hook留在計時forward。

**I（independent sequence）**：計時外先執行一次同節點D，取得每個位置所需輸入；在scope外保存／重建其exact tensor值與stride。測量時逐位置用這些固定輸入，按D相同順序、同stream提交，保留所有輸出到whole-forward end。不依賴前一位置本次輸出。這是斷資料邊的控制，不是無序並行。預先輸入、allocation與輸出存活期可能改cache，因此D−I是多因素診斷，不能直接歸因PDL或單一依賴成本。

**S（isolated）**：每個node position單獨一個工作，使用該節點I同一組預先輸入與參數。沒有把D每個op額外插CUDA event。S GI是單位置控制的完整event elapsed；D中的該op P CTX是profiler label，兩者不得相減後稱純context causal overhead。優先同模式比較P CTX比率，另報whole D/I GI及sum(S GI)，清楚標示profiled/unprofiled差異。

I/S中間值來自該node的D正確性run，各GPU可能因捨入產生不同bytes；應保存實際hash、數值差與scope，**不能把跨GPU I/S自動當成exact-byte成對資料**。主跨GPU可比對象首先是相同external assets與運算定義的D，以及核心isolated固定資產。若需要I/S也做exact-byte跨GPU預測，另凍結一套共享intermediate assets、新variant；不可悄悄替換這個same-node控制。

先比對node清單、logical inputs/outputs、runtime參數、是否真的保留依賴、input mutation、BN狀態和R開關parity。eager不保證各GPU或ATen路徑完全相同。若產生fused／split差異，使用能被兩端共同定義的logical group或保留unmatched，不按op_index硬配。沒有GPU activity的view單列，未知ownership不補0時間。

## 分段與時間分析交付

- 按配置／模板／類別列events、kernel/copy/memset種類、events/op的分布、L4/8/16/32超界、zero recorded與unknown owner。
- 列same-stream／cross-stream overlap，sum、union、span、positive gaps。相同事件不能被兩個op重複分配；時間重疊本身允許。以union作closure診斷，不假設op durations可直接相加等於GI。
- 在P-only事件索引順序上評估是否仍有唯一連續分段；非連續／多stream interleaving另列unsupported，不能裁掉難例。
- D/I/S逐process分布及GI/P/R穩定性獨立報；R時間不代P CTX。最先交CBR與Residual train尺度的實測報告，再擴其他模板。

## 重要但分批處理的其他 Context

以下具體探針是**候選第二批**，不包含在294工作／2,646cells上限內；先在核心結果與I2每任務busy-union覆蓋中決定優先，另生成manifest再收。一次只選最有價值的一類，不能用它們冒稱已完成六DNN。

| 探針 | 明定的第一個設定 | 尚需確認 |
|---|---|---|
| 串接GEMM | [64,96]×[96,128]→乘[128,48]，無bias／activation | 中間輸出確實作第二輸入；與A0 GEMM不同asset |
| 串接BMM | [4,32,64]×[4,64,48]→乘[4,48,24] | bmm與broadcast matmul分開 |
| YO型Conv塊 | [1,32,80,80]，3×3 C32→64→BN→SiLU | 只是可追溯conv塊，不代表YOLO26n one-to-one head／top-k |
| Swin型MLP | [1,49,96]，LayerNorm eps1e-5→Linear96→384→GELU→Linear384→96 | 不含window shift/pad/attention；exact norm/gelu版本固定 |
| RMSNorm/gating | [1,128,1024]，RMSNorm eps1e-6；gate/up 1024→3072→SiLU×up→down1024 | 作受控算子圖，不能把這些尺寸宣稱實際Qwen3配置；與已安裝model config另核對 |
| Prefill/decode attention | 先用4 query heads、4KV heads、D32、context128；prefill Qlen128，decode Qlen1分開；eager、固定causal mask | 這是通用attention探針，非Qwen原模型GQA；實際Qwen需讀config固定heads/rotary/cache adapter |
| KV update＋RoPE | 先核對現有Qwen實際cache類型、position/rotary參數與2B copy來源，再設計單步append | 每iteration計時外重建同cache長度；不以連續generate替代固定重複；未定位copy不擅自提供假對照 |

Bilinear resize、concat及hard activation已由核心列入；其他非連續copy、LayerNorm、SiLU、RMSNorm等加入時要擴exact taxonomy與unknown分支。不能把一個候選探針的一次成功當足夠該類訓練樣本。
