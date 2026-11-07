> 發布註記：下文為規劃回合的狀態；後續已上傳 A0 Smoke 原始碼／資產，新 174 配置、子圖與 I3 40×3 仍待實作。當前可執行指令見 [節點手冊](../O4_I3_NODE_RUNBOOK_20261107.md)。

# I3 下一輪具體收集計畫：GB10 先行，RTX 節點重用

2026-11-07，Asia/Taipei。**本輪完成規劃與離線一致性檢查；沒有新增 GPU 實測。** 本計畫依使用者授權，先在 GB10 驗證共用程式與協定，再交 RTX4090／RTX3090 依同一工作定義收集。最終主預測方向仍是 **RTX4090 P → GB10／RTX3090 GI**。

## 1. 已核對的來源與本輪決策

GitHub 最新研究文件位於 [Draft PR #8](https://github.com/terrychen0803/gpu-prediction/pull/8)，head `1711a5f1d27c282d640218611c01d9948c13cbd6`，base 為尚未合併的 PR #7 統一工程分支。`main` 仍是 `b204ae63ee2dff2bbc161bb5cc8f9aaecfb68c01`。本輪已讀最新 [務實路線](https://github.com/terrychen0803/gpu-prediction/blob/1711a5f1d27c282d640218611c01d9948c13cbd6/docs/o4_i3_progress_20261107/ROADMAP_AND_PRAGMATIC_DATA_POLICY_ZH_TW.md)、狀態、I0～I7 主計畫、工作負載矩陣與收集政策；五份快照保存在 `upstream/`。

本地依據為 [I3 實作報告](../o4_i3_progress_20261107/REPORT_ZH_TW2.md)、最終驗證及 worker／controlled／splits 與既有 seven-A formal worker 程式。本輪閱讀既有檢查結果，不冒稱重新重播全部 raw。現有 GB10 九種 A0 各一個 a 配置，選定 27 cells／81 measured；18 個資產配置不是 18 個已實測配置。155 項測試與 319 個重播產物是上一輪成果。Strict Reference 仍為 0。

**採用最新務實政策**：新增用途審查允許可追溯、特定標籤可用的資料做 development 研究。NVTX warning 不再阻擋所有探索；已知缺失 activity、錯誤 ownership 或不合法時間標籤仍不能當真值。本輪不改寫歷史 `formal_training_eligible=false`。本文「正式 40×3」指完整的版本化收集協定，**不代表 Reference verified 或正式模型結論**。

我的優先順序是：先提高配置資訊量與量測可追溯性，接著收 Context 對照，再決定 I4 模型。繼續堆同一配置的 repeats、先訓新 MLP，或現在重收全部七 A，都不是這一輪最有價值的工作。

## 2. 第一條線：174 個受控 Operator 配置

完整逐筆尺寸、tensor stride、layout、參數、輸出 shape 與 split 在 [OPERATOR_MATRIX.csv](OPERATOR_MATRIX.csv)、[OPERATOR_CONFIGS.json](OPERATOR_CONFIGS.json)。這些是新設計的 controlled benchmark，不宣稱逐筆等於既有 DNN runtime 配置，也不使用模型預訓練權重。

| 配置家族 | 12 個基礎 anchor 的範圍 | 額外控制，均與 anchor 00 同群 |
|---|---|---|
| Conv | B=1/2/4，Ci=16～320，Co=2Ci，H=W=112～14；3×3、s1、p1、g1 | stride2、channels_last、1×1/p0、dilation2/p2、groups4、batch4、padding0、bias=True |
| Depthwise | 同 B/C/H anchors；3×3、groups=C、Co=C | groups=C/2 對照（語意是 grouped Conv）、channels_last |
| GEMM | M=32～640，N=48～768，K=64～1024，12 個明列三元組 | 右矩陣轉置 view；左矩陣轉置 view；logical shape 不變 |
| BMM | 相同 M/N/K anchors，batch=4/8/16，無 broadcast | 右／左矩陣非連續 view；維持相同數值 |
| Linear | [B,M,K] → [B,M,N]；固定 bias=True | 移除 bias；輸入非連續 view |
| BN eval | [B,C,H,W]；eps=1e-5、固定 running stats | eps=1e-3；channels_last；不跑 training BN |
| ReLU | [B,C,H,W]，out-of-place | anchor00 batch1→2；channels_last |
| Add | 同 shape 的兩輸入，alpha=1 | other=[1,C,1,1] broadcast；channels_last |
| MaxPool | k3/s2/p1，ceil_mode=False | s1；channels_last |
| AdaptiveAveragePool | [B,C,H,W]→[B,C,1,1] | output2×2；channels_last |
| Bilinear Resize | 指定 output=2H×2W，align_corners=False，antialias=False | align_corners=True；channels_last |
| Concat | dim1，[B,C,H,W]+[B,C/2,H,W] | dim2 串接 H 與 H/2；channels_last |

共 12 家族 ×12 base=144，另加 30 個對照=174。一個家族有 12 個**設計群**，不是足以證明統計泛化的樣本數。控制組不增加獨立群數。配置表的 `family` 是收集分類；實際 class 須以運算語意與 R 審查，例如 groups=C/2 不能硬標 depthwise。

基礎 anchors 同時改動工作量，用來涵蓋範圍；它們不是單因子因果實驗。anchor00 對照才用於固定其餘條件比較；kernel/padding、dilation/padding 為保持輸出空間大小的成對設定，Concat 軸對照亦改 operand shape，應明示。Batch、layout 等仍同群，不把兩個 view 當兩個獨立測試。

channels_last 保持 logical NCHW shape；不是把 shape 誤改成 NHWC。轉置 view 是從反向尺寸的 backing storage 建立所需 logical shape，禁止偷偷 `.contiguous()`。保存實際 strides／storage offset；可觀測 copy 也保留。資產尚未產生，實作時 CPU 產生一次，FP16 bytes 與 logical-value hash 凍結，再送 RTX；不同 storage 佈局使用相同 logical values。預設輸入小幅隨機、BN variance 正值，明示 initialized_performance_only。

## 3. 第二條線：六種 Context、三個尺度

[CONTEXT_PLAN.json](CONTEXT_PLAN.json) 定義 18 個有依賴的 DAG（有向無環圖），每個 node 有輸入邊、shape、參數與輸出 shape；詳細對照在 [CONTEXTS_ZH_TW.md](CONTEXTS_ZH_TW.md)。

1. Conv→BN(eval)→ReLU。
2. 兩層 Conv/BN 的 residual block，加原輸入 skip 後 ReLU。
3. Depthwise→BN→Hardswish→Pointwise Conv。
4. SE：global pool→reduce 1×1 Conv→ReLU→expand Conv→Hardsigmoid→broadcast mul。
5. Resize→skip concat→Conv→ReLU。
6. Eager attention 核心：QK BMM→scale→softmax→AV BMM。

每個子圖比較三個條件：真實依賴執行、使用預先準備中間值的斷依賴序列、逐位置 isolated。三者都保留 GI/P/R；GI 包住整個控制工作，P CTX 為 operator 的 profiler span，不能互換。斷依賴仍用同 stream 串行提交，不是並行 benchmark，也不能宣稱只移除了唯一物理變因。

總計 18 dependent＋18 independent sequence＋84 isolated positions=120 個工作項。isolated positions 可能有相同配置但不同 operand，不當作額外 signature 群。先做 train 尺度的 CBR 與 Residual，再推其餘四種；不一次執行 120 項。

這六種控制涵蓋 RN/MV/UN 與 attention 的重要因素，不能宣稱已完成 YOLO neck/head、Swin window shift 或 Qwen cache 工作。後續候選與啟動依據另列在子圖文件，避免用相似子圖偷換原模型範圍。

## 4. GB10 執行批次與成本上限

**下一個執行回合先做到 S1，再交一份結果供後續擴充判斷。** 下列全部是規劃，未執行。

| 批次 | 實際工作 | 正式 40×3 cells／measured（單節點） | 完成後判斷 |
|---|---|---:|---|
| S0 工程 | 新 collector、資產、單元測試、CPU shape/value 檢查、dry-plan；保留舊 A0 | 0 | 版本凍結與失敗路徑可用 |
| S1 協定驗收 | BN、GEMM、Resize 的 anchor00；先各 GI/P/R 1×3 smoke，再各 40×3 | **27／1,080**；smoke 額外9／27 | 觀察已知 warning、GEMM overlap、resize 映射；新版本是否值得擴大 |
| S2 第一批 | 12 家族 anchors 00、03、04 共36配置，扣除 S1 已收3項 | **新增297／11,880**；累計324／12,960 | 有多個 train 配置，先看 per-family coverage 與成本，不報未見配置 test |
| S3 配置擴充 | 其餘108 base＋30對照；先train，再validation，最後封存的 exploratory test | **1,242／49,680** | 12 groups/family 的設計池是否真的可用；分開報缺項 |
| S4 Context | 三尺度的120項，按模板及 split 分批 | **1,080／43,200** | Context 時間、依賴對照與分段可識別性 |

若全數完成：294工作 ×3 modes ×3新程序=**2,646 cells／105,840 measured**，其中 **1,764份 P/R captures**。這是容量上限，**不是本輪必須跑完的配額**。不含 smoke、失敗、補跑或候選擴充。S1 不重複計入 S2；舊 A0 不湊入新 40×3。每工作每模式 120 measured，全模式360；120個 iteration 仍是一個配置。

S4 可以在 S2 後先插入 CBR／Residual train 尺度（3+7個 isolated、2 dependent、2 independent，共14工作、126 cells），檢查 context 是否有新資訊，再決定 S3/S4 完整排序。先不要拿 test 結果決定收哪種變化。

不虛構小時／硬碟估計。S1 保存 capture、export、parse、CPU correctness、暖機各階段實際秒數及每 capture bytes；用 P90 單工作成本估下一批，加 25%容量餘裕，磁碟需同時容納 raw+SQLite+derived。每批上限先設12工作，空間不足或外部 GPU 工作出現即保留進度暫停。每capture的5 transition與40 measured分開保存，暖機不算measured。

## 5. 怎樣使用結果

- 先回答：哪些類別因 shape/layout 改變而切換 kernel 分解；哪些含非 kernel/copy、長片段或未知 ownership；是否需要擴 taxonomy 或候選 L。
- 再回答：同一 static signature 在 isolated／dependent context 下 P CTX、union、gaps/overlap 及整體 GI 改變多少。不要拿 isolated event elapsed 當 CTX 真值。
- GI-only 可用就做 GI 描述；分類可用而 CTX 不合法就只做分類；缺少遠端配對仍保留單節點資料。用途按 [QUALITY_POLICY.json](QUALITY_POLICY.json) 判定。
- I4 可先作已標註限制的 oracle-boundary 分類與小型模型探索。正式主方向 N0 配對至少還需要 RTX4090；GB10先收不表示已完成跨設備模型。
- I5 ITER 必須另有有效 grouped OOF／校正資料；本輪不訓練辨識器、N0 或 ITER，不以收集筆數直接宣稱 OOF 充足。

## 6. 本輪交付狀態

| 項目 | 狀態 |
|---|---|
| 最新 GitHub 與本地程式／既有結果核對 | 已完成閱讀，來源快照已保存 |
| 具體矩陣、DAG、分組、收集協定、用途 gate、節點交接 | 已完成規劃，機器可讀表已產生 |
| 設計數量、基本尺寸／group、拓撲、split 一致性 | 已完成 CPU 檢查；不是 GPU correctness |
| 新 collector／新 adapters／新 assets／runtime collision audit | 待實作；JSON 不可直接交舊 A0 CLI |
| 本輪新增 GPU／RTX 執行、模型訓練 | 0 |
| Reference strict verification、正式訓練資格 | 未取得；歷史旗標不變 |
| GitHub 程式發布 | 本輪未推送；PR #8 仍主要是文件，不是完整可 clone 的 I3工具包 |
