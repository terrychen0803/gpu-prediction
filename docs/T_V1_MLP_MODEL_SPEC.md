# T-v1：以 MLP 辨識算子並預測跨設備時間的數學規格

更新：2026-10-21。**設計規格，沒有實作／訓練結果。** 使用者指定以 MLP 為主要模型；線性／倍率模型只作對照。實作與 GPU 量測在 RTX 3090、RTX 4090、GB10 節點。本文件細化 [O 系列設計](NEXT_EXPERIMENTS.md) §6～7；資料隔離與時間定義依 [資料契約](OPERATOR_DATA_CONTRACT.md)。數值超參數是可實作的初始方案，需 validation 選擇，不能稱最優值。

## 1. 一頁摘要：學什麼、輸入什麼、輸出什麼

```text
來源 GPU 的 P trace，已知 request 窗口、未知 op 邊界
  ↓ 清理與 API correlation，保留 GPU event 序列
  ↓ 列舉連續候選片段，萃取固定長度特徵
  ↓ MLP-S：每個片段屬於各算子類型的分數
  ↓ 動態規劃：找不重疊、不漏事件的完整分段
  ↓ 每片段類型機率、分段可信度、拒絕標記
  ↓ MLP-F：來源 trace 時間校正（僅 GE 時間目標）
  ↓ MLP-V：依片段、類型及硬體估計跨設備時間倍率
  ↓ 各算子目標時間、coverage、不確定性
  ↓ O4 的 MLP-A：完整 request 服務時間（另階段／另標籤）
```

MLP 學習數值函數，動態規劃施加序列結構限制；因此仍是以 MLP 為核心的模型。T-v1 不要求從 trace 完整逆推出 shape。真實 shape 只提供給 W-BCS 對照；可預測類型已構成 operation-aware 表示，但能否接近 shape-aware 方法需測試。

初始分類集合 C={GEMM,BMM,Conv2D}；擴充資料後再加入 Add、LayerNorm。每版 taxonomy 固定，不能將不在集合的工作硬歸成已知類。第一版採拒絕機制，而非假裝訓練過所有 unknown。

## 2. 記號、任務與適用範圍

| 記號 | 定義 |
|---|---|
| s,t | 來源／目標設備 |
| X_s=(e_1,…,e_n) | 一個 request 的來源 GPU 工作事件序列 |
| h_d | 設備 d 的獨立硬體校正向量 |
| Y={(a_i,b_i,c_i)} | reference 算子分段及類型，僅訓練／評估可見 |
| B=[a,b] | 連續事件候選片段，長度 ℓ=b−a+1 |
| z_B | 從來源片段取得的特徵，不含 wrapper metadata |
| q_B(c) | 在指定片段邊界下的類型機率 |
| t_s,t_t | 同一 logical operation 的來源／目標時間，標籤種類須相同 |
| τ | 固定時間單位 1 µs，使 log(t/τ) 無量綱 |

首版限定單 process、單工作 stream、eager、固定精度／layout、穩態、無背景共置、無融合／graphs。事件以可稽核 launch order 排序，驗證與單 stream 執行順序相容；不能只按 kernel 名稱排序。API 供關聯，不當另一個 GPU 工作事件重複計數。

一個 reference op 的 GPU events 必須形成連續區間；若有非連續交錯／多 stream／無法歸屬事件，標成超出首版範圍，保留失敗與 coverage，不能靜默重排。無 GPU 工作的 view 不能從 GPU 序列識別，另列。不以 CPU NVTX 時間相交取代 API→GPU correlation。

### 三種時間任務

1. **O1-GE**：獨立算子無 profiler 的 CUDA event elapsed。MLP-F 學 trace→GE 校正。
2. **O3-CTX**：混合序列中，目標 P trace 的真實算子 GPU span；受 profiler 影響的 proxy。來源亦是 P span，故此模式固定 F=0，不多學一個不必要的來源校正。
3. **O4-G0**：完整 request 無 profiler 的 host 完成時間，用 MLP-A，另有訓練集與標籤。

GE 模型與 CTX 模型使用不同 artifact/target_kind，不能把兩種 label 混入同一 loss。O3-CTX 的成功只證明上下文 trace proxy 的預測，最終服務時間目的還需 O4-G0。真實 DNN 內「無干擾的每算子時間」目前沒有直接真值，不能以獨立重播替代後宣稱量到。

## 3. 事件與片段特徵

### 3.1 原始量與時間間隙

事件 j 的 GPU start/end 為 u_j,v_j，duration d_j=v_j−u_j>0。單 stream 下內部間隙：

$$
g_j=\max(0,u_{j+1}-v_j).
$$

候選片段的 span、busy union、gap fraction：

$$
D_B=\max_{j\in B}v_j-\min_{j\in B}u_j,\quad
U_B=\left|\bigcup_{j\in B}[u_j,v_j]\right|,\quad
G_B=1-\frac{U_B}{D_B}.
$$

重疊時仍用 union，不直接加 duration；非預期 overlap 另做品質診斷。D_B=0 不進正時間模型。時間原始保存 integer ns，轉 µs 後才做 log。gap 不代表算子邊界，不能只用固定 gap 門檻當真值。

### 3.2 固定特徵 schema

首版 feature extractor 必須輸出同一欄位順序及 schema hash：

| 特徵群 | 明確定義 |
|---|---|
| 規模／時間 | ℓ、D_B、U_B、G_B；duration 的 min/median/max/mean/std；內部 gap 的 sum/max/mean；ℓ=1 時 gap=0 並附 empty-gap mask |
| 執行配置 | grid 三維乘積及 block 三維乘積的 min/mean/max；dynamic shared-memory bytes 的 min/mean/max；每群附 available fraction |
| 事件組成 | kernel/memcpy/memset 比例、實際可見的 copy bytes 總量；CPU API 平均提交時間及 availability mask |
| 局部結構 | 首／末事件 duration、前後外部 gap（捕獲邊界用 mask）；相鄰事件 kernel-name 相同比例 |
| 名稱（可選） | 每個 kernel 名稱依 ASCII 非字母數字分詞，保留數字；SHA-256 固定映射至64桶的 token count，除以片段總 token 數；首／末事件再各64桶，共192維 |

名稱 hash 與 dictionary 不含 config、NVTX、Python stack。保留數字可能提供實作尺寸線索，屬 name-enabled 組；name-free 組移除整個名稱特徵及名稱相同率。不能說使用名稱時完全沒有語意資訊。hash 不能用每個程序隨機 seed 的內建 hash。

數值變換：非負有單位 x 用 log(1+x/u_x)，u_x 固定為1µs、1byte或1count；比例不取 log。再用 training fold 的 mean/std 標準化，std=0 時設1並標記 constant。缺值以 training median 代入並附 mask；若整欄 training 都缺，移除該欄並固定 schema。測試出界值保留，另報 out-of-range，不依 test 重新 fit。ID、設備名稱、模式名稱、檔名、true op count 不輸入 MLP。

時間 MLP 使用上述**數值特徵子集**及 q/h，首版不再次輸入192維名稱 hash，降低小資料過擬合；名稱可透過辨識機率影響時間頭。直接加名稱給時間頭是另列消融。

### 3.3 Metrics 擴充 TM

對 O0 確認的共同 metric，每片段／合法上下文窗口提供 mean/std、有效採樣數、coverage 與 missing mask。同一次 capture 時間軸才可對齊。短窗口不足至少5有效採樣或候選80%覆蓋時標 missing，門檻凍結前須確認工具定義。不能將跨次 PM 值貼到 P；P/T 與 PM/TM 各保留收集干擾。

TM 增加同樣欄位至 MLP-S、F、V；T 保留完全無 metrics 的 artifact。這不改變真實邊界／類型不可輸入的限制。初版不需 metrics 即可實作。

## 4. O2：MLP-S 的分段與分類模型

### 4.1 候選區間

列舉所有 1≤a≤b≤n 且 b−a+1≤L 的區間，不依 test 真實 op count 篩選。L 初值為 training reference 最大正長度，至少1；只允許用 validation 檢查覆蓋後選較大值。若成本過高，在 pilot 選操作範圍或版本上限並凍結，不能默默截掉長算子。L 對候選 recall 的上限要報告。

### 4.2 MLP 的輸出是片段分數

設 K=|C|，兩個 hidden layers：

$$
r_B=\mathrm{ReLU}(W_2\mathrm{ReLU}(W_1z_B+b_1)+b_2),
\qquad A_\theta(B,c)=(W_3r_B+b_3)_c.
$$

初值 hidden widths=64,32，output=K，不在這裡先作 softmax。每類的 A 是「這段作為該類算子的結構分數」，不直接等於機率。允許相鄰同類算子，不能規定同類必合併。首版不加類型轉移矩陣，避免過度記住人工序列順序。

對完整合法分段 Y：

$$
S_\theta(X,Y)=\sum_{(a,b,c)\in Y}A_\theta([a,b],c),\qquad
p_\theta(Y\mid X)=\frac{\exp S_\theta(X,Y)}{Z_\theta(X)}.
$$

Z 對所有完整分段及類型組合求和。這是以 MLP 作勢函數的 segmental conditional model；借鏡 semi-Markov CRF 的片段與動態規劃形式，不宣稱該方法已證明適用 GPU。

### 4.3 訓練損失

使用有完整、可靠 reference 的 training request：

$$
\mathcal L_S=\frac{1}{N}\sum_{r=1}^N
\frac{\log Z_\theta(X_r)-S_\theta(X_r,Y_r^*)}{n_r}
+\lambda_S\|\theta\|_2^2.
$$

每 request 除 n_r，減少長序列主導；相同模板的 repeats 平均後再等權模板取樣，不能把 repeats 當獨立資料量。候選跨界片段已出現在 Z 的競爭路徑中，不需把它們任意標成某個算子來訓練。

先在 isolated 的已知完整片段用 cross-entropy 預訓練相同 MLP-S 是可選 warm start；它不能取代 mixed-sequence 的結構損失，因為 isolated 沒有學習切點。沒有完整 reference 的序列第一版不進此 loss，不臨時編造 partial labels。

### 4.4 Forward 與最佳分段

用 log-sum-exp（LSE）避免 overflow：

$$
\alpha(0)=0,\quad
\alpha(b)=\operatorname{LSE}_{a=\max(1,b-L+1)}^b
\operatorname{LSE}_{c\in C}\{\alpha(a-1)+A_\theta([a,b],c)\}.
$$

log Z=α(n)。推論把 LSE 換成 max，保存 a/c backpointer：

$$
V(0)=0,\quad V(b)=\max_{a,c}\{V(a-1)+A_\theta([a,b],c)\}.
$$

從 V(n) 回溯得到 B_hat、c_hat。在片段分數已計算後，DP 成本 O(nLK)；特徵／MLP 前向成本另外量，不把總成本也直接稱 O(nLK)。多種完整分段的資訊可保留，不只輸出硬標籤。

### 4.5 機率與拒絕

Backward 設 β(n)=0，β(k)=LSE_{b=k+1…min(n,k+L),c}[A([k+1,b],c)+β(b)]。片段含類型的 posterior：

$$
p(B,c\mid X)=\exp\{\alpha(a-1)+A(B,c)+\beta(b)-\log Z\}.
$$

指定 B 存在時的類型機率 q_B(c)=softmax_c A(B,c)；片段可信度 p_B=Σ_c p(B,c|X)。q 高不表示邊界可信，必須同時看 p_B。內部切點 j 的 posterior 是所有結束於 j 的片段 posterior 和。

**首版固定temperature T=1**，避免OOF與部署機率尺度不同。選 (τ_seg,τ_cls) 使 calibration joint precision 達預定0.90且 coverage最大；同時報樣本數及precision區間，若不可達或支持數太少，報 gate 失敗／證據不足。正式標籤：p_B<τ_seg 或 max q<τ_cls 時 reject。後續temperature消融可取 {0.5,1,2,4}，但須將全部 A/T 後重算α/β/q並依§8重建OOF，不能只改分類softmax。

MLP 可能對未知類過度自信；上述不是 open-set 保證。必須加入未參與訓練的 unknown/challenge 測試，與 precision/coverage 一起報。首版拒絕片段不強配時間類型，也不刪掉後宣稱完整 request 預測；保留 partial/incomplete 標記。通用 fallback 模型若新增，另列方法與成績。

## 5. 硬體表示：使用實測校正，不以 GPU 名稱查倍率

初版 h_d 的候選固定6個正值：SM數、E2 FP16 GEMM 128³及2048³的有效吞吐、固定256MiB copy 的 same-buffer及rotating_4_buffers有效頻寬、empty-kernel inner_n=1000／batched 的CUDA-event每呼叫時間。每值記精確 cell key、單位、protocol、median/spread與品質。對應 E2 cell 若缺失／不穩定，先用 training/pilot 決定共同可用較小集合，不能對 test GPU 偷換尺寸補值。

**與現有E2程式的對接：**目前summarize對launch使用host_complete_us，不能直接拿其median當CUDA-event時間。新hardware-view從原始microbench_samples.csv的gpu_event_us/(repetitions×inner_n)先取每process中位數，再取三process中位數，另算event側spread並驗證品質；這是待實作的衍生資料，不修改舊E2結果。copy頻寬依現有work_bytes=2×size，以相同logical_byte/s轉成GB/s，不冒充DRAM流量。

h 各值先除固定單位（SM=1、compute=1 TFLOP/s、memory=1 GB/s、time=1µs）取 log，再使用 training 設備統計標準化。held-out GPU 的獨立校正值合法，但不參與訓練 normalization。CPU launch／sync／transfer 另放 host profile，僅在服務時間模型使用。batched empty-event 耗時仍可能含提交供應不足，不能稱硬體純 launch latency。

高維 E2 曲線的擴充另列，不讓幾千次重複樣本掩蓋實際只有三個硬體點。h 中不放設備 one-hot 或 target workload time；已知設備 pair-ID 模型可作比較但不能稱未見設備模型。

## 6. O3：MLP-F／MLP-V 的跨設備時間模型

### 6.1 來源時間校正

GE 模式使用一個共享的小型 MLP-F，numeric z 與 h_s 為輸入，hidden=32,16、output=1：

$$
\widehat y_s=\log(D_B/\tau)+F_\eta(z_B^{num},h_s),\qquad
\widehat t_s=\tau\exp\widehat y_s.
$$

以 Huber loss 擬合 y_s=log(t_s^GE/τ)：

$$
\mathcal L_F=\sum_w\omega_w\,H_\delta(\widehat y_{s,w}-y_{s,w})+\lambda_F\|\eta\|_2^2.
$$

Hδ(e)=e²/2（|e|≤δ），否則 δ(|e|−δ/2)；δ初值0.2（自然對數單位）。ω 按 workload 等權、設備/repeats 內平均。CTX 模式直接設定 ŷ_s=log(D_B/τ)，不訓練 F。

### 6.2 硬體條件化 MLP-V

令 u_B=[z_B^num,h_s]，共享 MLP-V 接收 [u_B,h_d]，hidden=64,32、output=K（每類一個輸出頭，最後linear不設bias，因它在差分中抵消）。對 source/target 用**同一組權重**：

$$
\Delta_c(B,s,t)=V_\psi(u_B,h_t)_c-V_\psi(u_B,h_s)_c.
$$

每類預測：

$$
\widehat y_{t,c}=\widehat y_s+\Delta_c,\qquad
\widehat t_{t,c}=\widehat t_s\exp\Delta_c.
$$

此設計在 h_t=h_s 時倍率恆為1，避免 MLP 學到「同設備也有任意倍率」。對同一 u 的硬體比較具有差分一致性；若換來源 GPU 使 trace 表示 u 改變，就不能保證真實 s→t→s 的預測完全抵消。需要另報 cycle discrepancy，不能過度宣稱。

### 6.3 軟類型組合

$$
\widehat R=\sum_{c\in C}q_B(c)\exp\Delta_c,\quad
\widehat t_t=\widehat t_s\widehat R,\quad
\widehat y_t=\widehat y_s+\operatorname{LSE}_c[\log q_B(c)+\Delta_c].
$$

使用 log-sum-exp 計算，不能把 exp(ΣqΔ) 誤當相同公式。這是條件預測的 soft mixture，不自動等於統計上的真實條件期望；log loss、Huber 與類型機率的校準都會影響它。拒絕片段沒有完整已知類分布，不硬套此公式。

### 6.4 時間訓練損失與 supervision

第一版時間損失不反傳至分段器；先讓 O2 真正學分類／分段，再學時間。否則模型可能為了時間誤差，把兩個算子合成一段而仍自稱辨識成功。

訓練 T 時，q 與候選片段來自 group-OOF 的 MLP-S；GE 的 ŷ_s 來自 group-OOF 的 F。對合法 source/target 配對，令 r*=y_t−y_s，r_hat=LSE_c(log q+Δ_c)：

$$
\mathcal L_V=\sum_w\omega_w\{H_\delta(\widehat y_s+\widehat r-y_t)
+\gamma H_\delta(\widehat r-r^*)\}+\lambda_V\|\psi\|_2^2.
$$

γ初值0.5，candidate {0,0.5,1}，主項直接優化 target prediction，輔項限制倍率不要只補償來源校正錯誤。已知設備六方向每 workload 等權平均，不能讓同一 workload 因產生六個 pair 被算六倍獨立資訊。

**分段不準時的標籤策略：**CTX 時間訓練只用 OOF 預測邊界與 reference event set 完全一致（IoU=1）、但允許類型判錯的片段；source→target 依 logical call 配對。不把錯合併／只含半個算子的預測片段硬貼整算子 target label。記錄這項選擇的 coverage／偏差；若 exact-match 太少就停止擴訓並修 O2，不能改用 oracle 邊界後仍稱 T。IoU≥0.8 仍可作評估配對，時間誤差要保留錯邊界造成的影響。完整 request loss 可在 O4 處理所有片段及未識別比例。

GE 在 isolated scope 上訓練屬已知邊界時間研究，不能獨自證明 T 的無邊界能力。GE 頭直接套上下文僅能列 zero-shot 診斷；CTX 頭重訓後的結果是另一方法，不能混表。

## 7. Wrapper 比較如何使用相同 MLP

| 方法 | 邊界／類型 | 時間模型 |
|---|---|---|
| W-B | 真實邊界，類型由同一 S 的 conditional q 推估 | 相同 F/V 與資料政策 |
| W-BC | 真實邊界、真實 c，q=one-hot(c) | 相同 F/V |
| W-BCS | 真實邊界、c、shape及參數 | F/V 額外加入 shape 表示，其他結構／調參預算相同 |
| T | S 自動邊界及 q | 本文 F/V |
| TM | 自動邊界及 q，加合法 metrics | 同模型家族，增加 metrics/mask |

W-BCS shape schema：GEMM的M/N/K，BMM的B/M/N/K，Conv的B/Cin/Cout/H/W/k/stride/padding/groups，其他類型的元素數或rows/D；欄位不適用附mask。可加算法 FLOPs、輸入／輸出理論bytes與矩陣比例，來源是 wrapper，不稱實測流量。W-BCS 在相同 hidden width 下輸入參數較多，另報參數量；等寬主比較＋等參數量敏感度，不故意削弱 baseline。

CTX的T時間訓練會因OOF exact-match條件少掉部分樣本，需同時做兩種對照：(a)各組用同一有效訓練record子集，隔離額外資訊效果；(b)W-BCS使用同一training配置池中全部合法reference records，代表實際wrapper方法能達到的成績。主要「逼近」主張也要對(b)成立，不能只對被迫丟掉資料的wrapper說接近。列各組有效訓練量與被排除類別。

純 MLP timing 做同表；額外直接回歸 log t_t 的 MLP 作架構消融，與「來源時間×差分倍率」比較。compute/memory/launch 解析式、倍率與線性模型只是基線。

## 8. 訓練／validation／calibration／test 的完整流程

1. 凍結 canonical/equivalence/template group、taxonomy、時間 target、name policy。pilot signature 不進正式 test。
2. 外層維持5 folds：3 folds train、1 validation、1 test；按配置群與模板群隔離所有設備及repeat。
3. validation 群在看成績前再分成 V_select／V_cal（固定hash均分，類型／方向數量另報）。V_select 選 hyperparameters；V_cal 選reject thresholds；temperature首版固定1。小樣本不足則僅報探索，不聲稱完成可靠校準。
4. train 內做3-fold group cross-fitting：每一 inner held-out 群的 S、F、preprocessing 都不能看該群的任何設備資料，產生 OOF 片段、q、ŷ_s。
5. 使用上述 OOF 片段／q／ŷ_s，fit V。若S/F尚在比較超參數，整個OOF流程隨候選重做，不以full-train預測替代。
6. 為避免OOF q與部署q尺度不一致，temperature/reject校準僅用完整pipeline的V_cal；時間V訓練所用q先固定T=1，選非1 temperature後需以train內校準再生成同政策OOF並重fit V，或第一版全固定T=1而只校準拒絕門檻。**首版採後者**，temperature搜尋列後續消融。
7. S/F 在outer train全量重新fit（同超參數與規定epoch），V保留使用OOF inputs訓練的模型；test只推論一次，之後evaluator打分。
8. known-device六方向與leave-one-device-out各自完整重跑。後者的任何held-out GPU workload不可進train/validation/calibration，僅獨立hardware profile合法；不能在缺少真實訓練GPU多樣性時宣稱普遍新設備泛化。

MLP-S最終訓練與OOF模型因樣本量不同，輸出分布仍可能不同；記OOF/validation shift，這是stacking近似的限制，不聲稱完全消除偏差。若validation不穩定，可改固定inner ensemble部署並另立版本，不能test後改。

### 初始可實作超參數

| 項目 | T-v1 初值／選擇規則 |
|---|---|
| S / F / V hidden widths | 64→32 / 32→16 / 64→32；activation ReLU |
| Optimizer | Adam，learning rate=1e-3；gradient norm clip=5 |
| Regularization | 損失中L2：{1e-5,1e-4,1e-3}；optimizer不重複weight decay |
| Batch | S每batch 4個完整sequence，time每batch 32個workload群；不足用全部 |
| Epoch | 正式候選50/100/200，由V_select選整條pipeline；inner-held-out不作early-stop |
| Repeats | 固定seeds {17,42,73}，主報3seed平均與spread，不挑最佳seed |
| Capacity check | 額外32→16及128→64兩種寬度，僅在樣本量／validation支持時啟用 |
| Timing loss | 自然對數Huber δ=0.2；γ∈{0,0.5,1} |
| Candidate L | 由training reference長度決定；validation確認候選涵蓋與成本 |

每個inner training使用候選固定epoch；不能拿inner-held-out真值early-stop後還稱OOF。先採共用epoch候選（50/100/200），由外層V_select選整條pipeline。模型參數數量、獨立配置／模板數與learning curve須回報，重複量測不增加獨立樣本量。

## 9. Compute／Memory／Launch 的可解釋模型放在哪裡？

可另外訓練MLP產生非負代理a_C,a_M,a_L：

$$
\widehat t_t=\frac{a_C(z)}{P_t}+\frac{a_M(z)}{B_t}+a_L(z)L_t,
\quad a_\cdot=\operatorname{softplus}(\mathrm{MLP}(z))_\cdot.
$$

它是physics-inspired消融，不是T-v1主模型；未實測FLOPs/bytes時代理不可當真實工作量。三種GPU僅提供三個硬體點；log硬體差分向量來自三個點時，所有差分的線性span最多2維，所以不能僅靠這些轉移資料自由辨識多個獨立硬體效應。非線性MLP不會憑空補上硬體多樣性。

Compute/Memory/Launch控制實驗有助限制與解釋模型；新架構GPU、不同資源設定可增加證據，但頻率/功率干預會同時改變多種資源，須另立協定。本輪先用三設備做受限預測研究，避免宣稱完成唯一硬體因果分解。

## 10. O4 的 MLP-A：從算子到服務時間

只有在O2/O3成立且完整request taxonomy/unknown政策清楚後才開啟。令TΣ=Σ已預測片段 t_hat_t,i，R=[來源request span/gap、predicted op count、類型比例、unknown比例、target host profile]：

$$
\widehat T_{service,t}=T_\Sigma\exp\{A_\xi(\log(T_\Sigma/\tau),R,h_s,h_t)\}.
$$

初始MLP-A 32→16→1，label為target G0；train輸入全部由OOF O2/O3產生。TΣ=0或首版有拒絕片段時輸出incomplete，不用epsilon虛造全工作時間。若擴充unknown-aware request fallback，另列artifact／coverage。

A能校正host成本與context差異，但必須比較「直接whole-request MLP」和「算子和＋A」，避免只有最後A在吸收全部問題。不能靠優秀G0誤差取代算子辨識成功。CPU/GPU可能重疊，不直接相加API時間；排隊、網路、共置、端到端LLM多token另定義。

## 11. 驗證、停止條件與輸出

- O2：exact boundary F1、matched class Macro-F1、joint typed F1、unknown/challenge、可驗證比例與接受coverage。片段posterior不是時間信賴區間。
- O3：GE/CTX分表、六方向與每類分表；matched條件誤差、全部嘗試與拒絕、每配置MdAPE/P90/MAE；測試target trace僅evaluator讀。
- O4：G0誤差、完整預測coverage、真實短capture總成本，與直接request baseline比較。
- 候選門檻沿用主設計：boundary≥0.90、type Macro-F1≥0.90、joint≥0.85、coverage≥90%；接近W-BCS的MdAPE差≤5pp、P90差≤10pp並報群組bootstrap區間。小樣本區間寬即證據不足。
- R/P不可靠、reference非連續、L候選漏真實片段、OOF exact-match少、三GPU共同profile不足等，均須報阻礙與影響，不能只增加MLP深度。
- 輸出model weights、architecture、feature/schema hash、normalizer、h profile IDs、taxonomy、L、thresholds、split/group hashes、seeds、target_kind及全部失敗狀態。

時間區間第一版先報3seed變動與group-bootstrap評估區間；它們不等於單次prediction interval。若需單工作coverage保證，須另留足夠group calibration資料做區間校準，不能直接把softmax或seed spread當95%置信區間。

## 12. 目前需要的資料與實作順序

**現在不需使用者先提供AI運算公式，也不需在此電腦安裝GPU環境。** 本規格可先完成；以下資料決定最終特徵、L與可訓練性：

| 需要 | 最小內容 | 缺少的影響 |
|---|---|---|
| 一個可靠R/P配對樣本 | 同manifest的有標記／無標記trace、SQLite、log、run metadata | 不能確認事件schema、reference映射與真正無標記標籤 |
| 三設備E2結果 | 原始cell key、median/spread、單位、protocol及工具版本 | 不能凍結h向量與可比硬體錨點 |
| O1 pilot | 每類3配置，各設備G0/GE/P/R與input checksum | 不能確認計時噪聲、trace→GE可校正性 |
| O2小型mixed pilot | 同類連續、多kernel、不同shape，完整reference | 不能選L、驗證候選recall、確認MLP-S可學性 |
| Metrics inventory | supported/permission/sample-rate/definition | TM先不啟用，T可繼續 |

節點順序：① exporter/label gate → ② isolated MLP-S分類與GE F/V sanity → ③ mixed MLP-S＋DP → ④ group-OOF CTX V與wrapper對照 → ⑤ 擴矩陣／O4。九配置只驗工具與數值，不足以支持MLP泛化，不能要求pilot就達正式精度。

文件中的公式／張量維度可用手算或小型合成資料檢查數值一致性；這只驗規格，不是模型實作或GPU成績。本次未新增可執行O程式。

本次規格已用4事件、最大長度3、2類型的52種完整路徑枚舉，核對forward partition、max-path、backward片段posterior及事件coverage；時間soft-mixture亦核對算術與log-sum-exp一致。這些僅是公式數值稽核，沒有訓練MLP，沒有實測GPU或驗證辨識精度。

## 13. 參考依據與本研究新增部分

- [Habitat 原論文](https://www.usenix.org/system/files/atc21-yu.pdf)：提供operation層級跨設備預測與MLP的參考；本文從trace恢復邊界／類型是額外問題，不把wrapper metadata假裝已可直接取得。
- [NeuSight 原論文](https://arxiv.org/html/2407.13853v2)：參考硬體條件化與分層建模；本文未複製其tile利用率模型，也不宣稱復現其結果。
- [Sarawagi & Cohen, Semi-Markov CRF](https://www.cs.cmu.edu/~wcohen/postscript/semiCRF.pdf)：參考變長片段的條件模型與DP；本文MLP勢函數、GPU特徵、拒絕及跨設備時間頭是提出的研究設計。
- [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)：核對trace、annotation及metrics的工具觀測範圍；具體欄位與權限仍以三節點版本為準。
