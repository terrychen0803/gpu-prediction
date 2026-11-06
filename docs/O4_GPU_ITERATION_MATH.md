# O4 數學設計：operation → GPU iteration

更新：2026-11-06。這是 [O4 主計畫](O4_GPU_ITERATION_PLAN_ZH_TW.md) 的待實作規格。假設單 stream、已知 iteration 窗口、暖狀態、同一工作設定。**主方向 s=RTX4090，t∈{GB10, RTX3090}**；其餘方向為選配。公式中的真值只出現在訓練或 evaluator；部署不得讀 test target trace／GI。

## 1. 符號與可觀測資訊

| 符號 | 意義 |
|---|---|
| s、t | 來源／目標設備，含被固定的執行堆疊 |
| X_s=(e_1,…,e_N) | P 中 iteration 內來源 GPU 事件，含明示處理的 kernel/copy/memset |
| B=(a,b) | 事件索引 [a,b) 的候選片段 |
| z_B、D_B | 來源數值特徵、首 GPU start 到末 end 的 span（µs） |
| h_s、h_t | 獨立硬體校正，不是 test workload 的 target 值 |
| S*、c*、d* | reference 分段、類型及 target P 中 contextual span |
| Y_t^GI | target 無 profiler 的一次 iteration event 時間聚合標籤 |
| θ、φ、ξ | 辨識器、片段時間 MLP、完整迭代校正 MLP 參數 |

GT op count、shape、模型名及真實類型均不進主 predictor。q 是從合法來源特徵計算的摘要，但新主候選 N0 的時間頭不讀 q。事件 ID 只做關聯，不作 embedding。

## 2. 辨識器：保留目前 semi-Markov 架構

對候選片段與已知類 c 計分：

$$
a_\theta(B,c)=W_c\operatorname{MLP}_\theta([\widetilde z_B,\widetilde h_s])+b_c,
\qquad q_B(c)=\operatorname{softmax}_c a_\theta(B,c).
$$

在不重疊且覆蓋序列的分段集合中選最高分：

$$
\widehat{\mathcal S}=\arg\max_{\mathcal S}\sum_{(B,c)\in\mathcal S}a_\theta(B,c).
$$

訓練使用 partition 與 reference score：

$$
L_{seg}=\frac1J\sum_j\frac{\log Z_\theta(X_j)-\sum_{(B,c)\in\mathcal S_j^*}a_\theta(B,c)}{N_j}.
$$

這只是已知類、可分段序列的核心模型。最大候選長度 L、特殊 copy/fused 類別與拒絕策略由 development 凍結；L=4 是舊資料設定，不自動適用所有 DNN。強制覆蓋所有事件不等於辨識正確，需另作 unknown／超界挑戰。未訓練未知類時，不把低 max(q) 當已校準的未知機率。

純 GPU trace 無法看到的零工作 view、CPU-only metadata 操作，由 reference 另記；不要求辨識器為其虛構 GPU span。多 stream 的非連續事件集合不屬此簡單連續分段主模型。

## 3. N0 片段時間模型

$$
\widehat d_{B,t}^{P}
=D_{B,s}^{P}\exp\left(g_\phi(\widetilde z_B,\widetilde h_s,\widetilde h_t)\right).
$$

第一個可比較版本保留現有 43→64→32→1，ReLU；27 維 z＋兩端各 2 維硬體＋全零 3 維 q 欄＋全零 9 維 shape 欄。若直接縮為 31 維，屬新容量對照，不能與舊 N0 當完全相同模型。擴 taxonomy 後即使維度不變，也需擴資料重訓，不假設舊權重支援新類。

定義 log 殘差 e=log(預測/真值)，Huber：

$$
H_\delta(e)=\begin{cases}\frac12e^2,&|e|\le\delta,\\
\delta(|e|-\frac12\delta),&|e|>\delta.\end{cases}
$$

$$
L_{op}=\frac1J\sum_j\frac1{|\mathcal S_j^*|}
\sum_{B\in\mathcal S_j^*}H_{0.2}\left(\log\frac{\widehat d_{B,t}^{P}}{d_{B,t}^{P,*}}\right)
+10^{-4}\|\phi\|_2^2.
$$

先在正確 reference 片段訓練時間頭，再在盲分段上評完整結果；未知邊界錯切片段不得硬貼整個真算子的時間標籤。若未來以 OOF 預測片段校正，只用可明確對應的片段並報選擇 coverage／偏差。

舊超參數可作第一個重現基準：Adam 0.001、clip norm 5、120 epochs（舊資料為 720 updates）、三 seed 17/42/73。新資料量會改變 updates，不能同時假設固定 epochs 與固定 updates；先固定共同更新預算及 batch 規則，再以 validation 有限調整。

## 4. 為何算子加總不等於完整 iteration

### 4.1 單 stream 下的觀測恆等式

在同一 P trace 中，假設每個 GPU 工作恰好分配一次，片段時間區間不重疊，且已處理所有 copy/memset：

$$
U_t^P=\sum_i d_{i,t}^P+G_t^P,
\qquad G_t^P=\sum_{i=1}^{m-1}(start_{i+1,t}-end_{i,t}).
$$

U 是第一個實際 GPU 工作到最後一個完成的 span。d 已包含片段內 gap，只補片段間 gap，不能再加 kernel gap 全總和。驗證閉合差應只在 timestamp 精度範圍內；若差大，先查重複計數、漏工作、重疊、scope，不能用回歸把解析錯誤掩蓋。

這個恆等式不表示 GI=U：GI 邊界是 CUDA events，且不開 profiler；P 有 profiler 擾動及可能不同的首尾空檔。因此以下是需要驗證的統計映射：

$$
Y_t^{GI}=f\left(\{d_{i,t}^{P}\},G_t^P,\text{timing policy, platform state}\right)+\epsilon.
$$

不假定 f 為常數加法，也不要求校正一定為正。Profiler 影響／overlap／漏工作等因素必須分開診斷。

### 4.2 三個組合層次

**SUM：不另訓練的算子基準。**

$$
\widehat S_t=\sum_{B\in\widehat{\mathcal S}}\widehat d_{B,t}^{P}.
$$

把 S 與 GI 比較時名稱標為「CTX sum → GI baseline」，不可把 target_kind 改成已驗證的 GI 預測而不揭露差異。

**SPAN：可選的 profiler span 診斷。** 若要研究 gap 的貢獻，可只用 training P 標籤訓練一個總 gap 頭：

$$
\widehat G_t^P=\tau\operatorname{softplus}(b_\eta(r_s,h_s,h_t)),\qquad
\widehat U_t^P=\widehat S_t+\widehat G_t^P,\quad\tau=1\,\mu s.
$$

此頭不要求逐 gap 跨設備一一對應。對 0 gap 使用絕對／log1p loss，不以任意 epsilon 算 APE。SPAN 只驗證 P span，不代表無 profiler GI。這是可選消融，不要求為完成主線增加另一個必經模型。

**ITER：主要 GI 校正候選。**

$$
\widehat Y_t^{GI}=\widehat S_t\exp\left[A_\xi(v_{s\to t})\right].
$$

其中 v 只包含：log(S/τ)、預測片段數、預測時間分布統計、來源 iteration span／inter-segment gap／短 kernel 比例、合法來源 launch 統計、兩端硬體校正；新增 target launch/host profile 必須是獨立 microbenchmark。初版不使用真實類型、shape、模型 ID，也不使用 test target gap 或 P span。

建議 A 為小型 `d_iter→32→16→1` MLP，與 operation 時間頭分開。完整事件覆蓋未確認、S≤0、缺片段或 artifact 不相容時回傳 incomplete；不能以 epsilon 補成可預測。若通用 unknown residual 能給時間但語意不明，另列 fallback 成績。

$$
L_{iter}=\frac1K\sum_{k=1}^{K}\frac1{|\mathcal D_k|}
\sum_{(s,t)\in\mathcal D_k}H_{0.2}\left(\log\frac{\widehat Y_{k,s\to t}^{GI}}{Y_{k,t}^{GI}}\right)
+10^{-4}\|\xi\|_2^2.
$$

k 是配置／獨立群的明示權重單位，D_k 為有效設備方向。固定配置內 repeats 先聚合，不能讓擁有較多 repeats 的配置支配 loss。若各群包含不同數量配置，還須明定先群平均或先配置平均，所有比較一致。

主實驗 D_k 預定包含4090→GB10與4090→3090，兩方向等權；缺某目標時記缺失，不能不揭露地重配權重。主時間模型先共享 φ/ξ，使用 h_t 區分目標；每目標獨立頭僅作另列對照。來源 h_s 固定並不代表能驗證任意來源設備的泛化，兩個 target 硬體點也不足以證明通用硬體尺度律。

ITER 第一版先凍結 θ、φ，只訓練 ξ。候選 Adam 0.001、clip5、120 epochs、每5 epochs validation；依 GI time20 最大、P90 APE 最小、MdAPE 最小、較早 epoch 選模。這是新候選規則，正式收集前凍結；wrapper／direct 等方法用相同候選數與規則。相同表面容量不保證完全相同有效容量，報參數量與各階段成本。

訓練資料增多後若研究 end-to-end loss，可另立版本；不能讓 GI loss 補償錯誤語意後宣稱 operation-aware 全部成功。q 消融暫不支持重新增加必經 F 或複雜 q 路徑。

## 5. 多階段訓練與 OOF

### 5.1 資料流程

1. Outer train/validation/test 按研究問題建立分組，所有方向／repeats 同組。
2. Outer train 內按配置群產生 inner splits。每個 inner 教師 θ、φ 的訓練與 normalizer 均不得讀 inner held 的標籤。
3. 用每個教師對 inner held 的來源 trace 產生分段、預測時間、S 與 v；保存全部失敗。這些 OOF v 與合法 held 的 GI 標籤供 A 訓練。
4. 用完整 outer train 重訓 θ、φ，對 outer validation/test 產生 v。A 的 checkpoint 只在 validation 選；test 不參與 source view、L、threshold、錨點或模型選擇。
5. 只有一個固定 O3 教師且其訓練與所有 O4 校正配置完全不交集時，可直接用它產生 O4 校正特徵；需保存 signature 隔離證據，不能只因名稱叫「pretrained」就免除洩漏檢查。

現有五群資料每 outer train 只有三群，inner 教師會較弱。若新 taxonomy 在 inner train 缺類，OOF 不成立；改用獨立 upstream train 與 downstream calibration 集，或擴資料。不能拆 repetition 解決。候選只對完整有效教師輸出訓練 A 時，報明選擇偏差，測試仍保留全部失敗分母。

### 5.2 零調整與重新校正的差別

- Zero-shot context：固定已訓練 O3 模型，未見過新 DNN context label，測 CTX 轉移。若沒有 GI 校正資料，SUM 只作基準。
- Context-adapted：使用明確 training-context CTX 調整 operation 模型；與前者分表。
- GI-calibrated：A 使用 training GI；不能宣稱從來未用 target 設備工作負載資料。
- Leave-device-out：整個 target 設備的 workload CTX/GI 不進任一 fit，只允許事前硬體校正；是 O5 的額外難題。

## 6. 必要比較組

| 比較組 | operation 資訊 | 完整時間模型 | 解讀 |
|---|---|---|---|
| RATIO | 整次來源 P span＋train-only 倍率／硬體比例 | 簡單 GI 映射 | 所有人使用同合法來源觀測 |
| DIRECT | 整次來源 trace 數值統計 | 直接 MLP→GI | 檢查拆解是否有價值；不提供語意輸出 |
| T-SUM | 盲分段、類型，N0 時間 | 直接加總 | operation 建模的未校正能力 |
| T-ITER | 與 T-SUM 相同 | 小型 A→GI | 主候選 |
| W-B | reference 邊界、相同 P 特徵 | 對應 N0＋相同容量 A | 邊界資訊對照 |
| W-BC | 再提供真實類型 | 對應類型時間頭＋A | 已知類型對照 |
| W-BCS | 再提供 shape | 對應 shape 時間頭＋A | wrapper 資訊對照，不等於 Habitat 原法重現 |

各組的 A 以該組 OOF 特徵各自訓練，不能只用 T 的校正權重套 W 後稱公平最優比較。診斷若固定同一 A 必須另標「固定校正權重消融」。訓練範圍、label、來源 P、切分、checkpoint 搜尋預算相同；wrapper 額外標籤是該組宣告的資訊條件。

DIRECT 可使用和 T 相同硬體欄位與完整 trace 統計，但不偷加模型名／batch／shape；其參數量與訓練成本要報，先用小型容量對照，勿刻意弱化。若 DIRECT 相當或更好，仍可報 T 的可解釋性與運算輸出，但不能宣稱拆解提高了時間精度。

## 7. 誤差傳遞與解釋

在完全相同、正確分段的可加總時間目標下：

$$
\frac{\widehat S-S}{S}=\sum_iw_i\epsilon_i,
\qquad w_i=\frac{d_i}{\sum_jd_j},\quad
\epsilon_i=\frac{\widehat d_i-d_i}{d_i}.
$$

所以不是三個模型就三倍誤差，也不是把各算子的 MdAPE 相加。長算子的系統性低估可能主導整次誤差；正負抵消可讓總時間準，但不保證語意正確。錯切、漏工作與重複覆蓋破壞這個簡單式子，必須另報。

推薦診斷順序：先查 reference 與閉合 → oracle boundary 的 CTX 時間 → 盲分段的 coverage → SUM/P span 差 → GI 校正差。不要在前面資料出錯時只擴大 A。

## 8. LLM 與多 stream 的延伸

LLM 主標籤分為 `prefill_forward` 與 `decode_forward`。前者固定 prompt L，後者固定 forward 前 KV 長度 L_cache、token input、cache 實作／容量／position／attention backend。每次獨立 decode 前在計時外重建／還原相同 cache 狀態並完成準備工作；不把持續變長的生成序列當重複同配置。

若未來預測生成 G 個 token，prefill 的 logits 產生第一個 token，則可用：

$$
\widehat T_{gen}=\widehat T_{prefill}(L)
+\sum_{j=1}^{G-1}\widehat T_{decode}(L+j-1)
+\widehat T_{sampling/loop}.
$$

只補尚未包含的 sampling/loop；這是未來生成目標，不等於本輪單次 GI。輸出長度未知時另研究長度／停止規則，不靠單步時間解決。參考 [官方 cache 文件](https://huggingface.co/docs/transformers/kv_cache)。

多 stream 後續可用已知依賴／資源順序 E 與 release r_i 建立：

$$
F_i=\widehat d_i+\max\left(r_i,\max_{j\to i}F_j\right),\qquad
\widehat T=\max_iF_i-t_0.
$$

但 source trace 的觀測順序不一定提供完整資料依賴，target 的 kernel 分解、重疊與資源競爭也可能改變；不能只畫 DNN DAG 就宣稱排程模型正確。第一版發現多 stream 時應報 unsupported 或另協定，不能將其偷偷套入單 stream 總和。
