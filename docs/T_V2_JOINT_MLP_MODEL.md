# T-v2：共享 MLP 的算子辨識與跨設備時間預測

更新：2026-10-28。**研究模型設計，尚未實作或訓練。** 本文件整理使用者對 [T-v1 分段模型與 F/V 分解](T_V1_MLP_MODEL_SPEC.md) 的修正：F 的輸出沒有作為 V 的輸入，額外的來源時間模型並非主要研究目標所必需。T-v2 將「來源 trace／metrics → 目標設備算子時間」設為直接時間預測，並讓算子分段、類型與時間任務共享表示、共同最佳化。T-v1 保留為可比較的分解式基線。

本文件只修訂**模型**，不修改既有 [O1 pilot 程式](../o1_operator/README.md)、凍結量測協定或原始資料。主線目前有 O1 九配置 pilot 與 [GB10 部分結果](O1_GB10_PILOT_STATUS_20261021.md)；O2/O3 在主線仍未實作。三設備為 RTX 3090、RTX 4090、GB10。資料邊界、GE／CTX／G0 時間目標、品質規則仍依 [O 系列設計](NEXT_EXPERIMENTS.md) 與 [資料契約](OPERATOR_DATA_CONTRACT.md)。

## 1. 修正原因：原 F/V 的數學關係

原設計的前向計算為：

$$
\widehat t_s=D_B\exp F(z_B,h_s),\qquad
\widehat t_t=\widehat t_s\exp V(z_B,h_s,h_t).
$$

其中 D_B 是來源候選片段的 GPU span，z_B 是來源 trace 特徵，h_s/h_t 是兩端獨立硬體校正。**V 不接收 F 的輸出**；F 只改變最後用來乘倍率的來源時間基準。若 V 的訓練損失使用 \(\widehat t_s\)，F 的預測誤差會間接影響 V 的參數，但不構成 V 前向推論的必要輸入。

若兩個預測的相對誤差為 ε_F、ε_V，最後的相對誤差為：

$$
\epsilon_t=(1+\epsilon_F)(1+\epsilon_V)-1
=\epsilon_F+\epsilon_V+\epsilon_F\epsilon_V.
$$

這不等於「訓練三個模型，錯誤率就乘三」。誤差可能同向累積，也可能抵消。分段錯誤則會改變算子單位，不能當另一個獨立百分比誤差直接相加。原分開訓練的流程沒有直接最小化完整管線的主要時間誤差，且 CTX 目標原本已令 F=0，因此 F 不應是主方法的必選模型。

## 2. 任務範圍與觀測條件

來源 request 的事件序列 \(X_s=(e_1,\ldots,e_n)\) 來自真正**無逐算子標記**的 P trace；首版允許已知整個 request 的 capture 範圍。Reference wrapper 在訓練與 evaluator 端提供真實分段／類型 \(Y^*=\{(a_i,b_i,c_i)\}\)，不進 test predictor。目標設備測試工作的時間或 trace 只供評分。獨立事先量測的硬體資料可進 predictor。

首版單 stream、eager、固定 precision/layout，並要求每個 reference op 的 GPU 工作在事件序列中連續。融合、多 stream、無 GPU 工作的 view、無法驗證 R→P 標籤者，需另外定義或列出不適用／待審，不能強配類型。已知類別先從 GEMM、BMM、Conv2D 開始，Add／LayerNorm 依資料擴充。辨識不出來的片段保留 `unknown/abstain`，不冒充完整預測。

三種 label 不混用：

| 任務 | 目標 | T-v2 的時間輸出 |
|---|---|---|
| O1-GE | 獨立算子的 CUDA event elapsed | 用於獨立算子時間研究及可能的預訓練 |
| O3-CTX | 混合序列或 DNN 內可驗證算子的目標 P trace span | **第一個正式 joint 模型的逐算子時間目標**，是 profiler 下 proxy |
| O4-G0 | 未開 profiler 的完整 request host 完成時間 | 需另外的組合／服務時間頭，不能把 CTX 逐算子和當 G0 |

O1-GE 與 O3-CTX 可共用特徵抽取設計，但須使用不同時間頭或明示的 target-kind 條件及缺值 mask；同一列訓練資料只對有實測標籤的任務計 loss。不能把不同 target 混成一個 `latency`。首個正式 joint 驗證先以 O3-CTX 為主，O4-G0 待算子涵蓋與可靠標籤通過後再擴充。

## 3. 主模型：一個共享編碼器、兩個輸出任務

對所有連續候選片段 \(B=[a,b]\)，從來源事件抽取 \(z_B=\phi(X_s,B)\)，含 duration、kernel 數、gap、launch geometry、可用的名稱 token；metrics 只有在 O0 確認時間解析度與三設備可比性後加入。特徵不可含真實 op type／shape／邊界、NVTX 名稱、config ID 或目標測試結果。硬體向量 \(h_d\) 保存 E2 cell 的尺寸、單位、品質與來源；缺值不補零。

共享 MLP 編碼器：

$$
r_B=E_\theta(z_B,h_s).
$$

首版候選架構為兩層 ReLU、寬度 64→32；這是可實作初值，不是已驗證最優值。GPU 名稱與 pair ID 不輸入。來源設備資料可協助解釋來源 trace；目標設備 \(h_t\) **只進時間頭**，不改變同一來源工作應有的算子邊界／類型。

### 3.1 算子分段及類型頭

$$
A_\psi(B,c)=C_\psi(r_B)_c,\qquad c\in\mathcal C.
$$

對所有完整、互不重疊、沒有漏事件的分段 \(Y\)，定義：

$$
p_{\theta,\psi}(Y\mid X_s,h_s)=
\frac{\exp\left(\sum_{(B,c)\in Y}A_\psi(B,c)\right)}
{\sum_{Y'\in\mathcal Y(X_s)}\exp\left(\sum_{(B,c)\in Y'}A_\psi(B,c)\right)}.
$$

用 forward/backward 動態規劃求 partition、片段邊界 posterior；以 max/backpointer 推論最佳完整分段。固定片段 B 時的已知類型條件機率為 \(q_B(c)=\operatorname{softmax}_c A_\psi(B,c)\)。**類型機率與邊界可信度分開記錄。** 未知類首版用拒絕門檻，並非偽造一個已訓練好的 unknown 機率。最長候選片段與訓練細節可沿用 T-v1 §3～4，但此處的編碼器與時間頭共同訓練。

### 3.2 直接目標時間頭

時間 MLP 接收共享片段表示、目標設備向量及類型條件，對每個已知類輸出 log 時間：

$$
m_\omega(B,c,t)=\log(D_B/\tau)+G_\omega(r_B,h_t)_c,
\qquad \widehat t_{B,c,t}=\tau\exp m_\omega(B,c,t),
$$

其中 \(D_B\) 是來源片段 span，\(\tau=1\,\mu s\)。\(\log D_B\) 是已觀測的時間尺度，G 學會來源到目標的**整體修正**，可同時反映 profiling 狀態、算子特性與設備差異；不另設 F。若數值特徵已包含 \(D_B\)，實作需指定此 residual 加法，只加一次，並固定所有單位。首版 G 可用 64→32 的共享層加 \(|\mathcal C|\) 個輸出；小資料時需和較小模型比較。

對給定片段 B，保留類型不確定性的時間：

$$
\widehat t_{B,t}=\sum_{c\in\mathcal C}q_B(c)\widehat t_{B,c,t}.
$$

此式預測「假設 B 是一個算子」時的條件時間。若 B 是兩個算子的錯誤合併，不能把它的輸出冒充正確逐算子時間；辨識 coverage 和時間誤差必須一起報。MLP 也不會因為 joint training 自動恢復 trace 中根本不存在的語意資訊。

## 4. Joint 的兩個層次：先可實作，再驗證端到端

### 4.1 第一版：共享表示、共同訓練

有可靠 reference \(Y^*\) 和目標設備訓練時間 \(t^*_{i,t}\) 時：

$$
\mathcal L_{struct}=-\log p_{\theta,\psi}(Y^*\mid X_s,h_s),
$$

$$
\mathcal L_{op}=\frac1M\sum_{i=1}^{M}
H_\delta\!\left(
\log\frac{\widehat t_{B_i^*,t}}{t^*_{i,t}}
\right),
$$

$$
\mathcal L_{v2}=\lambda_{struct}\mathcal L_{struct}
+\lambda_{op}\mathcal L_{op}
+\lambda_{reg}\|\theta,\psi,\omega\|_2^2.
$$

\(H_\delta\) 是 Huber loss。時間 loss 在真實片段上使用**模型算出的 q**，因此可以把時間梯度傳給類型頭與共享編碼器；結構 loss 同時約束邊界與類型。這是明確的 joint training，但逐算子時間 loss 只在 \(B_i^*\) 計算，**尚未讓時間 loss 直接比較所有候選切法**。不能把它稱為完全端到端的分段最佳化。

各 request／workload 配置等權，repeats 先聚合；三設備及六方向的同一 logical 配置仍是一個相依群。\(\lambda\) 由 train/validation 決定，不以 test 調整。候選可先將各 loss 除以 train 首個 epoch 的固定尺度，再在 validation 選 \(\lambda_{struct}:\lambda_{op}\in\{1:0.25,1:1,1:4\}\)；實際值需由 pilot 驗證。分類／回歸 loss 的權重會影響多任務結果，不能假設同權一定最佳。[多任務損失權重研究](https://arxiv.org/abs/1705.07115)

### 4.2 延伸：讓完整時間誤差影響分段

若有與輸入序列對應且定義一致的**完整 request 時間** \(T_t^*\)，可加入可微分的序列期望。當預測量可寫為各片段時間的加總時：

$$
\widehat T_{sum,t}
=\sum_{B,c}p_{\theta,\psi}(B,c\mid X_s,h_s)\,\widehat t_{B,c,t}.
$$

片段 posterior 由分段模型 forward/backward 求得，因此 \(\mathcal L_{sum}=H_\delta(\log(\widehat T_{sum,t}/T_{sum,t}^*))\) 可以對片段分數傳梯度。**這裡的 \(T_{sum,t}^*\) 必須是同一時間目標下可驗證的片段時間總和**；不能把 profiler 下的 CTX 片段和直接拿去對無 profiler 的 G0 服務時間。即使總和正確，仍可能切錯，所以 \(\mathcal L_{struct}\) 和辨識指標不可移除。

O4 若使用 G0，需另定 host／context 組合頭 \(A_\xi\) 與對應 G0 loss；CPU/GPU 重疊、片段間 gap、未知類、拒絕與多 stream 會使簡單總和失準。若 \(A_\xi\) 非線性，\(E_Y[A_\xi(T_Y)]\) 一般**不等於** \(A_\xi(E_Y[T_Y])\)，不能把兩者互換。可先使用有限候選分段的加權估計並核對近似誤差，或把完整可微分組合另列研究版本。O4 尚未實作。

## 5. 資料怎麼沿用，哪些仍需收集

| 資料 | 修正版的用途 | 現況／缺口 |
|---|---|---|
| O1 獨立 GEMM/BMM/Conv | 檢查時間頭、預訓練共享表示、已知邊界時間基線 | 主線已有九配置 pilot；GB10 G0/GE 部分完成，數量不足以證明 MLP 泛化 |
| O2 混合序列 P/R | 訓練可靠的分段／類型，提供 O3-CTX 上下文工作 | 需要三節點同 manifest；主線尚無完整 O2 訓練資料 |
| E2 硬體校正 | 建立 \(h_s,h_t\)，可先選品質合格的共同 cell | 必須檢查原始 cell、時間範圍、單位與軟體差異 |
| O4 完整 request G0 | 將算子預測轉成 host 服務時間 | 依固定模型／輸入及標籤品質另收或核對既有資料 |
| PM metrics | 額外特徵與增益消融 | 僅在三節點實際支援及採樣覆蓋足夠後使用 |

改變模型不要求重跑已通過的 O1 G0/GE；O2 所需的混合序列仍需收集。GB10 O1 P/R 目前有事件完整性警告與 `manual_review_required_event_loss`，**不能直接拿它們當正式無標記算子邊界訓練標籤**。先做品質 gate，再決定是否可用。

## 6. 訓練、比較與防止「時間準、算子錯」

1. 凍結 taxonomy、特徵 schema、GE/CTX/G0 label、三節點共用 manifest。依 operation signature、等價群、混合序列模板共同切分所有設備與 repeats；同配置不跨 train/test。
2. 先檢查算子候選 recall、R→P 唯一映射及 trace 完整性。沒有可靠 P reference 時，joint model 只能作受限研究，不能報正式 O2 能力。
3. 可用 O1 作時間頭／編碼器 warm start；正式 mixed 訓練以 \(\mathcal L_{v2}\) 同時更新共享編碼器、分段頭與時間頭。warm start 是初始化，不是推論時額外一個 F 模型。
4. Validation 選網路容量、loss 權重與拒絕門檻；split、normalization、特徵選擇及 calibration 不看 test 或測試目標設備工作。留一設備測試只允許事先獨立的硬體 profile。
5. 主比較至少包含：同容量的分開訓練 S→直接時間頭、共享表示但阻斷時間梯度至分段頭、T-v1 F/V 分解、T-v2 joint，以及 W-B/W-BC/W-BCS wrapper 參考。所有方法使用相容 label 與測試群；W-BCS 應能使用其合法的完整訓練標籤，不故意弱化。
6. 分別報 exact boundary F1、matched type Macro-F1、joint typed F1、unknown/abstain、coverage；六方向與逐算子 MdAPE/P90/MAE；O4 另報 G0 服務時間。時間誤差低而算子錯誤的模型不能宣稱達成 operation-aware 目標。

一個值得預先指定的反例：真實序列是 30 µs Conv＋20 µs GEMM；模型錯合成 50 µs 單片段。總時間可很準，結構仍錯。\(\mathcal L_{struct}\) 和 joint typed F1 必須揭露這種情況。

## 7. 目前能承諾與不能承諾的事

**能成立的是方法設計與可驗證路徑：**原 F 沒有進 V 前向；移除獨立 F 可簡化主方法；共享編碼器與兩頭可用同一目標設備時間資料聯合訓練；O1 資料與已取得硬體校正有可重用價值。

**尚待實驗的是性能：**joint training 是否改善時間、分段或兩者；O2 的 P 標籤能否可靠取得；三設備硬體向量是否足以支撐未見設備推論；短算子的 metrics 是否有足夠採樣。MLP 不能解決語意不可識別、事件遺失或資料量不足。GB10 與 RTX 的 host／框架差異也需在服務時間結果標示。

本輪的交付是文件與研究設計。下一個節點實作門檻是完成可信的 O2 mixed-sequence P/R 對應及三設備配置配對；之後才可驗證 T-v2 joint 訓練的實際收益。
