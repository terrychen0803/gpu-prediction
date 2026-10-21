# O 系列研究報告範本

這是範本，沒有實測結果。複製到新的分析目錄後填寫；未知／未測／待審不要改成通過。方法依 [設計](NEXT_EXPERIMENTS.md) 與 [資料契約](OPERATOR_DATA_CONTRACT.md)。

## 1. 本次問題與證據

- 階段：O0/O1/O2/O3/O4/O5；要回答的研究問題：待填。
- 模式：isolated／mixed sequence／DNN；P／sanitized-R／PM 必須區分。
- 時間目標：isolated GE／contextual trace proxy／request G0，不混表。
- 程式 commit、dirty patch hash、protocol/taxonomy、collector／分析版本：待填。
- 原始資料 inventory/checksum、input/weights、split hash、hardware profile：待填。
- 實際 GPU／CPU、軟體堆疊、Nsight backend、功率／時脈狀態：待填。
- 這份結果是 GPU 實測／合成 fixture／僅規劃：待填。

## 2. O0 資料與可觀測性

| 設備 | 預期單元 | 找到 | 正確性通過 | 穩定 | trace 可驗證 | 失敗／待審 | 證據 |
|---|---:|---:|---:|---:|---:|---|---|
| RTX 3090 | — | — | — | — | — | 待填 | 待填 |
| RTX 4090 | — | — | — | — | — | 待填 | 待填 |
| GB10 | — | — | — | — | — | 待填 | 待填 |

| metric 精確名稱／定義／單位 | 3090 狀態 | 4090 狀態 | GB10 狀態 | 採樣頻率／scope | 是否可比 | 證據 |
|---|---|---|---|---|---|---|
| 待填 | unverified | unverified | unverified | 待填 | 未確認 | 待填 |

列全部排除／缺失與規則。列 Nsight diagnostics、runtime/driver 去重、PID/context 關聯、capture 截斷、未知 schema、事件數檢查及人工稽核。parser passed 不足以標 trace 完整。

## 3. R/P 標籤可信度與干擾

| workload 類別 | R 直接標籤 | P 唯一對應 | 多解／路徑差異 | 不可驗證比例 | R/P 時間與 gap 差異 |
|---|---:|---:|---:|---:|---|
| 待填 | — | — | — | — | 待填 |

說明 mapping algorithm、signature、時間軸、標籤 provenance。是否只刪除 R 的 NVTX？是否真的測 P？無法對應的資料如何保留在 coverage 分母？reference 與 predictor 是否獨立？

## 4. 切分與模型輸入

列 train/validation/test 的配置數、模板數、各類數、所有設備方向、OOF 層次、未知類集合。說明每組允許的 feature allowlist、kernel 名稱政策與來源時間條件。證明 config/equivalence/template/repeats/日期/crops 不洩漏。已知設備與 leave-one-device-out 分開報。

若用三輪設備留出，逐輪列訓練 GPU、留出 GPU、可用硬體校正及完全未使用的 workload 資料；不能用留出設備 target latency 調參。

## 5. 算子辨識

| 方法／設備／資料條件 | 嘗試 operations | 可驗證比例 | Boundary F1 exact | Matched type Macro-F1 | Joint typed F1 | unknown 檢出 | 預測 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 待填 | — | — | — | — | — | — | — |

附每類 confusion matrix、多 kernel、連續同類、BMM B=1/GEMM 等挑戰。精確語意與粗計算家族分開；附未匹配片段、錯切／漏切、拒絕案例。不能只報 matched 分類成績。

## 6. 時間模型及 wrapper 差距

每個 target_kind 各填一張表；六方向逐一，不只 pooled 平均。

| 方向／類別 | 方法 | 配置數／有效數 | MdAPE | P90 APE | MAE µs | 相對 W-BCS 的差（百分點） | 配對差 95% 上界 | coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 待填 | B0/W-B/W-BC/W-BCS/T/TM | — | — | — | — | — | — | — |

matched-conditional 誤差與完整 request 誤差分開。列 source 校正、分段、分類與時間回歸的消融。說明 bootstrap 單位；未達標或證據不足如實寫。oracle-time、R-only、kernel-name-free、metrics 子集不要混為一個主成績。

## 7. 上下文／完整 inference

列 taxonomy 覆蓋的 GPU 時間比例、event 比例與 operation 比例；未涵蓋類型及 unknown residual。區分 seen-shape/unseen-context 與 unseen-shape。isolated GE、context proxy 與 G0 三種標籤分別解釋，不能宣稱得到無干擾逐算子真值。

比較串行相加與經訓練的 request 組合模型，說明 CPU/GPU 重疊、cache／allocator、host 架構、fusion、多 stream 等限制。Qwen 若納入，列 prefill/decode 與 KV cache 狀態。

## 8. 成本與可用性

| 收集條件 | capture | export | parse | 辨識 | predict | 初始化 | 空間 | 時間誤差／coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 待填 | — | — | — | — | — | — | — | 待填 |

真實短 capture 與離線裁切分開。T/TM 使用共同樣本比較並另列 PM 干擾／metrics 不可用比例。未量測排隊、網路、共置干擾時，不延伸成 SLA 保證。

## 9. 結論與下一個 gate

1. 已有直接證據的能力：待填。
2. 未達成／不可辨識／仍不確定的部分：待填。
3. 哪些結果能支持使用者「從 trace/metrics 建立 operation-aware 時間模型」：待填。
4. O0～O5 各 gate：通過／受限／未完成及證據。
5. 下一個最小必要工作、三節點任務與預期產物。

可公開 aggregate 與研究報告可回存 Git；原始 trace、權重、私人資訊留節點。所有數字連結可追溯資料，不把規劃門檻寫成成果。
