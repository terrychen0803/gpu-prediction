# RTX 3090／RTX 4090／GB10 節點 Codex 交接

更新：2026-10-21。倉庫：[gpu-prediction](https://github.com/terrychen0803/gpu-prediction)。本次審查的原始 main 為 `73478bfd32a486246a2aecf3c8e59e0f3a839972`；文件更新後請記錄實際 checkout commit。**O 系列只有設計，未實作；GPU 程式開發與量測交由三個本地節點。** 本次設計電腦未執行 GPU 實驗。

## 1. 使用者已確認的目標

用來源 inference 的 trace／可取得 metrics，**自行找算子邊界、辨識 Conv2D/GEMM/BMM/LayerNorm 等類型**，建立 operation-aware 跨設備時間模型，再預測整個 request 服務時間。希望接近由 wrapper 直接提供邊界／類型／shape 的方法。不是只做完整 ResNet50 回歸，也不是只列兩種模型的誤差。

wrapper/reference 可做訓練與評分標籤；正式 T/TM predictor 不讀測試真實邊界、類型、shape。由來源 trace 推估的類型可合法作時間模型輸入。測試 target 時間／trace 只供 evaluator；預先獨立 hardware profile 可用。主要設備 **3090、4090、GB10**，5090 未列本輪必要設備。

順序是 O0 觀測性 → O1 受控算子 → O2 無標記混合序列辨識 → O3 六方向時間模型 → O4 DNN 轉移 → O5 短 dry run／metrics／設備留出。原 E 系列保留為基礎與完整 request 基線。

## 2. 先讀與現況證據

1. [一致性審查](GOAL_ALIGNMENT_REVIEW_20261021.md)：哪裡已實作、哪裡只有節點回報。
2. [完整設計](NEXT_EXPERIMENTS.md)：配置、模式、模型、對照、split、gate。
3. [資料契約](OPERATOR_DATA_CONTRACT.md)：predictor/evaluator 分離與三種時間 target。
4. [O 報告範本](OPERATOR_REPORT_TEMPLATE.md)、[既有驗證](../inference_pilot/VALIDATION.md)。
5. **[T-v1 MLP 數學規格](T_V1_MLP_MODEL_SPEC.md)**：使用者已指定MLP為主要方法；包含片段MLP＋DP、F/V時間MLP、損失、OOF與初始超參數。未知類首版採拒絕，GE與CTX使用不同模型。

Git 有 E1 的 six-phase ResNet50、E2 硬體校正、E3 12 配置及設備內 C0/C1。六 phase 不是六算子；E1 P0 是同步輔助 request 分段。E2 GEMM 不是完整 operation-aware 對照資料集。現有 trace diagnostics 有完整性疑慮，不能忽略。

使用者回報 E4 已在節點完成，六方向平均 MdAPE 42.52%／37.50%／16.70%；**本次遠端基準沒有 e4_prediction/ 或該報告，未重現**。先找實際節點程式與 session，保存 commit/patch、split、feature allowlist、逐配置結果、品質狀態，不覆蓋或從頭杜撰。若節點新結果與這份文件不同，以有 provenance 的實際證據更新狀態。

原始資料可能在節點 `/home/good_egg/RTchen/` 下；這是歷史線索，並非三節點保證存在路徑。先請當地 inventory 查證。資料／權重／trace 不在 Git。

## 3. 保護既有工作

- 更新前先看 `git status`、當地分支及未推送提交，保存工作；不要 reset/clean 覆蓋本地 E4。
- 保留 `inference_pilot/`、`e2_calibration/`、`e3_calibration/` 舊協定。O 收集／解析走新模組，不直接改 PHASES 清單冒充自動辨識。
- E3 原量測 code hash：`7743298b9e32fc40b1467174bd193d124e30fd00adcbe4ef626b8214a19b1eb9`；權重 hash：`72f27a9ddb2f3c57258816e525b12701cb0216f0e17192b780111c45ca8a8597`。原 runbook 凍結值不能因文件更新而改掉。
- 舊 hash 按 raw bytes 計算，LF/CRLF 也會影響；不要為通過驗證停用 hash 檢查。
- 不覆寫原始 sessions；重新 parse 在副本，分析寫新目錄並保存來源 hash。

## 4. 節點開發工作包

以下模組名稱是**建議的新實作**，目前不能執行 `python -m operator_study ...`。

| 工作包 | 實作內容 | 驗收／交付 |
|---|---|---|
| A 狀態對齊 | 盤點當地 E4、原始資料、環境與 metrics；共用協定 | O0 報告與三設備 support matrix，未確認不可填支援 |
| B workload manifest | pilot9、正式115/139、輸入 bytes、split/equivalence groups | 三設備相同 checksum，參數與數量檢查 |
| C `operator_study` collector | G0/GE/P/R，獨立正確性與穩定性，模式隔離 | pilot 真 GPU 正確性；無 profiler 主標籤；無逐算子 marker 的 P |
| D generic exporter/reference | CUDA 事件與 correlation，R labels、P 唯一映射 | allowlist、品質報告、R/P 多解不猜測、時間 target 清楚 |
| E `operator_analysis` | O1 時間基線、O2 分段分類、O3 分級對照 | group split、OOF、六方向、coverage／unknown、洩漏檢查 |
| F context transfer | 子圖與 ResNet50 eager taxonomy、request 組合 | isolated/context proxy/G0 分表，不漏計未支援算子 |
| G metrics/robustness | 可用 PM 子集、短 capture、三輪設備留出 | 同樣本對照、成本與可觀測性限制 |

先完成 A～D 的最小 pilot，不先做完整大型分類器。修改新 parser／量測時按資料契約補風險測試，既有 tests 也需通過。不要在沒有真 GPU 時用 mock latency 交付成功。

工作包E以T-v1規格的MLP-S/F/V實作；最小模型先在isolated資料做數值sanity，再做mixed sequence的結構loss與DP，最後做OOF時間預測。MLP是主方法，線性／倍率保留作baseline。候選片段長度與hardware profile可用性依pilot凍結。

## 5. 第一輪實際執行安排

三台共用同一 collector commit、manifest 與 input files。可先在 4090 做單配置正確性／export smoke，再在 GB10、3090 驗證 backend；**三台 pilot 都完成前不凍結大矩陣**。本機 collector 有相容性問題就記實際錯誤，不能默改精度或尺寸。

pilot 每台 9 配置×4 模式×3 repeats=108 配置量測單元，三台 324；可每 worker 掃9配置，因此基本每台12個程序 session。額外 correctness、reference audit、重測另列。先量成本與儲存量，再決定 139 配置／5,004 單元是否划算。

O0/pilot 的必要檔案：

1. 環境與硬體校正 provenance、metrics inventory、實際命令及版本。
2. input/manifest/程式 checksum、正確性與 G0/GE 穩定性。
3. 原始 trace/log 的儲存位置與 hash、allowlist 事件、完整性診斷。
4. R reference、R→P 對應、無法配對與 marker 干擾統計。
5. 單位、時間目標、失敗清單、工時與容量估算。
6. O0/O1 gate 結論：通過／受限／未完成及證據；正式矩陣是否可收。

算子短於 metrics 採樣解析度時不要延長空隙造出可分類邊界；metrics 可先不做。無法驗證 P 的邊界時停止主要辨識結論，保留 sanitized-reference 診斷與已知邊界時間研究。

## 6. 可以直接貼給本地 Codex 的任務

```text
請接手 gpu-prediction 的 O 系列節點實作。研究目標是由來源 GPU 的 trace/metrics
自行辨識 inference 內的算子邊界及類型，建立 operation-aware 跨設備時間模型，
與 wrapper 已知邊界/類型/shape 的基線公平比較。設備是 RTX 3090、RTX 4090、GB10。

先讀 AGENTS.md、docs/GOAL_ALIGNMENT_REVIEW_20261021.md、docs/NEXT_EXPERIMENTS.md、
docs/OPERATOR_DATA_CONTRACT.md、docs/CODEX_HANDOFF.md。O 系列尚未實作，請不要假設已有 CLI。
模型主線請依 docs/T_V1_MLP_MODEL_SPEC.md：MLP-S＋分段DP，以及MLP-F/V時間模型。
先檢查本地 git 狀態與 E4 工作，保存尚未推送的程式/報告，不覆蓋它們。
完成 O0 環境、trace 完整性、metrics 盤點，實作獨立的新 O collector/exporter，
先做9配置 G0/GE/P/R pilot及R→無標記P的可靠對應，不直接收全矩陣。

保留 E1～E3 程式、原始資料與凍結 hash。P 不插逐算子 marker/同步；oracle只供
訓練及 evaluator，predictor 不讀測試真實邊界/類型/shape/target時間。
辨識出的類型可供時間模型使用。單算子GE、上下文trace proxy、無profiler request G0
分開評估。未驗證資料/缺metrics/未知類/多解映射如實記錄，不補零或造數據。

請交付實作commit、必要測試、當地GPU pilot證據、失敗清單與O0/O1報告，
並列另外兩個節點需要用的同版檔案及命令。待pilot gate通過，再依設計擴到
混合序列辨識、六方向預測、DNN轉移及leave-one-device-out。
```

## 7. 回傳與 Git 整合

節點回傳程式 commit、可公開的 inventory／aggregate、split hash、prediction 表、完整報告與 limitations。原始 trace／權重／私人環境檔留本地；報告路徑需標明是節點路徑。多節點使用不同 session ID，分析保留各來源。

歷史 E 手冊仍可重現舊協定：[E2](E2_NODE_RUNBOOK.md)、[E3](E3_NODE_RUNBOOK.md)、[3090](RTX3090_NODE.md)；舊交接保留於 [archive](archive/CODEX_HANDOFF_20261020.md)。不要把歷史「下一步」當現行優先順序。
