# O4 本地節點交接：先完成量測與盤點，再建立完整迭代模型

更新：2026-11-06。**本次交付文件與 planned manifest，尚無可執行 O4 collector/model CLI。** 不要把下列介面名稱當作已存在命令。舊 E1～E3／O1 工具及其凍結 hash 保留；節點 O2/O3 先盤點，避免被較舊 Git 主線覆蓋。

## 1. 接手必讀與目標

1. [主計畫](O4_GPU_ITERATION_PLAN_ZH_TW.md)：I0～I7、時間標籤與 gate。
2. [數學規格](O4_GPU_ITERATION_MATH.md)：N0、SUM／ITER、OOF、對照。
3. [工作負載矩陣](O4_WORKLOAD_MATRIX.md)：六任務七 iteration 群、21配置。
4. [O3 最新證據](O3_Q_ABLATION_STATUS_20261106.md)：目前結果與未驗證部分。

主場景只有 **RTX4090→GB10、RTX4090→RTX3090**。4090 做 workload dry run；GI 是無 profiler、CUDA events 包住一次完整 forward 的 GPU elapsed，target_kind=`iteration_device_event_us`。目標設備 test trace／GI 不能進 predictor。G0 host 完成時間是輔助；排隊、網路及完整 LLM generation 都不在第一版主線。

## 2. 節點分工

| 節點／角色 | 必要工作 | Predictor 可用資訊 |
|---|---|---|
| RTX4090 來源節點 | 共同模型／input checksum；主來源 P；R reference；GI 供量測診斷；來源 E2 | 合法來源 P 數值＋已知 iteration scope＋硬體校正 |
| GB10 目標節點 | 同配置 GI/P/R、correctness、E2、環境與 reference audit | 只提供預先獨立 hardware profile；test workload資料交 evaluator |
| RTX3090 目標節點 | 同上，保留 clean V3／舊資料 provenance | 同上 |
| 分析節點 | 資料隔離、split/OOF、訓練、盲推論、封存預測後評分 | train/validation依角色讀；test predictor與evaluator分離 |

Training 可以用兩目標設備訓練配置的 CTX/GI；主研究是「已知設備、未見配置」預測。若要零 target workload 訓練，另列 leave-device-out，不能混稱。

## 3. 開始前保護既有資料

在節點使用現有 checkout，先看 status／branch／未推送 commits，保存當地變更。下列是已存在的 Git 讀取命令，不會啟動 GPU：

```sh
git status --short --branch
git remote -v
git log -5 --oneline
git fetch origin
git diff HEAD..origin/main -- docs/ AGENTS.md README.md
```

工作目錄乾淨且確認可快轉時才 `git merge --ff-only origin/main`；若分支分歧或有當地工作，不 reset/clean，先記錄差異並只整合需要的文件／程式。保存實際實作 commit，不以文件更新 commit 冒充 collector 版本。

需要盤點的本地歷史 run 名稱：`o3_information_controls_20261104T132810Z`、`o3_q_ablation_20261105T081458Z`、另存的 `o3_reference_pilot_gb10_20261105T083442Z`。這些名字不是保證存在的路徑。最後一份 pilot 尚未隨本次可讀資料交付，需當地確認。

## 4. 第一個實作工作包：GI＋三類 smoke

### 4.1 需新增的介面責任（待實作）

| 建議模組 | 輸入／輸出責任 | 禁止事項 |
|---|---|---|
| `o4_iteration/manifest` | 驗證 protocol、model signature、hash、mode與planned_only | 未凍結草案不可默認正式run |
| `o4_iteration/collector` | GI/P/R 同一 forward，event計時與獨立 process | 無GPU時不能以CPU數值代替；不得逐op同步 |
| `o4_iteration/adapters` | 六任務精確forward範圍，LLM cache準備，輸出correctness | 不呼叫高階image/text pipeline冒稱純forward |
| `o4_iteration/exporter` | 來源P allowlist、R參考、完整性與一對一映射 | 不用真值切P後當盲輸入 |
| `o4_analysis/compose` | 凍結模型、S/v特徵、OOF、ITER訓練／推論 | 不讀test目標資料，不補零未知工作 |
| `o4_analysis/evaluate` | CTX與GI分表、全部失敗、兩主方向配對 | 不把條件誤差當完整成功率 |

先核對節點現有功能再選擇最小新增模組，不必照此名稱重寫。實作後才產生真實 `--help`、smoke 命令與節點 runbook；本次不提供不存在的 `python -m o4_iteration ...` 命令。

### 4.2 第一輪具體交付

1. 三類各一個小配置（從既有已驗證 O1 選，保存 exact signature），三台使用同 manifest。
2. GI/P/R、三獨立程序、每程序40 measured iterations；若一cell一程序共3×3×3×3=81單元，先做單cell smoke再展開。此批不是21配置DNN矩陣，成本另計。
3. 驗證 event 單位 ms→µs、start/end stream、範圍、同步、輸出正確性及repeat穩定。
4. 核對 P/R event 數、iteration 對應、所有kernel/copy、warning／reference唯一性；為每項標 passed／warning_accepted／failed／unverified，不只給一個總PASS。
5. 保存初版 sourceexport schema，確認測試oracle刪除後 predictor 仍能解析；此時可先不訓練任何新模型。
6. 完成 ResNet RN-A 的 GI/P/R inventory，列舊三類未覆蓋的實際運算與時間；把結果交給分析節點決定 taxonomy。

Reference 未過時，交付限制與診斷。允許繼續GI工程量測或受限pilot，但不擴大成「正式逐算子已驗證」結果。

## 5. 第二個工作包：子圖與完整 ResNet

1. 凍結既有 N0／辨識器 artifact，保持各自來源 view；在新三類依賴子圖上測 zero-shot CTX。候選超過L／融合／多stream必須在輸出列出。
2. 補齊 ResNet 所需 BN/ReLU/add/pool/Linear 等受控資料。加入新類之後重訓並新增 taxonomy_version，舊三類結果不覆寫。
3. 驗證 block→stage→完整 forward 的 P span 閉合。trace 時間和與 GI 不相等時先按 scope／gap／profiler 分析，不能改 label 名稱消除差異。
4. 建立無 target test leakage 的 SUM、ITER、DIRECT、W-B/W-BC/W-BCS；按配置群產生 OOF。資料群不足就報不可訓練，改獨立 calibration 或擴資料。
5. 在兩主方向產生已封存預測，再讀 evaluator GI；依全分母報結果，不能只挑完整成功子集。
6. ResNet 完整流程成立後，按工作負載矩陣逐步擴 MobileNetV3/U-Net/YOLO26/Swin-T/Qwen3。

## 6. 建議資料產物契約

以下路徑是未來 run 的相對結構，原始大檔／權重留節點，不進 Git：

```text
run/
  protocol.json                    # draft/frozen狀態、flags、warmup與timer定義
  inventory.json                   # 所有配置、設備、模式、失敗、來源hash
  hardware_profiles/               # 非test工作負載的獨立校正
  raw/                             # capture/export/log，唯讀保存
  predictor/<opaque_id>/events.jsonl
  predictor/<opaque_id>/scope.json  # iteration範圍，不含op真值
  training/                        # 合法train labels與OOF provenance
  evaluator_only/                   # test reference、target P、GI與pairing
  quality/                         # warning、scope、single-stream與R/P audit
  splits/                          # signature/context/model留出規則
  artifacts/                       # model、normalizer、feature/taxonomy hash
  predictions/                     # predictor輸出先凍結再評分
  reports/                         # GI、CTX、coverage、成本分表
```

GI labels 至少包含 `workload_key`（evaluator使用）、device、mode、`target_kind`、timing protocol、repeat/iteration index、`elapsed_us`、聚合規則、correctness/quality、輸入／權重／collector hash。不要把 host ms、P span、GI µs 存同一 `latency`。

主 predictor 輸出每片段 event IDs／類型／CTX預測，整次 `predicted_iteration_us` 或 null、`target_kind`、time/semantic coverage status、失敗原因、模型與來源profile hash、解析／推論成本。null 不改成0；整個run失敗也要出現在attempt inventory。高softmax不等於完整性證明。

## 7. 必要的實作驗證

這些是節點實作後要執行的測試，**本次文件更新沒有宣稱它們已通過**：

- Timer測試：一個forward對一對已初始化events；單位、scope、準備與完成邊界、side-stream不誤算。
- LLM測試：每筆decode的cache_before相同，after增加1；token/position/cache容量符合manifest；重建在計時外且P/R/GI一致。
- Scope測試：YOLO adapter只執行宣告head，U-Net規格與權重種類明示；輸出正確性與完整原forward對照。
- Reference測試：多解join、漏event、巢狀重複、fused／零GPU工作、超長候選與未知schema。
- 完整性測試：missing／unknown／時間模型失敗一路傳到summary，不被硬編碼為available=True。
- 隔離測試：移除／poison target GI、reference、真實shape、config名稱不改盲預測；OOF各教師不能讀held truth選checkpoint。
- 組合測試：單stream觀測閉合、內部gap不重複計算、空預測不epsilon補齊、SUM與ITER target_kind清楚。
- 節點修改collectors/parsers後執行既有 `python -m unittest discover -s tests -v`，另加針對新風險的測試；fixture不當GPU實測。

## 8. 可直接交給節點的工作指示

```text
請在現有gpu-prediction工作區實作O4的第一個量測與盤點工作包。
先讀AGENTS.md及docs/O4_GPU_ITERATION_PLAN_ZH_TW.md、O4_GPU_ITERATION_MATH.md、
O4_WORKLOAD_MATRIX.md、O4_GPU_ITERATION_HANDOFF.md。

主目標是4090來源trace→GB10／3090的一次固定forward GPU迭代時間。
新主標籤為無profiler CUDA event elapsed，target_kind=iteration_device_event_us。
G0是host輔助，不能代替GI；LLM prefill/decode分開。

先保護本地O2/O3程式、權重、資料及未推送提交，盤點既有collector與N0 artifact。
不要重寫舊實驗或覆蓋原始run。這批Git文件是設計，沒有新增可執行O4 CLI。
先實作GI與三類小型GI/P/R smoke，通過後做ResNet RN-A完整運算inventory。
版本、權重、輸入、stream、同步、cache與adapter範圍須凍結並記checksum。

P不插逐op marker/sync，R提供train/evaluator reference；test target資料與predictor隔離。
Reference警告仍未解除時，保留warning_accepted，不能改成strict verified。
若資料不足、未知類、超長片段、side-stream或OOM，保留失敗與原因，不造時間或補0。

請交付實作commit、實際CLI/runbook、必要測試、GI/P/R小型證據、成本、reference audit、
ResNet的缺失類型清單及另外兩節點需使用的相同版本。先不要啟動完整567單元矩陣。
```

## 9. 回傳與版本管理

Git可提交收集／分析程式、schema、凍結manifest、匿名化aggregate與研究報告；原始trace、權重、環境憑證與大型預測資料留節點。可公開摘要要標記已知兩方向、資料量／群數、warning狀態、是否歷史test及實際已完成範圍。

本次文檔整合不代表已把全部節點 O2/O3 原始碼移植回主線；需要依當地實際程式、授權與依賴另作整合。後續接手不可因主線找不到某個模組就覆蓋已完成節點實驗。
