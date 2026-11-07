# O4 I3 第一批工程與 GB10 受控 smoke

日期：2026-11-07，Asia/Taipei。依[已核對的I3計畫](../o4_i3_plan_20261106T200400Z/PLAN_ZH_TW.md)執行。本輪完成I3-0主要離線工程及I3-1的A0資產／GB10小型smoke。I3正式辨識器／N0訓練尚未完成，Reference與資料群數仍不符合條件。

## 四種狀態

| 狀態 | 實際交付 |
|---|---|
| 已完成GPU實測 | GB10九類各一個a配置、選定27cells／81measured、9組P/R各4視窗exact；含版本追溯補跑共33cells／99measured、11組P/R、22captures，沒有GPU失敗 |
| 已完成工程實作 | exact taxonomy、全量inventory／ownership、參數缺口與靜態候選、舊特徵重播、18配置固定資產、完整R參數、新smoke collector與source snapshot、split feasibility、portable handoff |
| 僅有規劃 | b配置GPU、更多signature／依賴子圖、I3 40×3、RTX節點I3實測、舊模型實際推論、新辨識器／N0／ITER與新確認集 |
| Reference未驗證 | 所有新trace保留NVTX warning；strict verified=0、formal_training_eligible=false；clock／overlap不修補，舊資料資格不變 |

這不是完整I3或O4完成報告。所用原始資料、模型、版本、舊報告全部保留；未修改GitHub、driver、CUDA、Nsight、K3s、Pods、Netdata或RustDesk。

## 1. I3-0：全量分類／事件歸屬

重算三台七A、每台三repeat的2,520個matched measured forward，含2,448,720個reference operation、2,031,600個P事件。70個觀察ATen overload全部有明確interface對應；未知overload落unmapped，不用substring fallback。matmul保留未解析批次語意，Conv的general/depthwise等細節仍需參數。名稱對應不等於語意標籤verified。

原UN-A的`upsample_bilinear2d`被`linear`子字串誤分問題已在新版本修正：4090 repeat1的17.55% busy-union比例應屬bilinear resize；3090／GB10對應約13.72%／11.65%。舊表未覆寫。新表另拆pool、activation、topk等，不能把細分類名稱相同當作已訓練分類器。

全量ownership：720筆Qwen未歸屬事件仍保留，重複歸屬0、非連續片段0。記錄事件的唯一分配不證明submitted工作全收集；舊NVTX／unmatched API／clock資格均保留。family union可能重疊，不將比例相加當GI分解。

候選長度診斷（本批development非零ATen片段，按operation數加權）：

| L | 已記錄非零op涵蓋 | 超界op |
|---|---:|---:|
| 4 | 99.4856% | 8,880 |
| 8 | 99.9791% | 360 |
| 16 | 100.0000% | 0 |
| 32 | 100.0000% | 0 |

全體比例受Qwen大量短op影響；UN-A等個別工作須看[LENGTH_BY_CASE](analysis/LENGTH_BY_CASE.csv)。L16只覆蓋這批已看過的reference，不保證新類或完整DNN，也未用此直接凍結新模型L。

## 2. 配置參數與舊模型相容性

已建立1,404個節點／工作／repeat／ATen的缺口項目。舊reference缺少stride、layout及部分scalar kwargs等，不能將shape相同當完整配置相同。ResNet靜態候選核對159個代表Conv位置，其中120項的候選參數一致、39項仍有多種stride等可能；這是結構候選，不是歷史runtime proof，也未跨Torch版本硬貼參數。詳見[ATTRIBUTE_GAPS](analysis/inventory_v2/ATTRIBUTE_GAPS.csv)與[RN靜態候選](analysis/RN_STATIC_PARAMETER_CANDIDATES.json)。

舊O3：三設備×fold0/fold4共6份P輸入重播，27維特徵與float32 normalization均一致；15detector＋15N0權重與normalizer凍結hash核對相符。新七A的21個節點配置中19個在預定第8個P iteration含非kernel事件／缺legacy geometry；另外兩個只是geometry相容，不代表舊三類、L4、硬體錨點與來源view全相容。未補零、丟事件或改權重來套用。

本輪新smoke只有3measured，不能提供舊S1的request8或S2的requests2～8，因此沒有執行凍結模型推論。這是來源協定限制，不是新trace失敗。舊E2硬體校正尚未對目前軟體堆疊重新驗證；target test工作不能用來挑校正錨點。

## 3. I3-1：受控工程及實際 GPU smoke

固定A0九類×兩個batch變化，共18配置資產；38個operand檔逐SHA驗證，約23MiB。manifest SHA：`02e4fbe40f617d571d42b55d3e343c8ea4272b6f26febefa16f0de8bc4594e69`。這些是CPU一次產生、FP16固定bytes的initialized performance-only控制，不是原DNN預訓練權重；跨節點應傳原bytes。

實測每類只選a：GI/P/R各1獨立程序，8warmup＋1transition＋3measured。CPU FP32 parity使用事先固定relative-L2≤2%且scaled-max≤2%；R另驗active TorchDispatch output exact。所有選定cells通過原門檻。下表GI僅為本輪單程序三次中位數，不具有40×3穩定性資格，也不是DNN內CTX標籤。

| 受控配置 | GI µs（development） | GI相對L2 | 最大events/ATen |
|---|---:|---:|---:|
| conv_a | 45.088 | 0.0208% | 1 |
| gemm_a | 11.520 | 0.0212% | 2 |
| bmm_a | 11.392 | 0.0209% | 1 |
| linear_a | 12.864 | 0.0208% | 1 |
| batchnorm_eval_a | 19.488 | 0.0208% | 3 |
| relu_a | 6.112 | 0.0000% | 1 |
| add_a | 6.912 | 0.0225% | 1 |
| maxpool_a | 15.040 | 0.0000% | 1 |
| adaptive_average_pool_a | 9.056 | 0.0214% | 1 |

新的R保存完整綁定ATen schema參數，含defaults、tensor stride/dtype/layout/device/storage offset及output metadata，不讀tensor內容或加逐op同步；只進reference/evaluator。P-only輸出沿用既有allowlist，檢查112個事件無真實類型、shape、bound arguments、kernel name或config ID。所有新R視窗的已記錄事件唯一歸屬；這仍不等於完整reference驗收。

選定18份P/R capture皆為明示且確認的cuda-sw、schema3.28；無CUDA drop／未知diagnostic blocker，所有NVTX warning保留。unmatched work API皆0；GPU start早於API entry合計12筆、overlap pair 2對，原timestamp完整保存。GEMM P的重疊發生在已記錄splitK與reduce事件區間；未量測實際launch attributes，不能認定PDL根因。GPU end-before-entry本批0也不能解除歷史clock問題。

### 來源快照與補跑

收集期間補了資產驗證criteria equality guard；forward函式沒有變動，但初批開頭CODE index不能單獨代表之後四個cells的module bytes。完整記錄在[SOURCE_CHANGE_NOTE](evidence/SOURCE_CHANGE_NOTE.json)。初始來源已依保存SHA逐檔重建；原批次不刪除。maxpool與adaptive-average兩配置全部GI/P/R補跑成新批次，選取依來源可追溯性，沒有按時間／分數挑選。

collector已改成每collection複製並執行獨立`source_snapshot/`。本輪總33cells／99measured含6個保留的重複cells；主表選27cells／81measured，不能把33當獨立配置。第一批與最終驗證碼的差異、原始及重跑紀錄均保留。版本命名屬development，同樣protocol的forward與量測語意不變，另保存各批實際source SHA。

## 4. 測試與回歸

- 新I3 16項、原O4 85項、legacy 54項，共155項測試通過，涵蓋resize碰撞／unknown、ownership缺失與重複、三重union、CPU輸出形狀／無in-place狀態修改、R參數／輸出parity、拒絕放寬tolerance、repeat不能湊split。
- 原O4分發副本141檔SHA完全不變；沒有修改原parser。新collector／R instrumentation為獨立I3模組與新協定。
- 獨立process重解析本輪22SQLite、11P/R pairs，319個audit產物逐byte一致；raw SHA前後相同。這證明重現性，不證明reference語意完整。
- 所有舊inventory輸入的190個SHA（索引＋63×3檔）再次核對相符；模型／raw／舊結果無改寫。
- GPU程序已結束；結束後查詢見[evidence/post_GPU.txt](evidence/post_GPU.txt)及[compute程序清單](evidence/post_compute_processes.txt)。

## 5. 為何尚未開始正式訓練

目前每類僅兩個資產配置，且GB10只測其中一個；[split feasibility](analysis/SPLIT_FEASIBILITY.json)明確拒絕train/validation/test分配。三次measured與跨設備repeat不會增加獨立signature群數。RTX4090／3090尚未執行這個I3協定，沒有新的主方向CTX配對資料。

此外NVTX與其他Reference問題尚未解除，formal_training_eligible維持false。此次無辨識器、N0、ITER、OOF或其他訓練；也無新確認集。完整operation-aware能力不能由known配置的correctness、P/R exact或低時間差代替。

## 6. 接下來執行順序與交接

1. RTX4090與3090先拿相同A0資產／source包，各跑一個BN smoke，再選有用的A0配置。不同GPU不必有相同kernel分解，也不要求一次全收齊。
2. 新受控R已能取得groups、stride等必要參數；先以配置／API／GPU工作交叉證據做獨立Reference審查。4090原七A的unmatched memset byte-count與Qwen2B copy歸屬仍需最小診斷，這批A0不能替歷史資料升級。
3. 通過工程smoke後，按資訊缺口補更多獨立signature與Conv→BN→ReLU等依賴子圖，實作新版本40×3協定，再凍結A/B/C splits。不要先擴整個21配置／567cells。
4. 保持純結構辨識器與獨立N0候選；先做oracle-boundary分類診斷，再盲分段與CTX時間對照。正式Reference不足時先交探索限制，不宣稱完成正式訓練。ITER仍待I5與有效grouped OOF資料。

本機[portable runbook](source/PORTABLE_RUNBOOK_ZH_TW.md)、[TASK_STATUS](TASK_STATUS.csv)、[STATUS](STATUS.json)與[bundle index](BUNDLES.json)可供交接。程式、資產與協定已準備；遠端節點未執行，GitHub未發布，不能將本地包描述為已可從main clone的版本。原始trace、資產與source快照留本機，無credentials寫入交付。
