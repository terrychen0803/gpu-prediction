# S4-A Context節點交接

本run新增o4_i3_context package，不修改舊165檔source或parser。支援CBR5工作與Residual9工作，共14項、同context_scale_0一群。S4-B/C與其他尺度尚未實作／收集，不能把14項稱14個獨立樣本或六任務已支援。

分享資產時分三種：

- source：完整新source快照，含Context tests與舊回歸工具。
- shared_base：外部input、每位置weights／BN state與DAG定義；其他節點取得exact bytes，先驗SHA。
- GB10_local_inputs：GB10用shared_base在capture外產生的實際D中間FP16輸入／輸出，只是GB10本地對照證據。**不可直接當RTX本地D/I等價輸入**。RTX須在自己的GPU/stack下，以相同shared_base另行derive並保存hash；不同GPU中間bytes可能不同。

先在當地無其他使用者compute工作的情況，以gpu_guard與monitor包住derive（GPU資產準備，不是正式量測）；參考本run的derive_driver.py。當地Python、GPU identity、paths應另設，不能直接跑含GB10路徑的driver。

在source目錄設定PYTHONPATH=$PWD:$PWD/o4_pilot，使用相容CUDA Python與nsys：

```bash
"$I3_PYTHON" -m o4_i3_context.assets verify-base --path /fixed/shared_base/BASE.json
"$I3_PYTHON" -m o4_i3_context.assets derive --base /fixed/shared_base/BASE.json --path /new/local_assets --expect-gpu 4090
"$I3_PYTHON" -m o4_i3_context.assets verify --path /new/local_assets/ASSETS.json
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests_context -v
```

3090把GPU改3090。derive先外部GPU guard/monitor，失敗目錄保留、另立新out，不覆寫或以CPUfixture冒稱GPU。CPU同時跑tests_v2、tests、o4_pilot/tests及o4_pilot/vendor/legacy/tests；collector/node支援只以實際測試結果判定。套件內CONTEXT_RUNBOOK_ZH_TW.md含CLI與協定。

批次ID見SCOPE.json，CBR5、Residual9，各先plan-only再新out Smoke。每工作GI/P/R三cell completed、correctness/warmup合格、monitor_review=false、P/R唯一pair measured_exact=3且all_windows_exact，才以同清單新out跑formal40×3；工程trace gate必須accepted。actual backend預期cuda-sw，不silent fallback。

正式每工作9cells，每mode三個獨立process、40measured；spread>5%保留review，不挑最快repeat。R每iteration CBR3／Residual7個完整ATen位置（S各1），需檢查完整參數、ownership、submitted-work、correlation及P/R mapping。時間戳、overlap、未匹配API保留，不平移／裁切或猜0B。

D沿DAG傳值；I所有位置使用native D預算輸入、斷data edge但保留原次序與stream；S逐位置同一份輸入／權重／state。所有輸出保留到下次prepare，在測量外清除，無逐位置同步或P marker。此設計只讓對照更可解釋，不能將D/I差異全歸因依賴：cache、input placement、allocator/workspace、lifetime仍是限制。

回傳原始完整collections、source snapshots、protocol、environment、全部失敗與監測、shared/local assets manifest和hash、實際命令。原raw不改。Reference全部仍unverified；development用途不代表正式訓練資格。Predictor只讀P numeric allowlist，R真類型/shape/邊界與target test GI/trace只供evaluator。

本run發生對話中斷：原collections/cbr_formal留原bytes與當時running狀態；active完整版本為cbr_formal_recovery。active analysis view只包含cbr_smoke、cbr_formal_recovery、residual_smoke、residual_formal，避免重複計數。不刪除原中斷目錄、不可把複製42cells當新獨立量測。其他節點不要複製這種中斷狀態當已完成；只記錄自己的實際執行。

本輪沒有在RTX實測，沒有新模型／N0／ITER訓練。可攜包尚未發布GitHub，完成時SHA見BUNDLES.json；舊Core36/S3 release無Context package，不能用它們的CLI假裝支援S4。
