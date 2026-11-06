# RTX4090／RTX3090：I1 Software Audit與七A配置交接

狀態：GB10已實測；RTX兩節點未執行。Linux/POSIX source可移植，Windows live未實作。不要複製ARM binary到x86，當地編譯；不要依GB10的kernel名稱／數量／duration驗收DNN。

## 1. 版本與資產

使用本次獨立分支／commit的o4_pilot，並讀目前main已合併的PR #1規劃；不合併或覆蓋原O1/O2/O3。先記git status、checkout SHA與所有未提交修改，必要時建立新checkout，不reset/clean。

取得GB10 `assets/frozen_A/`完整14檔；Git只有ASSETS manifest與SHA index，不含權重。其ASSETS.json SHA為`55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4`。包含ResNet/MV/Swin/YOLO/Qwen原權重、一次初始化UNet、224/256/640輸入、固定token JSON與Qwen config/revision。**不在其他CPU重新prepare/init/torch.save。** 從Git複製manifest而沒有資產不是ready。

新venv可以繼承既有CUDA torch，再安裝requirements-i1-i2.txt；不升級driver/CUDA/Nsys或替換既有torch。版本／source hash差異會與GPU差異混合，必須保存。只允許單一GPU；unknown workload、其他user程序或忙碌GPU會阻擋，不能停服務以過gate。

## 2. Software Native Control

從o4_pilot：`bash native_control/run_native_node.sh /absolute/new/native_run 4090`（3090換尾端）。新入口只有correctness＋cuda-sw P/R，各8measured。原始source、known16kernel/8positive-memset/8zero-call審查與八項gate都附帶。

GB10的hardware0/80kernel限制未解除，本輪不自動收硬體trace。若其他節點另做工具對照，須獨立決策與新run，記actual backend／fallback；不能從指定cuda旗標推斷硬體路徑。DNN不要求與GB10相同kernel實作。

## 3. 直接參數觀測

依o4_research/README.md編譯SDK observer，先64B／0B控制再RN-A（都加`--expect-gpu 4090`或3090）。不可置於nsys profile下。CUPTI subscribe或工具缺失就報pending／failed，不能假裝讀到參數。RN-A op index可能因軟體stack改變而不同，須先核對ATen/shape，不把GB10的op140直接貼過去。

## 4. 小型Adapter Smoke

驗證ASSETS hash與每個檔案後，跑repo tests與vendor/legacy tests。先`--plan-only --workload RN-A`，再RN-A實測；核對correctness、scope、correlation、GPU coverage及warning後才另跑其餘六A。命令見o4_research/README.md。每A三mode、一process、3measured，不能稱三repeat穩定性或正式GI資料集。

UNet為initialized_performance_only；YOLO scope包含native auxiliary head與decode/topk、沒有外部NMS；QP/QD cache0→128／128→129，prefix/cache/position/mask準備在計時外，GPU準備另存。任何unsupported/OOM/precision錯誤保留，不縮batch或換model。

## 5. 驗收與回傳

分別回傳GPU實測完成、工程實作完成、planned-only、Reference-unverified；不能只有一個PASS。八項gate保留warning/unverified，P/R相同或低GI不自動verified。raw timestamp不shift/crop；只能在新audit目錄做union閉合。

保留完整新run：request/protocol/source與asset hashes、environment、correctness、warmup與iteration events、nsys-rep/SQLite、collect/export logs、reverse API、v4 quality、numeric P、evaluator R、prepare scopes與失敗。raw可能含主機metadata，留節點透過另行約定方式移交；Git只交整理摘要。測試目標GB10／3090 P/GI/reference不可進4090主predictor。

本次package source與Native entrypoint均通過離線測試；底層7A forward在GB10已測。RTX與Windows狀態沒有被升級；後續MLP訓練／OOF／新確認集仍未執行。
