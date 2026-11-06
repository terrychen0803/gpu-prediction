# O4 三節點統一工程版

Release：`o4-unified-engineering-v1-20261106`。這是離線驗證的候選工程版；新 GPU canary 尚未執行，Reference strict verified=0。主方向維持 RTX4090 P → GB10／3090 GI。所有節點使用同一 tag/commit、同一 frozen_A bytes；舊 O1/O2/O3/O4 session 不改版。

## 1. 取得同一程式

```bash
git clone --branch o4-unified-engineering-v1-20261106 --single-branch \
  https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-o4-unified
cd gpu-prediction-o4-unified
git rev-parse HEAD
git status --short
```

Release tag 是固定重現入口。整合候選分支為 `o4/unified-engineering-20261106`，main 尚未合併；不要在節點各自改程式。後續修正由新 commit/tag 發行。舊分支已封存為同名 tag，舊 GitHub tree 連結仍可讀；舊命令 `origin/<舊分支>` 改用 `refs/tags/<舊分支>`。

Linux/POSIX 為目前支援環境；Windows live pending。使用節點現有相容 CUDA torch 的**新隔離環境**，按 `o4_pilot/requirements-i1-i2.txt` 補套件；不升級 driver/CUDA/Nsys、不改系統服務。不同架構/工具版本必須記錄，不宣稱數值差異只來自 GPU。

## 2. 取得一次凍結的資產

Git 不含模型權重、原始 trace。將 GB10 已凍結的 `frozen_A/` 整目錄傳到節點，包含 ASSETS.json 與完整14檔。不要執行 prepare_assets 重生 U-Net、輸入或 token。完整索引在 `o4_pilot/manifests/a_inventory_file_sha256_20261106.json`。

ASSETS.json SHA-256：`55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4`。

同一個程式入口用三個節點參數：GPU 名稱、資產 manifest、全新輸出目錄。請在 shell 設定自己的路徑；output 必須在 checkout 外。

```bash
export PYTHON=/absolute/path/to/isolated/environment/bin/python
export O4_ASSETS=/absolute/path/to/frozen_A/ASSETS.json
export O4_DATA=/absolute/path/to/new/o4-node-runs
export O4_GPU=3090  # GB10 / 3090 / 4090
mkdir -p "$O4_DATA"
```

## 3. 離線驗證與只讀 preflight

從 repository 根目錄：

```bash
(cd o4_pilot && "$PYTHON" -m unittest discover -s tests -v)
(cd o4_pilot/vendor/legacy && "$PYTHON" -m unittest discover -s tests -v)
"$PYTHON" -m unittest discover -s tests -v
bash o4_pilot/scripts/o4_node.sh verify --gpu "$O4_GPU" \
  --assets-manifest "$O4_ASSETS" --out "$O4_DATA/verify_001"
bash o4_pilot/scripts/o4_node.sh preflight --gpu "$O4_GPU" \
  --assets-manifest "$O4_ASSETS" --out "$O4_DATA/preflight_001"
```

verify 驗完整 distribution source hashes 與14個資產檔；preflight 另做 GPU 空閒／程序 owner 檢查。兩者不啟動 CUDA workload，環境與失敗都存新目錄。其他使用者 GPU 工作、無法確認 owner、GPU忙碌時停止；不停止 K3s、Pod、Netdata、RustDesk。不要為過 gate 改 verifier 或生成本地 manifest。

## 4. 最小控制與新收集

新節點先做 software native；舊節點已有控制證據時先審查，不必為消除警告重複測相同 hardware/flush。

```bash
bash o4_pilot/scripts/o4_node.sh native --gpu "$O4_GPU" \
  --assets-manifest "$O4_ASSETS" --out "$O4_DATA/native_001"
bash o4_pilot/scripts/o4_node.sh plan --gpu "$O4_GPU" \
  --assets-manifest "$O4_ASSETS" --out "$O4_DATA/plan_001" --workload RN-A
```

plan-only 不啟動 GPU。collect 為**新 v2 協定**，RN-A 共9 cells：三模式×三程序，各50～500暖機、5 transition、40 measured。最早第60次暖機穩定。只有在 tests/assets/既有 parser regression/preflight 通過後，才由操作者明確執行：

```bash
bash o4_pilot/scripts/o4_node.sh collect --gpu "$O4_GPU" \
  --assets-manifest "$O4_ASSETS" --out "$O4_DATA/RN_A_40x3_001" --workload RN-A
```

檢查 SUMMARY.json 的 collection status、quality_status、三程序 spread、3個45-window P/R pairs、每 cell engineering_acceptance.json 和 v4_quality.json。`quality_status=review_required` 會非零退出，保留全部收集結果；strict/training永遠不自動放行。RN-A驗收後，分新run收MV/SW/UN，再YO/QP/QD；本版本發行不表示已跑這些新 GPU 工作。

可選 `smoke` 保留舊8/1/3協定，另寫 unified_audit 與 UNIFIED_SMOKE_REVIEW.json。已完成短版不必重跑。`replay --runs <舊根目錄...>` 用於舊 audit 同版數值回歸，動態計算 capture 數；v5 acceptance 額外檔不等於舊格式 byte-exact，須分開比對。

## 5. Gate、計時與證據界線

- 只有 severity=2 且內容完全相符的既知 NVTX warning 可 provisional engineering acceptance。CUDA completeness/drop、其他 warning/error、backend 不明／不符、API 非零 return、必要幾何／copy欄位缺失需阻擋工程驗收。event/stream query 的 not-ready 34/600 是明示例外。
- schema3.28/3.31只允許已知結構；v5另外驗特徵欄位和值。原 timestamp不平移、不裁 overlap，仍用 union closure。
- unmatched API／clock／semantic reference gate 各自保留。engineering acceptance並不證明submitted-work完整；strict verified=0、training_eligible=false。
- GI無profiler單次native forward；P/R時間不是GI。Qwen preparation在計時外但保留GPU工作稽核；QD每次128→129，QP0→128。
- v2 finalization 明確為釋放adapter/events、synchronize、sleep1s、worker自身context reset、ProfilerStop；這不是hardware修復成績。
- 同設備P/R須配對；不要求跨GPU kernel或Memset相同。目標test GI/P/R/reference只供evaluator。七A重複增加精度，不增加未見配置數，也不建立新確認集。

## 6. 凍結歷史版本與資料

原版 small-op/RN-A frozen 協定仍用其固定歷史 tag/checkout與原9檔資產；新七 A 用14檔索引。統一分支保留原 v1/v2 code_release manifests，新 `unified_release.json` 驗當前分發內容，解決 README 改動與舊索引衝突。歷史資料不回填為新協定。

每個 run 保存完整 O4 distribution source hashes（含vendor依賴）、commit/dirty狀態、套件與Nsys/NVCC/driver版本、asset SHA、request、raw、warnings與失敗。原始trace/權重留節點與備份，不提交Git。請交整理報告與雜湊，不能只交PASS字樣。
