# O4 統一工程入口

新節點請使用 [三節點 Quickstart](../docs/O4_QUICKSTART_ZH_TW.md)、[狀態](../docs/O4_NODE_STATUS.md) 與 [版本說明](../docs/O4_RELEASE_NOTES.md)。共用入口 `bash o4_pilot/scripts/o4_node.sh` 從repo根目錄執行；固定release tag與資產SHA。

新七A40×3 v2／native v3尚未GPU實測，strict verified=0。以下為原pilot與先前交付歷史入口，不代表新的統一版已完成GPU驗收。

# I1／I2 新增工程工具（2026-11-06）

最新結果見 [I1 Reference／七A配置實測](../docs/O4_I1_REFERENCE_I2_ADAPTER_STATUS_20261106.md) 與 [RTX交接](../docs/O4_I1_I2_NODE_HANDOFF_20261106.md)。新增可執行工具在 [o4_research](o4_research/README.md)，Native入口預設只做 [software control](native_control/NODE_RUNBOOK_ZH_TW.md)。七A的21個小型GI/P/R cell已在GB10完成；op140獨立replica直接讀到0-byte，歷史標記未回填，strict verified仍為0。RTX兩節點與正式MLP訓練未執行。

下方保留原frozen pilot說明及其協定；本次新development協定不回填為原40筆／三程序資料。

# O4 可執行工程 pilot

本目錄提供已在GB10完成實測的GI/P/R collector、保守SQLite parser與運算inventory。主方向4090→GB10／3090。原O1/E1程式保留，O4由本子目錄獨立執行。

- [GB10結果與品質限制](../docs/O4_GB10_PILOT_STATUS_20261106.md)
- [跨節點對齊清單與命令](../docs/O4_NODE_ALIGNMENT_20261106.md)
- [詳細操作手冊](docs/NODE_RUNBOOK_ZH_TW.md)／[資料語意](docs/DATA_SCHEMA_ZH_TW.md)

支援Linux/POSIX live收集；Windows live尚未實作／驗證。其他設備GPU實測待完成。只涵蓋三個小算子與RN-A，沒有六任務完整adapter或跨設備DNN模型。

```bash
cd o4_pilot
export PYTHON=/absolute/path/to/cuda-python
"$PYTHON" scripts/import_shared_assets.py --from /path/to/shared/O4/assets
"$PYTHON" scripts/verify_assets.py
"$PYTHON" -m unittest discover -s tests -v
(cd vendor/legacy && "$PYTHON" -m unittest discover -s tests -v)
bash scripts/run_node.sh 4090 single
```

assets不進Git。必須取得原GB10資產bytes，checksum由manifests/assets.json固定；不能重新torch.save或當地隨機生成後冒稱對齊。可從GB10交接包解出O4/assets，或使用其他已核對SHA的副本。import工具只複製／驗證資產，不下載、不執行GPU。通過single後才three／resnet；新run目錄拒絕覆寫。

`manifests/code_release.json`核對本Git子目錄檔案；本地原版fb74b6f是來源provenance，不是本遠端repo可checkout的commit。節點記錄實際GitHub checkout SHA與code hashes。原始trace／SQLite／權重及analysis_runs不提交Git。
