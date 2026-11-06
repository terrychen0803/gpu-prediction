# O4 其他節點對齊與執行清單

更新：2026-11-06。GB10已完成三類27cell與RN-A9cell；4090／3090仍待執行。實作位於 [o4_pilot](../o4_pilot/README.md)，不是repo根目錄的舊E1 CLI。 [GB10報告](O4_GB10_PILOT_STATUS_20261106.md)／[機器可讀manifest](o4_gb10_pilot_20261106/alignment_manifest.json)。

## 1. 三節點角色與必須相同的項目

| 項目 | 必須對齊 | 允許差異／需保存 |
|---|---|---|
| 主方向 | 4090→GB10／3090；4090供主來源P | 其他方向僅選配 |
| collector/parser | 同一GitHub checkout與o4_pilot release hash | 原GB10本地fb74b6f與6a99701只作provenance，不可當遠端checkout ref |
| 工作 | 三個固定小算子＋RN-A native ResNet50 forward | 不自動改batch、shape、model、head或precision |
| 輸入／權重 | 共用完整bytes、sidecar及每項SHA | 不當地重生CPU隨機數或重新torch.save權重 |
| 模式 | GI/P/R；每程序40 measured＋5 transition；三獨立程序 | warmup 50～500、三個20筆窗≤5%，最早60筆通過 |
| 執行 | FP16、eval/inference_mode、eager；autocast/compile/graph關 | 保留TF32/cudnn/reduction flags |
| 完成範圍 | 同stream初始化events包一次forward，end.synchronize | 多stream unsupported；不得逐op同步 |
| 環境 | 記實際driver、torch/vision/CUDA/cuDNN/Nsys/backend | 不為表面同版盲目升級；差異時結論屬設備＋執行堆疊 |
| 品質 | 原始warning、失敗、process spread、scope及correlation全保存 | 其他GPU的kernel數／分解、duration可不同，不以GB10數值作硬門檻 |

本版live collector為Linux/POSIX；Windows live未實作。只允許一個CUDA-visible GPU；節點先確認實際名稱與環境。GB10驗證環境：Python3.12、torch2.13.0+cu130、vision0.28.0+cu130、CUDA13、cuDNN92000、driver580.159.03、Nsys2026.4.1.191，cuda-sw。這不是要求RTX盲目重裝。

## 2. 保護現有工作並取得程式

先在原checkout執行`git status --short --branch`、`git log -5 --oneline`。有未提交／分歧分支則保存，不能reset/clean。可在新的checkout取得最新main，再進入`o4_pilot/`。記錄`git rev-parse HEAD`，每個run也保存code hashes。

資料凍結：`o4_gpu_iteration_smoke_v2_20261106`，protocol SHA-256為`28c25fe1af6935b517d8372d9a078b30f26237d1a1471a60c7038fd43022617d`；asset manifest SHA-256為`0a078f3d8bd2ed790d71c518a3f295e81d78e841d0e990bb47d3a45d3f431ca3`。

## 3. 資產移交：尚未隨Git提供

向GB10取得**完全相同**的`assets/`，或完整`o4_node_bundle_fb74b6f589b8.tar.gz`再解出`O4/assets`。此包105,083,782 bytes，SHA-256：

```text
8506d01447ead75a22ce74960651ad74d70df65d4246f0c155b6c6e6ca16f849
```

這是節點間移交檔名與checksum，GitHub沒有此binary下載連結。不要假設Git clone含權重。若無法取得資產，回報`pending_assets`，不替代權重或填假時間。

RN權重SHA為`72f27a9ddb2f3c57258816e525b12701cb0216f0e17192b780111c45ca8a8597`；sidecar `524b5f9af81a86770840a364061c4ea861132da349883c3065f85ef4b8498a49`；B32共用輸入 `0082e5f18717dcc379d22cf44250c8497c3f8ccc41763301ee8f67d7cfe7b152`。六個小算子operand checksums與shape見[完整資產manifest](../o4_pilot/manifests/assets.json)。

## 3.1 三節點先做最小控制診斷（不改舊 collector）

[GB10 最新唯讀品質摘要](O4_GB10_TRACE_QUALITY_UPDATE_20261106.md) 顯示 123 對同 stream kernel-kernel overlap、1,579 筆 GPU end 早於 CPU API start、270 個成功但無 GPU Activity 的 cudaMemsetAsync，以及 NVTX warning 未排除。它們是**不同品質問題**；PDL 只是待驗證的相容機制，不是已證實根因。

4090／3090 在放大 three／RN-A 前，先核對 GB10 本地最小控制程式及輸入 SHA，完成 serial/no-PDL、64B／0B memset 對照，依 native control 計畫執行 SW／auto × P/R 四 captures（各 8 measured）。要記錄 **actual trace backend** 及 fallback，不能只根據 CLI 要求旗標認定後端獨立。必要時再加 PDL launch attribute、device completion 錨點或 op140 1×1 Conv 診斷。**這些 native MRE 的原始碼目前仍在 GB10 本地，不要假裝 GitHub 已有可執行命令。**

各台必須分別輸出 `recorded_event_integrity`（GPU→API）、`submitted_work_coverage`（API→GPU；0-byte/no-op 必須有證據）、`reference_semantic_alignment`（ATen／NVTX）、`timestamp_consistency`、`event_index_partition_supported`、`nonoverlap_sum_supported`、`overlap_aware_closure_supported`、`strict_reference_verified` 等 gate，保留 passed／warning_accepted／failed／unverified 和 evidence path。若結果仍未解釋，維持 warning-accepted／unverified，不用原封不動的已收集 trace 憑空升級品質。

這是新增診斷層，不更動已凍結的 `o4_gpu_iteration_smoke_v2_20261106`。只讀重解析採新 audit/quality overlay 版本；真正變更 Collector 行為需新 protocol ID／SHA。每台都要保留舊 O1/O2/O3 與失敗 records，先 single smoke 再決定是否擴大。

## 4. 可執行命令與順序

在repo根目錄：

```bash
cd o4_pilot
export PYTHON=/absolute/path/to/cuda-python
"$PYTHON" scripts/import_shared_assets.py --from /path/to/shared/O4/assets
"$PYTHON" scripts/verify_assets.py
"$PYTHON" -m unittest discover -s tests -v
(cd vendor/legacy && "$PYTHON" -m unittest discover -s tests -v)
bash -n scripts/run_node.sh
"$PYTHON" -m o4_iteration --help
```

只列4090計畫，不使用GPU：

```bash
"$PYTHON" -m o4_iteration run --out analysis_runs/o4_4090_plan_new_id --expect-gpu 4090 --dry-run
```

4090依序執行，single核對後才展開：

```bash
bash scripts/run_node.sh 4090 single
bash scripts/run_node.sh 4090 three
bash scripts/run_node.sh 4090 resnet
```

3090把4090換成3090；GB10為GB10。single為3cell、three為27cell、RN-A為9cell，彼此分開；每台正式smoke主兩批共36cell，不是567cell矩陣。source code checks與22＋54離線測試已有GB10證據，其他GPU仍須當地single smoke。

預設trace backend為cuda-sw，若不支援，保留錯誤與`nsys profile --help`，核對`--trace-backend cuda`後新run明示差異；不得默默fallback。每次使用新run，不覆寫。工作程序只reset自己的CUDA context，先釋放events/tensors，不做全GPU reset，不修改K3s／服務。

## 5. 審查gate與回傳

- GI每process取40筆中位數，再聚合3process；spread≤5%。超標保留全樣本並標不穩定，不挑最快process。
- P只有iteration scope NVTX；R攔截非巢狀ATen單位。保存全部kernel/copy/memset、零GPU工作、scope與API correlation。
- P/R需在**同設備、同配置**核對完整signature；跨GPU kernel一一相同不是要求。融合／拆分不同另定可對應logical group，不硬貼標籤。
- NVTX／CUDA warning保留；即使warning消失，也須審查log/report與reference，不自動verified。GB10本輪strict verified=0。
- GPU interval 重疊保留原 timestamps；區分 `event_index_partition_supported`、`nonoverlap_sum_supported=false`、`overlap_aware_closure_supported`，不能把單 stream 當無 overlap 保證。union 閉合只驗算 recorded-span，不代表 reference 已 verified；不裁切重疊、不平移時間、不補零。
- `reference_contiguous`、候選L覆蓋、GI穩定、語意與時間完整性分開。GB10 RN-A有9個op/forward超L4，需另評估taxonomy與L。

回傳完整新run：inventory/protocol、request/commands/environment、iterations/warmup/correctness、nsys-rep/SQLite、collect/export logs、quality/diagnostics、P-only exports、R reference與reports。不要只交aggregate。保存checkout SHA、code/asset hashes、backend與失敗原因；raw大檔留節點／另行傳給分析節點，不提交Git。

來源4090的test P可交predictor；目標GB10／3090 test GI／trace及真實shape/reference僅evaluator。此批均為development工程pilot，不能當未見確認集或正式DNN能力。其他五任務adapter、跨設備訓練／OOF及新確認資料另列後續。
