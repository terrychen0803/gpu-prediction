# O4 凍結 Collector：Source Release Manifest 文件 SHA 勘誤（2026-11-06）

**範圍：** 只更新 `o4_pilot/manifests/code_release.json` 中一個非執行文件的 expected SHA-256；原版 `code_release_v1_historical.json` 原文保留。**不變更** `o4_iteration` Python/CUDA Collector、`scripts/run_node.sh`、`scripts/verify_assets.py`、78 個原始被查核檔案的實體內容、`manifests/protocol.json` 或 `manifests/assets.json`。

## RTX3090 上發現的問題

在獨立 checkout `b204ae63ee2dff2bbc161bb5cc8f9aaecfb68c01`，同一台 RTX3090 先從 GB10 取得完整原版 `O4/assets/`（其中 RN-A 102,546,585B 權重、B32 共用輸入及三個小算子 operand），執行 `scripts/verify_assets.py` 發現：

```text
Release file mismatch: docs/NEXT_STEPS_ZH_TW.md
```

使用者另執行 **完整、唯讀 78/78 source file SHA 核對**，結果是且只有：

| Path | v1 expected SHA256 | b204 source checkout actual SHA256 |
| --- | --- | --- |
| `docs/NEXT_STEPS_ZH_TW.md` | `165e958426e7749dd6e60265bf11ddf77c4e892df49d0470f07ebdb36a927410` | `480dcf1fc0f3e26e18e1175b099776e88e8742a269d605feeb52a2b3b109b029` |

Git working tree `git status --short` 空白，其餘 77 個檔案 SHA 正確。原驗證程式先驗 asset 再查 source：這只證明當次資產項目驗證未在首個文件錯誤前失敗；請在新分支重跑完整工具，取得明確 `PASS` 證據。

## 勘誤內容與保障

- `manifests/code_release_v1_historical.json` **完全複製** v1，保留歷史發行狀態。
- `manifests/code_release.json` 宣告新的 `schema=o4_github_source_release_v2_docs_reconciled_20261106`，將上述一個文件 expected SHA 更新為實際 checkout 值。加上 `release_reconciliation` provenance 欄位，提供舊／新 hash 與基底 commit。
- 量測 Protocol SHA 繼續為 `28c25fe1af6935b517d8372d9a078b30f26237d1a1471a60c7038fd43022617d`；原版資產 manifest SHA 繼續為 `0a078f3d8bd2ed790d71c518a3f295e81d78e841d0e990bb47d3a45d3f431ca3`。
- 不覆寫 RTX3090 先前 I1/I2 七組 A 的 raw 或 strict quality；原始 3-measured Development Smoke 也不會被當成 40×3 Pilot。
- 程式碼更新需在 `o4/frozen-release-doc-reconcile-20261106` 對 `main` 的獨立 PR 審查，避免將此發行修復和 I1/I2 新研究程式混為同一歷史 commit。

## RTX3090 的驗證及收集順序

獨立 `gpu-prediction-o4-frozen` checkout 且 `git status --short` 為空時，fetch/switch 到勘誤分支最新提交，然後：

```bash
cd "$HOME/RTchen/gpu-prediction-o4-frozen/o4_pilot"
export PYTHON="$(command -v python)"
"$PYTHON" scripts/verify_assets.py
"$PYTHON" -m unittest discover -s tests -q
(cd vendor/legacy && "$PYTHON" -m unittest discover -s tests -q)
nvidia-smi
PYTHON="$PYTHON" bash scripts/run_node.sh 3090 single
```

只在新 `single` 的三個 Cell 與獨立 Quality Report 審核後，才執行 `three`（27 Cell）與 `resnet`（9 Cell）。單一程序 40 measured／5 transition 和三程序 repeated 40 值均按照原版 Protocol，不以凍結八 warmup/三 measured 的 I2 Smoke 替代。**當時此文件建立時尚無 single GPU 結果；實測最新進度見下節，three/resnet 尚未執行。**

若 fail：保留新 run `analysis_runs/o4_3090_*`、`reports` 與 error log；不修改 driver 或底層程式、也不消除 warning 並改寫 verified。所有 CLI 結果的 `collector_git_commit` 應指向**勘誤分支**實際 commit，不能在報告中稱使用未修改的 b204 commit。

## 2026-11-06 RTX3090 節點後續驗證及 Single Pilot 實績

使用者在獨立 frozen checkout 更新至 `2ca671ad8c4d5b73d92efcb07c31ed62971edf0b`，與原 b204 code/protocol/assets `git diff` **空白**。實際 RTX3090 驗證結果：

- `scripts/verify_assets.py`：**PASS: 9 shared asset hashes; release code checked when available; no CUDA work**。
- O4 offline tests：**25/25 PASS**；Legacy tests：**54/54 PASS**；`pip check` 無 broken requirements。
- Preflight：NVIDIA RTX3090，driver 580.178.04，CUDA 13.0；GPU 0% utilization、無 compute processes。
- `bash scripts/run_node.sh 3090 single`：**3/3 completed**、0 failed，P/R Trace **2/2 structural passes**、**1/1 exact ordered P/R signature**，120 valid measured iterations（GI/P/R × 40），GPU overlap 0、clock inversions 0。
- 原始 run：`analysis_runs/o4_3090_single_20261106T141008Z`（保存在節點，未上傳 Git）。
- `collector_git_commit=2ca671ad8c4d5b73d92efcb07c31ed62971edf0b`；`protocol_id=o4_gpu_iteration_smoke_v2_20261106`。
- Single GEMM `gemm_16_64_64` 的單程序 40 iterations 中位數：GI **13.312 µs**，P **25.040 µs**，R **56.544 µs**。不可將 P/R 的 profiler interval 與 GI 相互代替。
- `process spread=0.00%` 僅因 repeats=1，Reporter 已註明 **未通過／重複不足**，**不是跨程序穩定性 PASS**。
- P/R 警告 **`Not all NVTX events might have been collected.`** 保留；`reference_gate=not_verified`、`strict_verified_operations=0`，不是正式 Reference。
- `REPORT.md` 標題仍寫「O4 GB10」，且模板文字寫「未取得4090／3090新資料」；這與當次 `inventory.json.expected_gpu=3090`、實測輸出不一致，屬報告模板過時。此次**不修改凍結報告生成程式**；統計以當次 inventory/snapshot/trace 為準。

**接續門檻：** 可以開始工程 `three`（三小算子 × GI/P/R × 3 independent processes = 27 Cells）；需查所有 27 Cell、重複 spread、P/R 9 對結構／NVTX／clock、正式 Reference gate。通過工程品質審核再做 `resnet`（9 Cells）。不更改 release assets、工作負載、計時協定、不混用 I2 3-measured smoke，也不會因 warning-accepted 自動升級 strict Reference。
