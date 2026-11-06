# O1 七配置挑戰集方法（程式碼版）

本分支從 [正式 139 配置方法](O1_FORMAL_MATRIX_METHOD.md) 延伸，新增與正式矩陣分開的七配置診斷。此分支只含方法、凍結 catalog／split 和測試；各節點的校準檔、結果、原始 trace 應在本地保存。

[`challenge_matrix_20261022.json`](../o1_operator/challenge_matrix_20261022.json) 凍結相同 FP16 operand bytes 的 GEMM 與 B=1 BMM、1×1 Conv、ReLU、softmax、metadata-only transpose view、顯式 `.contiguous()` 的 transpose copy。輸入使用 seed 42 的 SHAKE256→FP16 LUT，GEMM/BMM 配對的兩個 operand bytes 各自相同，只有 tensor rank 不同。所有操作為 eager、單 CUDA stream、無 autograd/autocast/CUDA Graph，計時前檢查 CPU FP32 參考輸出。

挑戰集採用 `o1_challenge_bootstrap_8192_v2_20261022` 與 `o1_challenge_calibrated_v2_20261022` 兩個獨立協定。先對七配置的 G0、GE、P 各跑一次，每程序五組、每組 8192 次；再由每組最短 host 時長校準同配置／模式的固定呼叫數，一般目標 125 ms，256 次取整，上限 131072 次。R 沿用同配置的 P 呼叫數。正式挑戰集是七配置 × 四模式 × 三個獨立程序，共 84 程序。

Metadata-only transpose view 預期零 GPU kernel。它只要求 100 次預熱；P/R 各組固定 8192 次，不要求 100 ms 組長，改以零 kernel、P 無逐算子 NVTX、R 預期 scope ID 清單完整為品質條件。G0/GE 仍校準 host 組長；GE 的事件時間對純 metadata 操作只作診斷。Nsight 的 NVTX 事件遺漏警告必須在 P/R 配對判定中保留為人工審查，不能宣稱標籤完整。

在可見單一預期 GPU 的節點上，以該節點的 GPU 名稱和**全新輸出路徑**執行：

```sh
python -m unittest discover -s tests
python -m o1_operator run --matrix challenge --bootstrap --modes G0 GE P --repeats 1 \
  --expect-gpu GB10 --out /path/to/challenge_bootstrap
python -m o1_operator calibrate --matrix challenge \
  --g0ge-root /path/to/challenge_bootstrap --p-root /path/to/challenge_bootstrap \
  --out /path/to/local_challenge_calibration.json
python -m o1_operator run --matrix challenge --modes G0 GE P R --repeats 3 \
  --expect-gpu GB10 --calibration-manifest /path/to/local_challenge_calibration.json \
  --out /path/to/challenge_full
python -m o1_operator summarize --root /path/to/challenge_full \
  --out /path/to/challenge_analysis
```

在其他節點把 `--expect-gpu` 改成當地實際 GPU 名稱，並生成當地校準檔；不得重用 GB10 校準。正式 session 須核對 protocol、catalog／split、輸入 SHA-256、程式 SHA-256、correctness、適用的組長門檻、G0/GE 三輪 spread、P/R 固定次數與 trace 警告。預定角色在 [`challenge_split_20261022.json`](../o1_operator/challenge_split_20261022.json)：1×1 Conv 和 ReLU 作 validation，配對 GEMM/BMM、softmax、view/copy 作 test；這只凍結診斷用途，不構成未知類模型成績。
