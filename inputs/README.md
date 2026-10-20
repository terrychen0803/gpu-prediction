# E1 共用輸入

正式的 ResNet50 E1 量測請使用同一份 [FP16 檔案](e1_shared_fp16_b32_seed42_20261020.bin)及相鄰的 [sidecar](e1_shared_fp16_b32_seed42_20261020.bin.json)。檔案為 B32 × 3 × 224 × 224、NCHW contiguous、little-endian；B1／B8 使用檔案前綴。不要在各設備重新執行 `prepare-input`，因 CPU `torch.randn` 在不同架構可能產生不同 bytes。

整檔 SHA-256：`0082e5f18717dcc379d22cf44250c8497c3f8ccc41763301ee8f67d7cfe7b152`。

| Batch | `input_sha256` |
|---:|---|
| 1 | `c6105d609934198afaeaf5a6badca3d8bf057321e31a3de07ed95840a94d29e2` |
| 8 | `a8fb1d59123c8018182441c199134009006811559bc560ebd13dcbc15ca46e17` |
| 32 | `0082e5f18717dcc379d22cf44250c8497c3f8ccc41763301ee8f67d7cfe7b152` |

在倉庫根目錄核對後，所有正式 `run`／`suite` 命令都帶上此參數：

```sh
sha256sum inputs/e1_shared_fp16_b32_seed42_20261020.bin
python -m inference_pilot run --label rtx3090 --expect-gpu 3090 --mode G --batch 1 \
  --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --out results/3090_b1_g_fixed_input_smoke
```

每個 run 的 manifest 會保存整檔與實際 batch 輸入 hash。沒有 `--input-file` 的舊式 CPU 隨機生成路徑不屬於本輪共用輸入 E1 協定；舊 session 不與新 session 合併。3090／4090 先跑 B1 G／P1 smoke，確認 hash 與正確性，再做正式矩陣。5090 待設備可用後使用同一檔案。
