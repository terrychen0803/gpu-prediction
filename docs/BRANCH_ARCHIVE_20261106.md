# 分支整理與精確封存：2026-11-06

保留 `main` 與 `o4/unified-engineering-20261106`。其餘分支轉成**同名 tag**，tag 指向原分支精確 commit，建立並核對後才刪分支；commit與資料不刪。現行候選 PR #2～#6 由一個整合PR取代，原PR關閉而非合併main。

歷史GitHub `/tree/<名稱>` 可由同名tag解析。原本 `git fetch origin <名稱>` 可取tag；請用 `git checkout --detach refs/tags/<名稱>`，不再依賴 origin遠端分支。封存tag不可force更新。所有舊分支另有本機完整Git bundle備份。

| 舊分支／同名封存tag | 原始commit | 用途 |
|---|---|---|
| `docs/o4-overlap-aware-reference-gates-20261106` | `99376baae0a62e295886ea58fed702e28132214d` | 已合併或空白候選 |
| `integration/o1-o2-methods-20261106` | `b204ae63ee2dff2bbc161bb5cc8f9aaecfb68c01` | 已合併或空白候選 |
| `integration/o1-o2-v4-snapshot-20261106` | `ce97d0e0269ff90ba53f45cf70dc84197966cdcc` | 統一候選已保留其提交歷史 |
| `o1-challenge-v2-method-20261022` | `1e5eeefebafc243fcf1305fe2c5fb7f0de3255bd` | 歷史方法凍結 |
| `o1-formal139-method-20261022` | `89431b415ebf4cbe3c886795b39d7b7aba32e7a8` | 歷史方法凍結 |
| `o1-gb10-pilot-v2-20261021` | `8e6dac845bd6a1162f3beb92d6d4497bc19f0eb3` | 歷史方法凍結 |
| `o1-gb10-pilot-v4-20261022` | `915750efb8194b6b1ad872872aa76878ec93af3e` | 歷史方法凍結 |
| `o2-g0-window-pilot-20261101` | `9c1421eb72935522cae16604ccd986ef94735a41` | 歷史方法凍結 |
| `o2-mixed-sequence-method-20261024` | `77ccb5db59a359cdba1ef61bc54c48c33ffc28bb` | 歷史方法凍結 |
| `o4/frozen-release-doc-reconcile-20261106` | `c534136c0c09bcd39af4b1de10a4fcbb814e45fa` | 統一候選已保留其提交歷史 |
| `o4/i1-i2-20261106T091134Z` | `3d5d0a39de15d6637be947bcd1664abf94c17488` | 統一候選已保留其提交歷史 |
| `o4/rtx3090-evidence-compat-20261106` | `54376aa4ad6b92a438fc41facb944d0d0be74bcd` | 統一候選已保留其提交歷史 |
| `o4/seven-a-40x3-collector-20261106` | `f2fe7f44fd0d18a9bec61c9e0f6b286faecfd1ea` | 統一候選已保留其提交歷史 |
