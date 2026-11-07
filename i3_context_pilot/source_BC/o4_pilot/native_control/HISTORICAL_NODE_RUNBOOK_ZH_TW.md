# RTX4090／RTX3090 原生 Trace 最小控制交接

本包是新的 native 診斷協定，不替換既有 O4 FP16 GI/P/R collector、權重或資產。GB10 已完成實測；其他設備仍需當地驗證。不要將 GB10 的執行檔複製到 x86。

## 1. 先做什麼

取得 `NODE_DIAGNOSTIC_BUNDLE.tar.gz` 與其 SHA-256，解壓至新目錄，核對 `BUNDLE_MANIFEST.json`。需要既有的 nvcc、CUDA driver、Nsight CLI、NVTX3 headers、Python3；缺少工具就回報，不由腳本安裝或更換 driver。單一 CUDA-visible GPU 是此 MRE 的執行邊界。

先保存現有工作狀態及 GPU 使用情況，不停止服務。GPU繁忙時延後效能實驗；若保留背景context，只能將這次控制作結構診斷，記錄背景限制。本輪GB10的RustDesk保留未動。

```bash
sha256sum NODE_DIAGNOSTIC_BUNDLE.tar.gz
tar -xzf NODE_DIAGNOSTIC_BUNDLE.tar.gz
cd o4_native_diagnostic
bash run_native_node.sh /absolute/path/to/NEW_diagnostic_run 4090
# 3090節點把末尾換成3090；不得用GB10數值作時間驗收門檻。
```

如需指定工具，可使用 `PYTHON=/path/to/python NVCC=/path/to/nvcc bash ...`。NVTX3 headers必須可被本地工具鏈找到；需要額外include path時，另存明示的編譯命令與source hash，不關掉NVTX冒稱同條件。未知SQLite schema3.28.0以外會停止，需要先調查並新增版本適配。

腳本先編譯、跑14項GPU-free測試，再做一個unprofiled correctness process；A1軟體P的已知工作／scope失敗就停止。其餘順序A2軟體R、A3 auto P、A4 auto R，各8 warmup＋8 measured。全程最多四個initial captures；不自動追加B/C/D、不啟動three／RN-A。

## 2. 必須核對的內容

- 每個capture應有16個普通serial kernel、8個64B memset、16個成功memset API；其中8個0B API可以無GPU activity，需依固定source／R scope核對，不能按duration猜測。
- P有8個iteration NVTX；R有8 iteration＋32 op NVTX，全部唯一閉合。每個正工作API須唯一對應GPU記錄。
- SW／auto須讀DIAGNOSTIC確認actual backend；RTX3090／4090可能都落在software，必須列 `backend_contrast_distinct=false`，不宣稱完成硬體後端驗證。
- 普通serial控制不應有GPU-GPU overlap或GPU end早於API entry；缺kernel時，不以剩餘memset的零反轉作整個時間軸通過證據。
- eight gates見 `reports/final_v4/<cell>/quality.json`。警告消失、P/R signature一致、correctness成功，都不能單獨升級Reference。

## 3. GB10已知結果與停止規則

GB10的software控制完整，但NVTX／CUDA warning皆保留；另有17個kernel start早於API entry（最大1488ns），雖然沒有end-before-entry，時間軸仍未獨立校準。hardware控制在baseline P/R及獨立repeat中都未輸出16個kernel；工具明示16筆incomplete CUPTI events dropped。顯式flush旗標及量測後200ms host drain也未恢復；但flush metadata仍顯示false，因此不能宣稱已完成有效flush的機制驗證。這是本次toolchain/harness結果，不預先要求RTX重現。

若遇到同樣問題，保留raw、完整collect/export log、request、execution與quality；先分析，不擴大矩陣。不得刪除未匹配API、裁切overlap、移動時間戳、換API猜配或以低GI誤差放行。native通過只支持本控制，不解除RN-A op140／PDL／ATen／NVTX問題。

## 4. 回傳

保留完整新run與source/binary hashes，先回傳summary、quality、reverse_API_coverage、diagnostics、commands及版本。raw留節點，透過另行約定的方式移交；Nsight raw可能含主機metadata，不直接提交Git。本包只含可公開的source／手冊／測試，不含raw、權重、執行檔或環境值。

腳本以timeout和process group管理自己啟動的程序，不清理其他服務或context；native只reset自己的CUDA context。任何失敗都保留，使用新目錄重做。
