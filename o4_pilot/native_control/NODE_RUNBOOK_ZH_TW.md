# Software-only Native Control 交接

本版run_native_node.sh只啟動correctness＋cuda-sw P/R，總2個capture各8measured，使用原始native CUDA source；不自動做硬體trace、flush或drain。

在已安裝CUDA/NVTX3/Nsys的Linux/POSIX節點，以bash run_native_node.sh /absolute/path/to/NEW_run 4090（或3090）執行。每台自行編譯，不能複製ARM binary至x86。結果需要檢查每capture16kernel＋8個64B memset、8個已知0B API、NVTX閉合、唯一correlation與clock。

collection protocol為o4_native_software_handoff_v2_20261106；新增discard-environment，防止Nsight儲存不必要環境值。此entrypoint調整僅通過離線測試；底層native source有GB10實測。舊四capture手冊保存在HISTORICAL_NODE_RUNBOOK_ZH_TW.md供追溯，不是本版預設。沒有warning或signature一致不能自動verified；Windows live仍待實作。
