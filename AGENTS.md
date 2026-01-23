## Python 指示

- 所有參數都必須有明確的類型標註
- 所有函式都必須有對應的說明標註，文檔註釋
- 以具有良好擴充性為目標，採用模組化設計
- 用參數決定細節，而不是在程式中寫死流程

## 專案架構

- `stereo.py`：主程式與核心流程，提供 CLI 與主要 API（計算視差、聚合、WTA）。
- `census.py`：Weighted Census Transform（WCT）與 cost volume 計算。
- `filters.py`：聚合濾波器集合（median / gaussian / bilateral）。
- `guided_filter.py`：Guided Filter 相關實作（積分圖、box filter、guided filter）。
- `stereo_io.py`：影像 I/O 與灰階轉換、尺寸檢查等工具。
- `eval_npz.py`：針對輸出 `disparity.npz` 的重新評估工具。
- `convert.py`：資料格式轉換與輸出輔助腳本。
- `README.md`：使用說明、參數說明與範例。
