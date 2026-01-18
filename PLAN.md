# Stereo Matching 復現計劃（WCT + GIF + WTA）

目標是用 Python 從零復現 paper 的主要演算法流程，包含 Census Transform、加權版 WCT matching cost、以 Guided Image Filter 進行 cost aggregation、以及 WTA 輸出視差圖。限制是不可使用現成的立體匹配或濾波演算法函式庫，但允許影像 I O 與灰階轉換使用現成函式。

## 1. 範圍與允許事項

允許使用現成函式的部分僅限於影像讀取、影像尺寸取得、灰階轉換與基本型別轉換。從建立 cost volume DSI 開始，所有演算法步驟都自行實作，包含 Census、WCT、加權 Hamming cost、box filter、integral image、Guided Image Filter、以及 WTA。

建議統一資料約定，灰階影像以 float32 表示並正規化到 0 到 1，這會讓 Guided Filter 的 ε 參數更直覺且數值更穩定。

## 2. 輸入輸出與資料流

輸入為左影像 L、右影像 R、最大視差 Dmax。輸出為視差圖 disp。中間會產生 cost volume DSI，尺寸為 H × W × Dmax。

主流程如下。

1. 讀取 L、R 並轉灰階。
2. 對每個 disparity d 建立 DSI[:, :, d]。
3. 對每個 disparity d 使用 Guided Image Filter 對 cost layer 做 aggregation。
4. 對 aggregated DSI 做 WTA，輸出 disp。

## 3. Census Transform 與 WCT matching cost

採用 9 × 9 視窗，但只取 8 個方向的路徑，每個方向取 r = 1 到 4，因此每個像素有 8 × 4 = 32 個比較點。

1. 定義 8 個方向位移集合 (dx, dy)，包含上下左右與四個對角。
2. 對每個方向取 r = 1, 2, 3, 4 形成 32 個採樣點座標 q。
3. 對每個像素 p，比較 I(q) 與 I(p) 得到 bit。
4. WCT 權重設計為距離每增加 1，權重除以 2。
   1. r = 1 使用 W
   2. r = 2 使用 W/2
   3. r = 3 使用 W/4
   4. r = 4 使用 W/8
5. Matching cost 用加權 Hamming 實作。
   1. 對 disparity d，右影像對應座標為 (x − d, y)
   2. 對 32 個比較點做 XOR
   3. 若不同則累加該點權重，若相同則加 0
   4. 得到 DSI[y, x, d]

邊界處理需一致。對於 (x − d) < 0 或比較點超出影像範圍的情況，建議直接把該 disparity 的 cost 設成大值，避免 WTA 選到無效視差。

3.1 Padding 與邊界規則

本復現採用邊界複製 padding。當 Census 或 WCT 的採樣點 q 落在影像外時，將其座標夾到最近的有效像素位置，等價於用最靠近邊界的像素值延伸填補，使每個像素都能用固定的 32 個比較點與固定的權重總和計算 cost，避免邊界區域因為比較點數量變少而造成 cost 尺度縮小。

Guided Image Filter 的 box filter 也採用相同的邊界複製規則，確保 guide I 與 cost layer p 的局部平均與變異數估計在邊界處仍維持固定視窗大小，降低邊界條紋與不連續的風險。對於視差位移造成的幾何無效情況，例如 (x − d) < 0，仍以大 cost 排除該 disparity，避免用 padding 人為補出不存在的對應。

## 4. Cost aggregation：Guided Image Filter

對每個 disparity d，把 DSI 的 cost layer 當成輸入 p，使用左影像灰階作為 guide I，對 p 做 Guided Filter 得到聚合後的 cost q。

Guided Filter 的局部線性模型為 q = aI + b。計算流程如下。

1. p = DSI[:, :, d]，I = LeftGray。
2. 使用 box filter 計算 mean_I、mean_p、mean_II、mean_Ip。
3. var_I = mean_II − mean_I^2。
4. cov_Ip = mean_Ip − mean_I · mean_p。
5. a = cov_Ip / (var_I + ε)。
6. b = mean_p − a · mean_I。
7. 對 a 與 b 各自再做一次 box filter 得到 mean_a、mean_b。
8. q = mean_a · I + mean_b，寫回 aggregated_DSI[:, :, d]。

box filter 必須自行實作，建議用 integral image 以達到每個像素 O(1) 的視窗加總。

超參數建議先用 r = 3 或 r = 4，ε 先從 1e-3 到 1e-1 的量級掃描，前提是 I 已正規化到 0 到 1。

## 5. 視差決策：WTA

對每個像素在 disparity 維度選擇聚合後 cost 最小的 d 作為視差輸出。

可同時輸出 min cost 圖作為除錯輔助，用來觀察哪些區域匹配困難或邊界處理是否正確。

## 6. 驗證策略與里程碑

為避免全流程一次接起來難以除錯，建議分段驗證。

1. WCT 單點驗證。
   1. 用小 patch 手算或列印 32 個比較點的座標
   2. 檢查權重是否符合 W、W/2、W/4、W/8
2. DSI 一致性測試。
   1. 用人造平移影像，右影像為左影像水平平移 k
   2. WTA 應該選到接近 k 的 disparity
3. Guided Filter 單獨測試。
   1. guide 用階梯邊緣影像
   2. p 用帶噪平坦圖
   3. 檢查輸出能平滑噪訊且不跨邊緣過度擴散
4. 全流程測試。
   1. 先在人造資料跑通
   2. 再換標準資料集跑 full pipeline

## 7. 程式結構建議

建議拆成下列模組以降低耦合、方便測試。

1. io.py
   1. read_image(path)
   2. to_gray(img)
2. census.py
   1. offsets 生成
   2. WCT matching cost
   3. DSI 生成
3. guided_filter.py
   1. integral image
   2. box filter
   3. guided filter
4. stereo.py
   1. pipeline 組裝
   2. WTA 輸出
5. metrics.py
   1. 若需要可加入 PBM、RMS 等評估

效能瓶頸主要在 DSI 生成與每個 disparity layer 的 Guided Filter。允許使用 NumPy 做陣列操作與逐元素運算，通常能顯著加速，但不可呼叫任何現成的濾波或 guided filter API。
