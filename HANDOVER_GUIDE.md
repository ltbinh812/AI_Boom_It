# 🎮 TÀI LIỆU CHUYỂN GIAO & HƯỚNG DẪN HUẤN LUYỆN AI BOOM IT (DUELING DQN)

Tài liệu này được biên soạn nhằm đóng gói toàn bộ cấu trúc thư mục, chức năng, cơ chế hoạt động chi tiết của từng file, hướng dẫn thiết lập môi trường, quy trình huấn luyện 3 giai đoạn và các mẹo tối ưu hóa hệ thống AI **Double/Dueling DQN** cho môi trường **Bomberland (GDGoC AI Challenge - 4 người chơi)**. 

> [!IMPORTANT]
> **QUY TẮC BẮT BUỘC:** File này phải được cập nhật **ngay lập tức** sau mỗi thao tác sau:
> - Thay đổi bất kỳ file code nào trong `my_agent/`
> - Chạy lệnh huấn luyện (`train.py`)
> - Chạy lệnh kiểm thử hoặc hiển thị trận đấu (`run_local_match`, `estimate_rankings`, ...)
> - Phát hiện vấn đề kỹ thuật hoặc thay đổi cấu hình
>
> Ghi vào mục **📋 Nhật Ký Hoạt Động** bên dưới để người tiếp quản biết chính xác trạng thái hiện tại.

---

## 📋 Nhật Ký Hoạt Động (Session Log)

*Cập nhật mới nhất ở đầu danh sách.*

### 📅 2026-06-02 (Phiên làm việc đầu tiên)

| Thời gian | Hành động | Kết quả |
| ~01:39 | Tạo file `.gitignore` lọc rác trước khi đẩy GitHub | ✅ Tạo thành công [.gitignore](file:///d:/Antigravity/AI_Boom_It/.gitignore) ở thư mục gốc dự án |
| ~01:33 | Đồng bộ hóa thư mục `my_agent` gốc theo chuẩn mới | ✅ Cập nhật [agent.py](file:///d:/Antigravity/AI_Boom_It/my_agent/agent.py) và [train.py](file:///d:/Antigravity/AI_Boom_It/my_agent/train.py) dùng `model.pth` |
| ~01:31 | Điều chỉnh format nộp bài chuẩn (`model.pth` + `requirements.txt`) | ✅ Cập nhật thư mục [submission/](file:///d:/Antigravity/AI_Boom_It/submission) và file [submission.zip](file:///d:/Antigravity/AI_Boom_It/submission.zip) theo đúng format chuẩn |
| ~01:29 | Tạo thư mục nộp bài hoàn chỉnh của riêng bạn | ✅ Đã tạo thư mục [submission/](file:///d:/Antigravity/AI_Boom_It/submission) chứa đúng 3 file (`agent.py`, `model.py`, `best_model.pth`) của riêng bạn và đồng bộ hóa file [submission.zip](file:///d:/Antigravity/AI_Boom_It/submission.zip) |
| ~01:26 | Bổ sung cảnh báo chi tiết cấu trúc phẳng khi submit | ✅ Đã cập nhật hộp cảnh báo (WARNING) trực quan vào mục 6 của tài liệu này |
| ~01:19 | Sửa hướng dẫn & nén phẳng `submission.zip` cho Agent của mình | ✅ Chạy 1-ep để sinh [best_model.pth](file:///d:/Antigravity/AI_Boom_It/my_agent/best_model.pth) và tạo [submission.zip](file:///d:/Antigravity/AI_Boom_It/submission.zip) phẳng chứa `my_agent` |
| ~01:14 | Cập nhật hệ số Reward Shaping tối ưu (train mạnh) | ✅ Đã cấu hình các thông số thưởng phạt mới trong [reward.py](file:///d:/Antigravity/AI_Boom_It/my_agent/reward.py) |
| ~01:09 | Thêm hướng dẫn nộp bài, Kaggle train & tải checkpoint | ✅ Đã cập nhật chi tiết vào phần 6 (6.1, 6.2, 6.3) và phần 8 |
| ~01:08 | Sửa lỗi `UnicodeEncodeError` trên terminal Windows | ✅ Thay thế `→` và `ε` thành ký tự ASCII chuẩn trong [utils.py](file:///d:/Antigravity/AI_Boom_It/my_agent/utils.py) |
| ~00:47 | Khởi động lại 25 episodes (test lỗi code) với `max_steps=100` | 🔄 Đang chạy ngầm để sinh checkpoint đầu tiên |
| ~20:45 | Tạo `HANDOVER_GUIDE.md` và gộp với `HUONG_DAN_TRAIN.md` | ✅ Hoàn tất, xóa file cũ `HUONG_DAN_TRAIN.md` |
| ~00:22 | Chạy thử hiển thị trực quan 1 trận: `agent/dqn_agent` vs 3 bot ngẫu nhiên | ✅ Cửa sổ Pygame mở thành công |
| ~00:23 | Bắt đầu train Giai đoạn 1: 3000 episodes vs TacticalBot | ⏹ Dừng thủ công để test nhanh hơn |
| ~00:25 | Train thử 60 episodes (~10 phút test) | ⏹ Dừng — ms/step leo từ 37ms → 510ms do CPU bị tranh chấp |
| ~00:38 | Chẩn đoán tốc độ chậm — kiểm tra tài nguyên hệ thống | ⚠️ Phát hiện Discord + trình duyệt đang ngốn CPU |
| ~00:43 | Kiểm tra checkpoint hiện có — không có `.pth` nào trong `my_agent/` | ❌ Chưa train được gì |
| ~00:44 | Hiển thị trực quan: `agent/dqn_agent` vs `TacticalRuleAgent x2` vs `GeniusRuleAgent` | ✅ Pygame chạy thành công |

### ⚠️ Trạng Thái Hiện Tại (Tính đến 2026-06-02 00:44)
*   **Checkpoint tốt nhất hiện có:** Không có — chưa hoàn thành bất kỳ episode huấn luyện nào.
*   **Chưa có file `best_model.pth`** trong `my_agent/` — AI hiện đang chạy ngẫu nhiên hoàn toàn nếu gọi `Agent.act()`.
*   **Vấn đề đang mở:** Tốc độ train trên máy này chậm (~200–500ms/step trên CPU) do tài nguyên hệ thống bị tranh chấp. Khuyến nghị train qua đêm hoặc tắt Discord/trình duyệt trước khi train.

### 💡 Bước Tiếp Theo Được Đề Xuất
```powershell
# Tắt Discord & đóng bớt trình duyệt, rồi chạy lệnh sau để bắt đầu train Giai đoạn 1:
py -3.13 my_agent\train.py --mode rule --enemy_type tactical --num_episodes 3000 --save_model
```
*Ước tính thời gian: 5–13 giờ trên CPU (khuyến nghị để qua đêm).*

---

## 📂 1. Cấu Trúc Thư Mục Dự Án

```
AI_Boom_It\
├── engine\                ← Game Engine từ BTC (Không sửa đổi)
├── agent\                 ← Rule-based bots từ BTC (Đối thủ để huấn luyện)
├── my_agent\              ← ⭐ Vùng làm việc chính của AI DQN
│   ├── agent.py           ← Bộ mã hóa trạng thái, TrainingAgent & Agent nộp bài
│   ├── model.py           ← Kiến trúc mạng nơ-ron (Dueling DQN kết hợp CNN & MLP)
│   ├── reward.py          ← Hàm tính điểm thưởng (Reward Shaping) 4 người chơi
│   ├── train.py           ← Vòng lặp huấn luyện theo giáo trình (Curriculum Learning)
│   ├── utils.py           ← Tiện ích lưu/load mô hình & vẽ biểu đồ huấn luyện
│   └── ckpts\             ← Thư mục lưu checkpoint mô hình khi train
├── scripts\               ← Công cụ kiểm thử của BTC
├── competition\           ← Bộ bảo vệ runtime của BTC
├── requirements.txt       ← Danh sách thư viện bắt buộc
└── HANDOVER_GUIDE.md      ← Tài liệu bàn giao & Hướng dẫn (File này)
```

---

## ⚙️ 2. Yêu Cầu Môi Trường & Cài Đặt (Setup)

### Môi trường hoạt động
*   **Hệ điều hành:** Windows (hoặc Linux/macOS).
*   **Python Version:** Khuyến nghị **Python 3.13** (sử dụng trình quản lý `py -3.13` trên Windows) hoặc môi trường Conda đã được thiết lập `aic_gdgoc`.

### Cài đặt Dependencies
Chạy lệnh cài đặt toàn bộ thư viện cần thiết từ thư mục gốc `d:\Antigravity\AI_Boom_It\`:
```powershell
# Cài đặt qua PIP
py -3.13 -m pip install -r requirements.txt
```
*Các thư viện chính bao gồm: `numpy`, `pygame`, `torch` (PyTorch), `tqdm`, `matplotlib` (vẽ biểu đồ), `trueskill` (tính rank).*

---

## 🔍 3. Cơ Chế Hoạt Động Chi Tiết Của Từng File

Mã nguồn AI nằm hoàn toàn trong thư mục [my_agent](file:///d:/Antigravity/AI_Boom_It/my_agent):

### 📄 3.1. [my_agent/agent.py](file:///d:/Antigravity/AI_Boom_It/my_agent/agent.py)
Chứa các thành phần chính để tương tác giữa môi trường game và mạng nơ-ron.

#### A. Hàm Mã Hóa Trạng Thế: `encode_obs(obs: dict, agent_id: int)`
Chuyển đổi dữ liệu thô (`obs` dạng dict từ môi trường) thành các Tensor đầu vào cho mạng nơ-ron:
1.  **Spatial Grid Features (Đầu vào Conv2D - 11 channels, kích thước 13x13):**
    *   `Channels 0-4`: Lần lượt biểu diễn vật lý của bản đồ gồm: Cỏ (`Grass`), Tường cứng (`Wall`), Hộp gỗ (`Box`), Item tăng bán kính bom (`Item Radius`), Item tăng sức chứa bom (`Item Capacity`).
    *   `Channel 5`: Vị trí hiện tại của chính Agent mình (1.0 tại ô đang đứng, còn lại là 0.0).
    *   `Channel 6`: Vị trí gộp của tất cả kẻ địch còn sống (Combined enemies).
    *   `Channel 7`: Thời gian đếm ngược của bom (Đã được chuẩn hóa từ $0 \to 1$, giá trị càng cao tức bom sắp nổ).
    *   `Channel 8`: Đánh dấu quyền sở hữu quả bom (1.0 nếu là bom do mình đặt, 0.0 nếu là bom đối thủ).
    *   `Channels 9-10`: Vị trí cụ thể của 2 kẻ địch gần nhất. Danh sách kẻ địch còn sống được **sắp xếp theo khoảng cách Manhattan** so với mình để giữ tính nhất quán cho các channel này.
2.  **Auxiliary Features (Đầu vào MLP - 5 scalars):**
    *   `[0] bombs_left`: Số lượng bom có thể đặt tiếp theo (chuẩn hóa theo tối đa 5 quả).
    *   `[1] radius_bonus`: Bán kính nổ hiện tại của bom mình (chuẩn hóa theo tối đa 5 ô).
    *   `[2] n_alive_foes`: Số lượng đối thủ còn sống sót trên bản đồ ($0 \to 1$).
    *   `[3] own_timer_norm`: Thời gian của quả bom mình đặt gần nhất sắp nổ ($0 \to 1$).
    *   `[4] nearest_dist_norm`: Khoảng cách Manhattan ngắn nhất đến kẻ địch gần nhất ($0 \to 1$).

#### B. Lớp Lưu Trữ Trải Nghiệm: `ReplayBuffer`
*   Được cấp phát tĩnh bằng mảng `numpy` ngay khi khởi tạo để **tránh việc cấp phát bộ nhớ liên tục trong Python**, tăng tốc độ lấy mẫu (`sample()`).

#### C. Lớp Huấn Luyện: `TrainingAgent`
*   Sử dụng thuật toán **Double DQN** nhằm giảm thiểu hiện tượng đánh giá quá cao giá trị Q (Q-value overestimation):
    $$\text{Target} = r + \gamma Q_{\text{target}}(s', \text{argmax}_{a} Q_{\text{online}}(s', a))$$
*   Đồng bộ hóa mạng Target sau mỗi $N$ episodes.
*   Sử dụng hàm lỗi **Smooth L1 Loss (Huber Loss)** để chống nhiễu gradient khi cập nhật và thực hiện **Gradient Clipping** tối đa là 10.0.

#### D. Lớp Nộp Bài: `Agent`
*   Dành riêng cho BTC chạy đánh giá (Inference).
*   **Ràng buộc khắt khe:** Chỉ chạy trên **CPU**, giới hạn luồng xử lý `torch.set_num_threads(1)` để kiểm soát tài nguyên, tự động load file `best_model.pth` (hoặc model `.pth` cuối cùng tìm thấy) cùng thư mục.
*   Có cơ chế dự phòng (fallback): Nếu load lỗi hoặc không tìm thấy file trọng số, Agent tự tạo mạng ngẫu nhiên và chọn hành động `STOP` (0) khi xảy ra lỗi đột xuất để tránh bị xử thua trực tiếp.

---

### 📄 3.2. [my_agent/model.py](file:///d:/Antigravity/AI_Boom_It/my_agent/model.py)
Định nghĩa kiến trúc mạng nơ-ron phối hợp hai luồng dữ liệu (Fusion Network).

*   **CNN Encoder Branch:** Xử lý đầu vào bản đồ 11 kênh bằng 3 tầng tích chập `Conv2d` kế hợp hàm kích hoạt `ReLU` để trích xuất các đặc trưng không gian.
*   **MLP Encoder Branch:** Xử lý vector bổ trợ 5 chiều bằng 2 tầng tuyến tính `Linear` + `ReLU`.
*   **Feature Fusion:** Nối (Concatenate) đầu ra phẳng của hai nhánh thành vector duy nhất làm đầu vào lớp dự báo.
*   **Dueling Head:** 
    *   Tách đôi lớp dự báo thành 2 nhánh con: Nhánh **Value** $V(s)$ (dự đoán chất lượng của bản thân trạng thái, đầu ra kích thước 1) và Nhánh **Advantage** $A(s, a)$ (đánh giá lợi thế từng hành động, đầu ra kích thước 6).
    *   Công thức phối hợp (Dueling DQN Formula):
        $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')\right)$$
        Cơ chế này tách biệt việc đánh giá trạng thái hiện tại tốt thế nào ($V$) và lợi ích của từng hành động cụ thể ($A$), giúp AI học nhanh hơn trong các trạng thái mà hành động không làm thay đổi trực tiếp kết quả ngay lập tức (ví dụ: đang né bom đứng chờ).

---

### 📄 3.3. [my_agent/reward.py](file:///d:/Antigravity/AI_Boom_It/my_agent/reward.py)
Chứa hàm định hình phần thưởng (Reward Shaping) để định hướng hành vi của AI.

#### Bảng thông số phần thưởng hiện tại (`REWARDS`):
*   `win` (3.0): Thắng trận (là người duy nhất còn sống).
*   `enemy_death` (1.0): Tiêu diệt được 1 đối thủ.
*   `agent_death` (-2.0): Bị chết (do tự nổ bom mình hoặc bị đối thủ gài).
*   `time_penalty` (-0.005): Phạt theo thời gian mỗi step (khuyến khích AI kết thúc game nhanh).
*   `standing_still` (-0.01): Phạt đứng im (khuyến khích di chuyển, nếu di chuyển sẽ được cộng ngược lại nhỏ).
*   `plant_near_box` (0.08): Đặt bom cạnh hộp gỗ (giúp học cách phá hộp kiếm item ở đầu game).
*   `item_collection` (0.15): Ăn được item bổ trợ tăng sức mạnh.
*   `danger_evasion` (0.12): Né thành công khỏi vùng bom nổ (được nhân hệ số 1.5 nếu bom sắp nổ khẩn cấp $\le 3$ ticks).
*   `danger_enter` (-0.06): Tự đi vào vùng bom sắp nổ khi di chuyển.
*   `own_blast_loiter` (-0.04): Đứng lì trong vùng bom nổ của chính mình đặt (phạt tăng dần khi ngòi nổ ngắn lại).
*   `approach_enemy` (0.02): Tiến lại gần đối thủ gần nhất (tính theo khoảng cách Manhattan trước và sau step).

---

### 📄 3.4. [my_agent/train.py](file:///d:/Antigravity/AI_Boom_It/my_agent/train.py)
Script cốt lõi điều khiển toàn bộ luồng huấn luyện.

*   **Chia sẻ kinh nghiệm:** Cả 4 slot người chơi trong trận đấu đều thu thập dữ liệu và đẩy chung vào **một Replay Buffer duy nhất** của `learner` (Agent ID 0).
*   **Epsilon Decay:** Bắt đầu khám phá ngẫu nhiên từ $1.0$, giảm dần theo tỷ lệ nhân $0.9995$ qua mỗi episode cho tới giới hạn tối thiểu $0.05$.

---

### 📄 3.5. [my_agent/utils.py](file:///d:/Antigravity/AI_Boom_It/my_agent/utils.py)
Cung cấp các hàm công cụ vẽ biểu đồ matplotlib lưu vào thư mục checkpoint: `loss.png` (sự hội tụ), `rewards.png` (phần thưởng tích lũy), `win_rate.png` (tỷ lệ thắng trung bình trượt 100 trận), `epsilon.png` (tiến trình giảm thăm dò).

---

## 🚀 4. Giáo Trình Huấn Luyện (3 Giai Đoạn)

### Giai đoạn 1 — Warm-up với Bot Rule-based
AI của bạn sẽ đấu với 3 bot **TacticalRuleAgent** để học cách né bom cơ bản, đặt bom, và thu thập vật phẩm.
```powershell
# Chạy từ thư mục gốc d:\Antigravity\AI_Boom_It\
py -3.13 my_agent\train.py `
    --mode rule `
    --enemy_type tactical `
    --num_episodes 3000 `
    --save_model
```
*   **Kết quả:** Checkpoint lưu vào `ckpts/rule_tactical_3000ep_42seed/`
*   **Thời gian ước tính:** ~30-60 phút (tùy CPU).

### Giai đoạn 2 — Self-Play (Đấu với bản sao chính mình)
AI sẽ đấu với các checkpoint trước của bản thân để phát triển các chiến lược tự vệ và gài bẫy nâng cao.
```powershell
py -3.13 my_agent\train.py `
    --mode selfplay `
    --num_episodes 5000 `
    --load_model ckpts\rule_tactical_3000ep_42seed\best_model.pth `
    --save_model
```
*   **Kết quả:** Checkpoint lưu vào `ckpts/selfplay_tactical_5000ep_42seed/`

### Giai đoạn 3 — Mixed (Kết hợp tối ưu)
Kết hợp 50% đánh với bot cứng nhất (`genius`) và 50% đấu self-play để duy trì khả năng đối phó với cả bot luật cứng lẫn đối thủ linh hoạt.
```powershell
py -3.13 my_agent\train.py `
    --mode mixed `
    --enemy_type genius `
    --num_episodes 10000 `
    --load_model ckpts\selfplay_tactical_5000ep_42seed\best_model.pth `
    --save_model
```
*   **Kết quả:** Checkpoint lưu vào `ckpts/mixed_genius_10000ep_42seed/`

---

## 🧪 5. Kiểm Thử AI Cục Bộ

### Xem AI đấu trực quan (có giao diện Pygame)
```powershell
py -3.13 -m scripts.participant.run_local_match `
    --agent_paths my_agent None None None `
    --visualize true `
    --num_episodes 3
```

### Đo tốc độ phản hồi (Bắt buộc dưới 100ms per step)
```powershell
py -3.13 -m scripts.participant.estimate_agent_time `
    my_agent `
    --opponents None None None `
    --num_matches 20
```

### Ước tính xếp hạng TrueSkill trên Leaderboard
```powershell
py -3.13 -m scripts.participant.estimate_rankings `
    --agent_path my_agent `
    --num_matches 100
```

---

## 📤 6. Quy Trình Nộp Bài Lên Server BTC

⚠️ **CHÚ Ý QUAN TRỌNG:** Giải đấu nghiêm cấm nộp agent tham khảo của BTC. Bạn bắt buộc phải đóng gói Agent do chính mình phát triển (`my_agent/`).

1. Đảm bảo file trọng số tốt nhất `best_model.pth` của bạn đã tồn tại trong thư mục `my_agent/` sau khi kết thúc huấn luyện.
2. Tiến hành nén zip **phẳng** (flat zip - không chứa thư mục cha) các file cấu thành của bạn bằng cách chạy lệnh sau trên PowerShell:
    ```powershell
    # Bước 2.1: Tạo một thư mục tạm thời và copy các file nộp bài vào đó
    New-Item -ItemType Directory -Path temp_submission -Force
    Copy-Item -Path my_agent\agent.py, my_agent\model.py, requirements.txt -Destination temp_submission -Force
    # Copy và đổi tên file trọng số của bạn thành model.pth đúng format nộp bài
    Copy-Item -Path my_agent\best_model.pth -Destination temp_submission\model.pth -Force

    # Bước 2.2: Nén phẳng tất cả các file trong thư mục tạm thời thành submission.zip
    Compress-Archive -Path temp_submission\* -DestinationPath submission.zip -Force

    # Bước 2.3: Xóa thư mục tạm thời sau khi nén xong
    Remove-Item -Path temp_submission -Recurse -Force
    ```

    > [!WARNING]
    > **Cơ chế nén file khi submit:**
    > Khi submit, hãy chọn trực tiếp toàn bộ các file cần submit rồi nén lại, **KHÔNG nén folder chứa các file đó**. File `agent.py` bắt buộc phải nằm tại root của file `.zip`. Nếu `agent.py` nằm bên trong một thư mục con (ví dụ: `submission/agent.py`) thì hệ thống sẽ không thể tìm thấy agent của bạn để chạy.
    >
    > **Ví dụ cấu trúc đúng trong `submission.zip`:**
    > ```text
    > submission.zip
    > ├── agent.py
    > ├── model.py
    > ├── requirements.txt
    > └── model.pth
    > ```

3. Nộp file `submission.zip` thu được lên cổng thông tin của BTC cùng Team ID và Token của bạn (qua Google Form của giải đấu).

### ⚠️ Lưu Ý Quan Trọng Về Thông Số Khi Nộp Bài (Trong Code Agent)

Để đảm bảo file nộp hợp lệ và không bị Server của BTC xử thua hoặc loại do lỗi quá thời gian (Timeout), bạn cần đảm bảo các thông số sau trong file [agent.py](file:///d:/Antigravity/AI_Boom_It/my_agent/agent.py) đã được cấu hình chính xác:

| Tên biến / Đoạn code | Giá trị bắt buộc | Tác dụng | Tại sao quan trọng? |
| :--- | :--- | :--- | :--- |
| `torch.set_num_threads(1)` | **`1`** | Giới hạn PyTorch chỉ sử dụng 1 CPU thread | **Cực kỳ quan trọng!** Server của BTC giới hạn rất nghiêm ngặt về tài nguyên CPU. Nếu sử dụng nhiều luồng, CPU sẽ bị tranh chấp tài nguyên dẫn đến bị Timeout (>100ms) và bị xử thua trực tiếp. |
| `self.device` | **`torch.device("cpu")`** | Chạy suy luận (inference) trên CPU | Server của BTC đánh giá bài làm trên môi trường CPU, không hỗ trợ GPU CUDA. Phải đảm bảo toàn bộ tensor và model đều chạy trên CPU. |
| Tên file trọng số | **`best_model.pth`** | Tên file mô hình được load tự động | Hệ thống của BTC sẽ giải nén file zip và gọi trực tiếp `agent.py`. Bạn phải đổi tên checkpoint tốt nhất của mình thành `best_model.pth` và đặt cùng thư mục để agent load được. |
| `dueling=True` | **`True`** | Bật kiến trúc Dueling DQN trong `BomberDQN` | Phải đồng bộ với kiến trúc mạng nơ-ron mà bạn đã cấu hình lúc train. Nếu lúc train dùng Dueling nhưng lúc nộp tắt đi (hoặc ngược lại) sẽ gây lỗi lệch số lượng tham số và agent bị crash. |

### 💡 6.2. Các Kỹ Thuật & Cấu Hình Để Agent Đạt Sức Mạnh Tối Đa Khi Nộp

Khi nộp bài lên Server của BTC, để Agent thi đấu hiệu quả nhất và "mạnh" nhất, bạn cần chú ý tối ưu hóa các điểm sau:

1. **Sử Dụng Bộ Não Đã Được Huấn Luyện Đầy Đủ (Best Weight):**
   * Đảm bảo file `best_model.pth` trong file nộp là file checkpoint tốt nhất thu được sau **Phase 3 (Mixed Mode)** thay vì các checkpoint thử nghiệm hoặc các Phase đầu.
   * Cách kiểm tra: Chạy `estimate_rankings` hoặc `estimate_agent_time` trên file đó để đo TrueSkill rank trước khi nộp.

2. **Chỉnh Sửa Hàm Thưởng Phạt (Reward Shaping trong [my_agent/reward.py](file:///d:/Antigravity/AI_Boom_It/my_agent/reward.py)):**
   Hành vi của AI được định hình trực tiếp qua hàm thưởng phạt. Bạn có thể tinh chỉnh các thông số sau để Agent thi đấu khôn ngoan hơn:
   * **`agent_death` (Đặt thành `-5.0` hoặc `-10.0` thay vì `-2.0`):** Tăng hình phạt khi chết. Việc này giúp AI cực kỳ thận trọng, tránh tự đặt bom nổ mình hoặc đi vào góc kẹt nguy hiểm.
   * **`plant_near_box` (Tăng từ `0.08` lên `0.15`):** Tăng phần thưởng đặt bom gần hộp gỗ để AI tích cực mở đường, nhặt item ở đầu game.
   * **`item_collection` (Tăng từ `0.15` lên `0.25`):** Tăng phần thưởng ăn vật phẩm. Item tăng bán kính bom và tăng số lượng bom là chìa khóa để chiếm lợi thế cuối game.
   * **`danger_evasion` (Tăng từ `0.12` lên `0.20`):** Tăng thưởng né bom để AI học cách chạy thoát nhanh khi bom sắp nổ.

3. **Loại Bỏ Hoàn Toàn Hành Động Ngẫu Nhiên (Epsilon = 0):**
   * Khi thi đấu thực tế, Agent cần đưa ra nước đi tối ưu 100% theo mô hình đã học thay vì đi ngẫu nhiên để thăm dò.
   * Trong lớp `Agent` ở [my_agent/agent.py](file:///d:/Antigravity/AI_Boom_It/my_agent/agent.py), hành động được chọn trực tiếp bằng cách lấy `argmax` từ mạng Q-network mà không có tham số `epsilon` (tương đương `epsilon=0.0`). Bạn **không được phép** thêm tính ngẫu nhiên vào đây khi nộp bài.

4. **Tối Ưu Hóa Mạng Nơ-ron (Trong [my_agent/model.py](file:///d:/Antigravity/AI_Boom_It/my_agent/model.py)):**
   * Nếu muốn AI thông minh hơn, bạn có thể tăng số lượng channel (filters) trong các lớp Convolution (ví dụ từ `32` lên `64` hoặc `128`) hoặc tăng số chiều ẩn (hidden dimension) của các lớp Linear.
   * **⚠️ Lưu ý cực kỳ quan trọng:** Việc tăng kích thước mạng sẽ làm tăng thời gian tính toán (Inference Time). Server BTC giới hạn thời gian phản hồi của mỗi bước đi **phải dưới 100ms**. Bạn bắt buộc phải chạy thử script `estimate_agent_time` cục bộ trên CPU để kiểm tra tốc độ trước khi nộp.

### ☁️ 6.3. Quy Trình Train Trên Server (Kaggle) & Tải Não Về Local

Do máy tính cá nhân (local CPU) train tương đối chậm, việc huấn luyện DQN trên **Kaggle** (sử dụng GPU miễn phí) là giải pháp tối ưu nhất để tiết kiệm thời gian (nhanh gấp ~10-20 lần). Dưới đây là hướng dẫn từng bước:

#### Bước 1: Đẩy mã nguồn local của bạn lên GitHub
1. Tạo một repository mới trên GitHub cá nhân của bạn (để ở chế độ Private hoặc Public).
2. Commit và Push toàn bộ mã nguồn thư mục local của bạn lên repository đó (nhất là các file trong `my_agent/` như `agent.py`, `model.py`, `reward.py`, `train.py`).

#### Bước 2: Chuẩn bị trên Kaggle
1. Đăng nhập vào [Kaggle.com](https://www.kaggle.com/) -> Click **Create New Notebook**.
2. Tại bảng **Settings** ở góc phải màn hình Notebook:
   * **Accelerator:** Chọn **GPU T4 x2** hoặc **GPU P100** (Bắt buộc bật GPU để PyTorch train nhanh).
   * **Secrets:** Nhấp vào **Add Secret**. Nhập `Label/Key` là `github_token`, và `Value` là mã **Personal Access Token (PAT)** được sinh từ tài khoản GitHub của bạn (cho phép quyền đọc repo).

#### Bước 3: Viết Code chạy train trên Notebook
Tạo các cell code trên Kaggle và chạy theo thứ tự:

* **Cell 1: Lấy GitHub Access Token từ Secrets của Kaggle:**
  ```python
  from kaggle_secrets import UserSecretsClient
  user_secrets = UserSecretsClient()
  github_token = user_secrets.get_secret("github_token")
  ```

* **Cell 2: Clone Repo của bạn về Kaggle (thay thế username và repo_name của bạn):**
  ```bash
  # Ví dụ: thay username và repo_name tương ứng
  !git clone https://{username}:{github_token}@github.com/{username}/{repo_name}.git
  ```

* **Cell 3: Di chuyển vào thư mục dự án và cài thư viện:**
  ```bash
  %cd /kaggle/working/{repo_name}
  !pip install -r requirements.txt
  ```

* **Cell 4: Chạy lệnh huấn luyện (Đảm bảo dùng GPU bằng cách bỏ cờ thiết bị nếu PyTorch tự nhận CUDA):**
  ```bash
  # Chạy Giai đoạn 1 (Ví dụ train 3000 episodes với 500 max_steps)
  !python my_agent/train.py --mode rule --enemy_type tactical --num_episodes 3000 --max_steps 500 --save_model --save_every 500
  ```

#### Bước 4: Cách thu lại bộ não đã train (Tải file `.pth` về máy local)
Sau khi script train kết thúc, các file trọng số sẽ được lưu trong thư mục `/kaggle/working/{repo_name}/ckpts/`. Bạn có 3 cách để lấy các file này về:

* **Cách 1: Tải trực tiếp qua bảng Output của Kaggle (Dễ nhất):**
  1. Trong khung bên phải của Notebook Kaggle, tìm mục **Output** -> `/kaggle/working/`.
  2. Nhấn nút **Refresh** (nút xoay vòng tròn) bên cạnh chữ Output để cập nhật các file mới sinh ra.
  3. Tìm đến thư mục `ckpts/` -> click vào file checkpoint mong muốn (ví dụ `best_model.pth` hoặc `final_*.pth`) -> Click vào dấu 3 chấm bên cạnh -> Chọn **Download**.

* **Cách 2: Nén tất cả thành 1 file zip rồi tải xuống:**
  Nếu có quá nhiều file checkpoint hoặc ảnh đồ thị muốn tải cùng lúc, hãy chạy cell code sau trên Kaggle để nén chúng:
  ```bash
  !zip -r trained_brain.zip /kaggle/working/{repo_name}/ckpts/
  ```
  Sau đó, bạn chỉ cần tải file đơn `trained_brain.zip` từ mục Output của Kaggle về máy giải nén ra.

* **Cách 3: Tải thông qua Version của Notebook (Tránh treo trình duyệt):**
  1. Nhấn nút **Save Version** ở góc trên bên phải Notebook -> Chọn **Quick Save** hoặc **Save & Run All (Commit)**.
  2. Chờ cho notebook chạy hoàn thành (nếu chọn Run All) hoặc lưu xong.
  3. Mở link Notebook đó ra -> Cuộn xuống mục **Output** ở cuối trang -> Nhấp vào file zip hoặc `.pth` cần thiết để tải xuống máy tính.

---

## 🎛️ 7. Bảng Tham Số Điều Chỉnh (Hyperparameters)

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--num_episodes` | 5000 | Số episodes huấn luyện |
| `--lr` | 5e-4 | Learning rate (giảm xuống 1e-4 nếu đồ thị loss dao động quá mạnh) |
| `--batch_size` | 128 | Kích thước batch lấy từ Replay Buffer |
| `--buffer` | 50000 | Kích thước bộ nhớ lưu trữ trải nghiệm (Replay Buffer) |
| `--epsilon_decay` | 0.9995 | Tốc độ giảm epsilon (nhỏ hơn = khám phá lâu hơn) |
| `--epsilon_min` | 0.05 | Epsilon tối thiểu (giữ ít nhất 5% cơ hội khám phá ngẫu nhiên) |
| `--target_sync` | 20 | Đồng bộ hóa trọng số từ Online sang Target network mỗi N episodes |
| `--enemy_type` | tactical | Loại Bot đối thủ khi train chế độ rule: `simple`, `smarter`, `tactical`, `genius`, `box_farmer` |
| `--no_dueling` | (tắt) | Sử dụng cờ này nếu muốn dùng DQN chuẩn (không dùng cấu trúc tách nhánh Dueling) |

---

## ⚠️ 8. Reset Thông Số Để Huấn Luyện Thật Sự (Train Mạnh)

Khi chạy huấn luyện thực tế (trên GPU/Server/Kaggle) để đạt sức mạnh tối đa thay vì chạy thử nghiệm (test nhanh cục bộ), bạn **phải reset/cấu hình lại các thông số sau** trong lệnh chạy `train.py`:

| Thông số (CLI Argument) | Chạy thử nghiệm nhanh (Local Test) | Chạy thực tế (Train Mạnh / Nộp bài) | Tại sao cần reset? |
| :--- | :--- | :--- | :--- |
| **`--max_steps`** | `100` (giảm để xong nhanh) | **`500`** (Mặc định chuẩn BTC) | Nếu chỉ chạy `100` bước, trận đấu kết thúc quá sớm. AI không học được cách sinh tồn lâu, nhặt item cuối game hoặc chiến thuật gài bẫy khi bản đồ bị thu hẹp. |
| **`--num_episodes`** | `25` hoặc `60` (test lỗi code) | **`3000`** (Giai đoạn 1) <br> **`5000`** (Giai đoạn 2) <br> **`10000`** (Giai đoạn 3) | AI cần hàng ngàn episodes để mạng nơ-ron học cách hội tụ và giảm độ ngẫu nhiên `epsilon` xuống giá trị tối thiểu `0.05`. |
| **`--save_every`** | `10` (lưu liên tục để test) | **`500`** (Mặc định) | Giảm số lần ghi đĩa I/O giúp tối ưu hóa thời gian chạy trên server, tránh việc ghi file checkpoint `.pth` quá nhiều gây chậm hệ thống. |
| **`--save_model`** | Không có (chỉ chạy test) | **Phải thêm cờ `--save_model`** | Nếu không có cờ này, khi kết thúc huấn luyện AI sẽ **không lưu** checkpoint `best_model.pth` hay `final_*.pth` nào cả. |

### Cách thực hiện:
Các thông số này được truyền trực tiếp khi khởi chạy script huấn luyện thông qua các tham số dòng lệnh CLI (không cần sửa code):
```powershell
# Ví dụ chạy Giai đoạn 1 - Train mạnh thực sự:
py -3.13 my_agent\train.py --mode rule --enemy_type tactical --num_episodes 3000 --max_steps 500 --save_model --save_every 500
```
*(Nếu muốn thay đổi mặc định trực tiếp trong file code, bạn có thể chỉnh sửa tại phần `argparse` ở cuối file [train.py](file:///d:/Antigravity/AI_Boom_It/my_agent/train.py)).*

---

## 💡 9. Mẹo Để Tối Ưu Hóa Hiệu Suất AI

1.  **Reward Shaping:** Điều chỉnh bảng hệ số `REWARDS` trong `my_agent\reward.py` để định hướng chiến lược cho AI (ví dụ: tăng phạt tự tử `agent_death` nếu AI hay tự đặt bom nổ mình, tăng phạt `standing_still` để ép AI di chuyển năng nổ hơn).
2.  **Curriculum học từng bước:** Đừng huấn luyện đấu trực tiếp với `genius` ngay từ số 0. Hãy cho AI học cách phá rương và né bom cơ bản trước ở Phase 1.
3.  **Kiểm tra đồ thị học:** Sau mỗi run huấn luyện, xem các file đồ thị PNG sinh ra trong thư mục `ckpts/` tương ứng để đảm bảo đường Loss có xu hướng giảm dần và tỷ lệ thắng (Win Rate) trung bình trượt tăng dần ổn định.
4.  **Tiếp tục từ checkpoint:** Luôn tận dụng cờ `--load_model path/to/checkpoint.pth` để huấn luyện tiếp thay vì huấn luyện từ đầu khi nâng cấp thuật toán.
