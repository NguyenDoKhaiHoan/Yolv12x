# Chạy dữ liệu fog L/M/H với YOLOv12n trên Kaggle

Mở notebook có sẵn: [`notebooks/DTFA_YOLOv12n_Kaggle.ipynb`](notebooks/DTFA_YOLOv12n_Kaggle.ipynb).
Repo GitHub: https://github.com/NguyenDoKhaiHoan/Yolv12x

## Dữ liệu của bạn đã được kiểm tra

Thư mục local: `mydata/fog/fog/`. Hai lớp theo đúng thứ tự `data.yaml`:

0. `Self-exploded_insulator`
1. `broken_stockbridge`

| Split | Clear | Fog L | Fog M | Fog H | Tổng fog |
|---|---:|---:|---:|---:|---:|
| Train | 1.491 | 1.491 | 1.491 | 1.491 | 4.473 |
| Val | 213 | 213 | 213 | 213 | 639 |
| Test | 426 | 426 | 426 | 426 | 1.278 |

Mỗi `F_0001_L.jpg`, `F_0001_M.jpg`, `F_0001_H.jpg` ghép với clear `0001.jpg`
trong **cùng split**. Kiểm tra local xác nhận: đủ ba mức, kích thước/nhãn tương ứng
khớp và ID ảnh không giao nhau giữa train/val/test. Không chia ngẫu nhiên lại các ảnh
fog sau khi trộn L/M/H vì sẽ gây rò rỉ các phiên bản của cùng một cảnh.

## 1. Đưa dữ liệu lên Kaggle

Tạo Kaggle Dataset từ thư mục `mydata` (có thể nén rồi upload), giữ nguyên cấu trúc:

```text
fog/fog/
  data.yaml
  train/images/F_0001_L.jpg
  train/labels/F_0001_L.txt
  val/images/...
  val/labels/...
  test/images/...
  test/labels/...
  clear images/vesion2/
    train/images/0001.jpg
    train/labels/0001.txt
    val/images/...
    val/labels/...
    test/images/...
    test/labels/...
```

Giữ chính xác `clear images` và `vesion2` theo dữ liệu hiện có. Git chỉ chứa mã,
notebook, tài liệu; `mydata/`, weights, prepared và kết quả train được ignore.
Không cần sửa đường dẫn Linux cũ trong `data.yaml`: script prepare đọc tên lớp
và tạo YAML mới dựa trên vị trí dữ liệu thực tế trên Kaggle.

## 2. Tạo notebook

Trong Kaggle, import notebook `.ipynb` nói trên, bật GPU và Internet, thêm dataset
bằng **Add Input**. Notebook clone mã từ GitHub và cài môi trường Python 3.11 riêng
với PyTorch 2.2.2/CUDA 12.1; không cài đè PyTorch của kernel Kaggle.
Máy cần driver hỗ trợ CUDA 12.1; cell kiểm tra GPU sẽ dừng nếu không khả dụng.

Điền cấu hình trong notebook nếu không tự tìm được dữ liệu:

```python
DATA_ROOT = Path('/kaggle/input/TEN-DATASET/fog/fog')
EPOCHS = 100
BATCH = 4
IMAGE_SIZE = 640
LEVELS = ['L', 'M', 'H']
```

Đường dẫn thực tế có thể có thêm tầng thư mục tùy cách upload. Chọn thư mục chứa
`data.yaml`, `train`, `val`, `test`, `clear images`. Nên thử `EPOCHS=2` trước, rồi
chọn folder output mới để train thật. GPU thứ hai nếu có sẽ chưa được sử dụng;
training hiện tại là single GPU FP32. Giảm batch nếu thiếu VRAM.

## 3. Chuẩn bị annotation

Notebook chạy lệnh tương đương:

```bash
python -m dtfa12.prepare_data --root /kaggle/input/TEN-DATASET/fog/fog --output /kaggle/working/prepared --levels L M H
```

Script kiểm tra cặp ảnh, nhãn YOLO, kích thước, lớp, split rồi ghi JSONL. JSONL hỗ trợ
đường dẫn có khoảng trắng trong `clear images`. Không sao chép hoặc sửa ảnh gốc.

- `train_fog.jsonl`: cả 4.473 ảnh fog.
- `train_paired_clear.jsonl`: clear tương ứng từng ảnh fog, cùng thứ tự.
- `train_clear.jsonl`: 1.491 ảnh clear, không lặp ba lần, dùng cho SPT.
- Các file `val_*`, `test_*` tương ứng.
- `L.yaml`, `M.yaml`, `H.yaml`, `mixed.yaml`, `clear.yaml`: phục vụ native mAP.
- `report.json`: số lượng kiểm tra.

Muốn train riêng mức H: đổi `LEVELS=['H']` và dùng thư mục kết quả riêng. Mặc định
train một model trên cả ba mức và báo kết quả test riêng từng mức.

## 4. Thứ tự train

Notebook truyền rõ `--scale n` cho **student và SPT**, khởi tạo từ `yolov12n.pt`
bản turbo chính thức. Các bước:

1. **Baseline**: YOLOv12n học trên fog, không có teacher.
2. **SPT**: YOLOv12n học trên clear, sau đó đóng băng.
3. **AWD**: student YOLOv12n học trên fog cùng guidance từ SPT và IRT trên clear.

**AWD cần IRT pretrained.** Download checkpoint từ [README DTFA gốc](https://github.com/huruo1010/DTFA),
thêm nó vào Kaggle Input và đặt:

```python
IRT = Path('/kaggle/input/TEN-DATASET-WEIGHTS/IRT.pth')
```

Repo DTFA gốc thiếu module được import bởi script pretrain IRT, nên bản tích hợp
hiện yêu cầu checkpoint IRT có sẵn. Không lấy weights YOLOv12n/SPT làm IRT.
Chưa có IRT thì chạy baseline và SPT trước; cell AWD sẽ báo rõ điều kiện còn thiếu.

## 5. Đánh giá và lưu kết quả

Notebook tính mAP50 và mAP50–95 trên test L/M/H và mixed, cho baseline lẫn AWD.
Nếu chỉ train baseline, sửa `MODELS=['baseline']`. Không dùng test để chọn epoch;
`detector.pt` được chọn theo validation detection loss, chưa phải best validation mAP.

Output mặc định: `/kaggle/working/runs/yolov12n/` gồm `baseline`, `spt`, `awd` và `evaluation`.
Các checkpoint và metrics:

- `last.pt`: để resume đầy đủ optimizer/scheduler/bridge.
- `best.pt`: checkpoint train có validation loss thấp nhất; dùng làm SPT teacher.
- `detector.pt`: model student độc lập để predict/evaluate.
- `metrics.csv`: loss theo epoch; `evaluation/*/metrics.json`: mAP.

Lưu Kaggle version có output hoặc tải `yolov12n-results.zip` từ cell cuối.
Khi resume ở phiên khác, khôi phục runs, chạy lại setup/prepare, dùng cùng lệnh
train và thêm `--resume .../last.pt`. Giữ epochs/batch/imgsz và nội dung dữ liệu/teacher.
Teacher có thể được mount ở đường dẫn khác; checkpoint kiểm tra SHA-256 của nội dung.
Không tăng epochs trong resume vì sẽ thay đổi lịch cosine.

Notebook được kiểm tra cấu trúc và mã Python tại local; chưa thực thi trên một phiên
Kaggle GPU. Kết quả kiểm thử dữ liệu giả không phải độ chính xác trên bộ dữ liệu này.
