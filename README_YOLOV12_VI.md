# DTFA với baseline YOLOv12s

**Chạy YOLOv12n với dữ liệu fog/clear L/M/H của bạn trên Kaggle:** xem
[hướng dẫn riêng](KAGGLE_YOLOV12N_VI.md) và [notebook](notebooks/DTFA_YOLOv12n_Kaggle.ipynb).
Notebook truyền `--scale n`; các ví dụ YOLOv12s dưới đây vẫn dùng được với `--scale s`.

Luồng mới nằm trong `dtfa12/`. Mặc định dùng **YOLOv12s-turbo** từ repo tác giả,
không chỉ thay backbone rồi giữ head/loss YOLOX. Mã YOLOX gốc được giữ để đối chiếu.

## Kiến trúc đã chuyển

| Thành phần | Cách dùng trong bản chuyển đổi |
|---|---|
| AWD student | YOLOv12s: backbone, neck, Detect head và loss native (box, cls, DFL / TaskAlignedAssigner) |
| SPT | YOLOv12s được train trên ảnh sạch, đóng băng khi train AWD |
| IRT | Encoder phục hồi gốc DTFA, lấy stem/dark2/dark3 từ checkpoint IRT có sẵn, đóng băng |
| AFB | Lấy feature backbone tại stride 8 và 4; chiếu số kênh về 128/64 rồi dùng `AFBLoss` gốc |
| Suy luận | Chỉ YOLOv12s student; giải mã và NMS native, không chạy teacher/AFB |

Ở YOLOv12s-turbo, output backbone layer 4/2 có 256/128 kênh. IRT có 128/64 kênh.
Hai bridge độc lập cho SPT và IRT; mọi tham số bridge đều thuộc optimizer và checkpoint.
Ảnh student/SPT là RGB `[0,1]`; ảnh IRT được chuẩn hóa ImageNet như DTFA gốc.
Resize/letterbox và lật ngang luôn đồng bộ giữa hai ảnh trong một cặp.

Loss AWD: `(0.2 * L_detection + L_IRT + L_SPT) / batch_size`.
AFB giữ `alpha=1e-7`, mask ratio `0.65` từ mã train AWD gốc. Thang loss detection
đã đổi theo YOLOv12, nên hệ số 0.2 và alpha cần được đánh giá/tinh chỉnh trên validation.

Đây là bản chuyển baseline để thực nghiệm, **chưa có kết quả mAP trên dữ liệu thật**.
Training dùng một device, FP32, AdamW + cosine LR, letterbox + horizontal flip;
không phải bản tái lập toàn bộ augmentation, optimizer và lịch train của bài báo.

## Cài đặt

Dùng môi trường riêng Python 3.11. Ví dụ từ thư mục `DTFA`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-yolov12.txt
```

Với GPU, cài đúng wheel `torch==2.2.2` và `torchvision==0.17.2` cho CUDA trước khi
cài requirements. Môi trường kiểm thử cục bộ ở `../.runtime/python/python.exe` là CPU.
Không dùng `requirements.txt` cũ (PyTorch 1.10), không cài chồng `ultralytics` từ PyPI.
Fork YOLOv12 được khóa tại commit `2abab7153a065fb2925e8088e9ca2b19016ab7d6`.
FlashAttention là tùy chọn; fork có fallback sang PyTorch SDPA.

Tải trọng số khởi tạo [YOLOv12s-turbo chính thức](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt)
vào `model_data/yolov12s.pt`. Lấy checkpoint IRT từ liên kết pretrained trong
[README DTFA](https://github.com/huruo1010/DTFA#computer-usage), lưu `model_data/IRT.pth`.
Trọng số YOLOX/SPT cũ không dùng trực tiếp cho student/SPT YOLOv12; cần train lại SPT.
Nếu bỏ `--weights`, student sẽ khởi tạo ngẫu nhiên và in thông báo.

**Lưu ý mã upstream:** `train/pretrain_IRT.py` import `nets.reconstruction.model`
và `train/pretrain_SPT.py` import `nets.detection`, nhưng các thư mục này không có
trong repo đã tải. Luồng mới thay thế SPT bằng script dưới đây và yêu cầu IRT đã
pretrain. Chưa bổ sung một mạng phục hồi thay thế IRT để tránh đổi thêm phương pháp.
Loader kiểm tra đủ toàn bộ tensor stem/dark2/dark3, không âm thầm dùng teacher ngẫu nhiên.
Chưa có checkpoint IRT thật trong workspace để xác nhận định dạng của file tác giả.

## Dữ liệu

Giữ định dạng annotation TXT của DTFA (tọa độ pixel, id lớp bắt đầu từ 0):

```text
datasets/fog/000001.jpg 10,20,100,150,0 120,30,200,160,1
datasets/fog/000002.jpg
```

Dòng chỉ có đường dẫn là ảnh không có object. Đường dẫn tính từ thư mục chạy lệnh,
không chứa khoảng trắng khi dùng TXT cũ. JSONL từ `dtfa12.prepare_data` hỗ trợ khoảng trắng.
`model_data/classes.txt` chứa một tên lớp mỗi dòng, đúng thứ tự id.
Với AWD, `train_fog.txt` và `train_clear.txt` phải cùng số dòng; mỗi dòng tương ứng
phải là **cùng cảnh, cùng tọa độ và kích thước ảnh**. Nhãn lấy từ file ảnh xấu.
Không tự ghép cặp chỉ bằng vị trí nếu dữ liệu chưa được sắp đúng. Validation chỉ cần ảnh xấu.

## Huấn luyện

Chạy từ thư mục `DTFA`. Dùng `--device cpu` nếu không có CUDA; batch tùy VRAM.
Các lệnh mặc định YOLOv12s và kích thước 640.

1. Train SPT trên ảnh sạch:

```powershell
python -m dtfa12.train --stage spt --classes model_data/classes.txt --train datasets/data_info/train_clear.txt --val datasets/data_info/val_clear.txt --weights model_data/yolov12s.pt --epochs 100 --batch 8 --device cuda:0 --output runs/spt12s
```

2. Train baseline YOLOv12s trên ảnh xấu (để so sánh công bằng với AWD):

```powershell
python -m dtfa12.train --stage baseline --classes model_data/classes.txt --train datasets/data_info/train_fog.txt --val datasets/data_info/val_fog.txt --weights model_data/yolov12s.pt --epochs 100 --batch 8 --device cuda:0 --output runs/baseline12s
```

3. Train DTFA–YOLOv12s:

```powershell
python -m dtfa12.train --stage awd --classes model_data/classes.txt --train datasets/data_info/train_fog.txt --clean-train datasets/data_info/train_clear.txt --val datasets/data_info/val_fog.txt --weights model_data/yolov12s.pt --spt runs/spt12s/best.pt --irt model_data/IRT.pth --epochs 100 --batch 8 --device cuda:0 --output runs/awd12s
```

Mỗi lần chạy tạo:

- `last.pt`: student, bridges, optimizer, scheduler, metadata và RNG để resume.
- `best.pt`: checkpoint train tại epoch có **validation detection loss thấp nhất**, chưa phải best mAP.
- `detector.pt`: student tại epoch tốt nhất, định dạng native YOLOv12 để predict/val.
- `metrics.csv`: loss train, IRT, SPT và validation detection loss từng epoch.

Resume bằng cùng lệnh cũ, thêm `--resume runs/awd12s/last.pt`. Giữ cùng cấu hình,
teacher và dữ liệu; không tăng `--epochs` vì sẽ thay đổi lịch cosine ban đầu.
Thư mục output đã có dữ liệu sẽ được bảo vệ nếu không dùng resume.

## Suy luận và đánh giá

```powershell
python -m dtfa12.predict --weights runs/awd12s/detector.pt --source datasets/test --device cuda:0
```

Ảnh kết quả và nhãn dự đoán nằm dưới `runs/dtfa12-predict/`.
Không đưa `detector.pt` vào `yolo.py/get_map.py` cũ vì chúng giải mã output YOLOX.

Để tính mAP native, chuẩn bị dataset YAML và labels dạng YOLO tương ứng rồi chạy:

```powershell
yolo detect val model=runs/awd12s/detector.pt data=datasets/weather.yaml imgsz=640 device=0
```

Giữ cùng split, thứ tự lớp, image size và protocol mAP khi so baseline/AWD.
VOC mAP từ script gốc và mAP50–95 native không phải cùng một chỉ số.

## Kiểm thử

```powershell
python -m unittest discover -s tests -p test_dtfa12.py -v
```

Kiểm tra hình học cặp ảnh/nhãn rỗng, từ chối IRT thiếu tensor, gradient qua
student/bridge và đóng băng teacher, một epoch SPT/AWD với dữ liệu giả, export/reload
và suy luận student; train baseline bị ngắt rồi resume đúng epoch/scheduler.
Teacher trong test là trọng số ngẫu nhiên, chỉ để kiểm tra kỹ thuật.

Nguồn: [DTFA](https://github.com/huruo1010/DTFA),
[YOLOv12 tác giả](https://github.com/sunsmarterjie/yolov12).
