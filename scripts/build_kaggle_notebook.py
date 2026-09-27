"""Generate the checked-in Kaggle notebook without notebook dependencies."""
import json
from pathlib import Path

cells = []
def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})
def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip().splitlines(True)})

md("""
# DTFA + YOLOv12n — fog L/M/H và clear

Repo: https://github.com/NguyenDoKhaiHoan/Yolv12x

Trong Kaggle Notebook: bật **GPU** và **Internet**, rồi **Add Input** dataset chứa
`fog/fog/data.yaml`, `train`, `val`, `test`, `clear images/vesion2`.
Giữ nguyên thư mục và tên file khi upload. Không chia lại L/M/H ngẫu nhiên:
ba mức của cùng cảnh phải nằm cùng split.

Notebook tạo Python 3.11 riêng trong `/kaggle/working`, không thay PyTorch của kernel.
Ảnh nằm trong `/kaggle/input`; annotation, model và metrics ghi vào `/kaggle/working`.
Notebook tự pretrain IRT từ cặp fog/clear, vì vậy không cần tải `IRT.pth` bên ngoài.
`IRT.pth` sinh ra là checkpoint tái triển khai của dự án và được đánh giá bằng PSNR/L1.
""")
code("""
import os, sys, subprocess
from pathlib import Path

REPO = Path('/kaggle/working/Yolv12x')
if not (REPO / '.git').exists():
    subprocess.run(['git', 'clone', 'https://github.com/NguyenDoKhaiHoan/Yolv12x.git', str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
PY = '/kaggle/working/dtfa-env/bin/python'
if not Path(PY).exists():
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed',
                    '/kaggle/working/dtfa-env'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PY,
               '--index-url', 'https://download.pytorch.org/whl/cu121',
               'torch==2.2.2', 'torchvision==0.17.2'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PY,
               '-r', str(REPO / 'requirements-yolov12.txt')], check=True)
ENV = dict(os.environ, YOLO_CONFIG_DIR='/kaggle/working/yolo-config',
           MPLCONFIGDIR='/kaggle/working/matplotlib')
def run(*args):
    subprocess.run([PY, *map(str, args)], cwd=REPO, env=ENV, check=True)
run('-c', "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'Enable GPU'; print(torch.cuda.get_device_name(0))")
""")
md("""
## Cấu hình
Mặc định **YOLOv12n**, trộn cả L/M/H khi train và giữ riêng mỗi mức khi test.
Đặt `EPOCHS=2` cho lần chạy thử rồi chọn output directory mới cho lần train thật.
Batch 4 là điểm bắt đầu; giảm còn 2 nếu hết VRAM. Chỉ dùng một GPU (`cuda:0`).
Nếu auto-detect có nhiều dataset, điền `DATA_ROOT` vào thư mục chứa `data.yaml`.
IRT được tự train từ cặp fog/clear; không cần thêm file `.pth` vào Kaggle Input.
""")
code("""
DATA_ROOT = None  # ví dụ Path('/kaggle/input/your-dataset/fog/fog')
EPOCHS = 100
IRT_EPOCHS = 50
BATCH = 4
IMAGE_SIZE = 640
LEVELS = ['L', 'M', 'H']
RUNS = Path('/kaggle/working/runs/yolov12n')
PREPARED = Path('/kaggle/working/prepared')

if DATA_ROOT is None:
    candidates = [p.parent for p in Path('/kaggle/input').rglob('data.yaml')
                  if (p.parent / 'clear images' / 'vesion2').is_dir()]
    if len(candidates) != 1:
        raise ValueError(f'Set DATA_ROOT explicitly; found {candidates}')
    DATA_ROOT = candidates[0]
DATA_ROOT = Path(DATA_ROOT)
print('Dataset:', DATA_ROOT)
run('-m', 'dtfa12.prepare_data', '--root', DATA_ROOT, '--output', PREPARED,
    '--levels', *LEVELS)
""")
md("""
## Trọng số khởi tạo YOLOv12n
Tải từ release **turbo** của tác giả, tương ứng kiến trúc được pin trong requirements.
Checkpoint YOLOX hoặc YOLOv12s không thay thế được YOLOv12n.
""")
code("""
import urllib.request
WEIGHTS = Path('/kaggle/working/yolov12n.pt')
if not WEIGHTS.is_file():
    temporary = WEIGHTS.with_suffix('.download')
    urllib.request.urlretrieve('https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt', temporary)
    temporary.replace(WEIGHTS)
COMMON = ['--scale', 'n', '--classes', PREPARED / 'classes.txt', '--weights', WEIGHTS,
          '--epochs', EPOCHS, '--batch', BATCH, '--imgsz', IMAGE_SIZE,
          '--device', 'cuda:0', '--workers', 2]
""")
md("""
## 1. Baseline YOLOv12n trên fog
Baseline giúp đo phần cải thiện do DTFA. Có thể bỏ qua cell này nếu chỉ cần train AWD.
""")
code("""
run('-m', 'dtfa12.train', '--stage', 'baseline', *COMMON,
    '--train', PREPARED / 'train_fog.jsonl', '--val', PREPARED / 'val_fog.jsonl',
    '--output', RUNS / 'baseline')
""")
md("""
## 2. SPT YOLOv12n trên clear
Mỗi ảnh clear chỉ dùng một lần trong tập SPT (không lặp ba lần theo L/M/H).
""")
code("""
run('-m', 'dtfa12.train', '--stage', 'spt', *COMMON,
    '--train', PREPARED / 'train_clear.jsonl', '--val', PREPARED / 'val_clear.jsonl',
    '--output', RUNS / 'spt')
""")
md("""
## 3. Tự pretrain IRT từ fog/clear
Không cần tải file IRT của tác giả. IRT được huấn luyện bằng reconstruction L1:
fog L/M/H → clear tương ứng; validation chỉ dùng để chọn checkpoint. Mạng encoder
tương thích AFB và decoder reconstruction là tái triển khai có tên recipe riêng.
""")
code("""
IRT_RUN = RUNS / 'irt'
run('-m', 'dtfa12.train_irt', '--classes', PREPARED / 'classes.txt',
    '--train', PREPARED / 'train_fog.jsonl', '--clean-train', PREPARED / 'train_paired_clear.jsonl',
    '--val', PREPARED / 'val_fog.jsonl', '--clean-val', PREPARED / 'val_paired_clear.jsonl',
    '--epochs', IRT_EPOCHS, '--batch', BATCH, '--imgsz', IMAGE_SIZE,
    '--device', 'cuda:0', '--workers', 2, '--output', IRT_RUN)
IRT = IRT_RUN / 'IRT.pth'
run('-m', 'dtfa12.train', '--stage', 'awd', *COMMON,
    '--train', PREPARED / 'train_fog.jsonl',
    '--clean-train', PREPARED / 'train_paired_clear.jsonl',
    '--val', PREPARED / 'val_fog.jsonl', '--spt', RUNS / 'spt' / 'best.pt',
    '--irt', IRT, '--output', RUNS / 'awd')
""")
md("""
## 4. Đánh giá test theo L/M/H
Không dùng tập test để chọn epoch/hyperparameter. `detector.pt` được chọn bằng
validation detection loss; báo mAP50 và mAP50–95 native trên test ở đây.
Nếu mới train baseline, đặt `MODELS=['baseline']`.
""")
code("""
MODELS = ['baseline', 'awd']
for model_name in MODELS:
    for level in ['L', 'M', 'H', 'mixed']:
        run('-m', 'dtfa12.evaluate', '--weights', RUNS / model_name / 'detector.pt',
            '--data', PREPARED / f'{level}.yaml', '--split', 'test',
            '--imgsz', IMAGE_SIZE, '--batch', BATCH, '--device', '0',
            '--output', RUNS / 'evaluation' / f'{model_name}_{level}')
""")
md("""
## 5. Lưu kết quả / tiếp tục phiên sau
`last.pt` chứa optimizer/bridges/scheduler/RNG; `best.pt` dành cho teacher hoặc resume;
`detector.pt` chỉ có student để predict/val. File `metrics.csv` ghi loss từng epoch.

Để giữ kết quả qua phiên Kaggle: lưu notebook version có output hoặc tải archive dưới đây.
Trước khi session kết thúc, cần giữ cả folder `runs`, không chỉ `detector.pt`.
Resume: khôi phục folder runs vào cùng vị trí `/kaggle/working`, cài lại môi trường,
chạy lại bước prepare và dùng **cùng lệnh train cũ** thêm
`--resume /kaggle/working/runs/yolov12n/awd/last.pt`.
Giữ nguyên epochs, batch, image size, lớp và nội dung teacher/dataset. Không tăng epochs
khi resume vì sẽ đổi lịch cosine. Đường dẫn mount teacher có thể đổi nếu nội dung giữ nguyên.
""")
code("""
import shutil
archive = shutil.make_archive('/kaggle/working/yolov12n-results', 'zip', RUNS.parent, RUNS.name)
print('Download from notebook Output:', archive)
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
    "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell["id"] = f"dtfa12-{i:02d}"
destination = Path(__file__).resolve().parents[1] / "notebooks" / "DTFA_YOLOv12n_Kaggle.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
