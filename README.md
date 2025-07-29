# Môn Học: Nhập Môn Xử Lý Ảnh Số
## Đếm Phương Tiện Giao Thông - Vehicle Counting

## XÁC ĐỊNH ĐỐI TƯỢNG TRONG ẢNH
- **Sinh Viên Thực Hiên:** Phạm Thế Hùng, Nguyễn Chí Dũng, Đoàn Hoàng Quân, Phan Trần Quang Thái 
- **Môn Học:** Nhập Môn Xử Lý Ảnh Số
- **Giảng viên:** TS.Đỗ Hữu Quân

## Tổng quan dự án

Đây là một hệ thống đếm xe thông minh được phát triển cho môn học **Nhập Môn Xử Lý Ảnh Số**. Hệ thống sử dụng công nghệ **YOLOv5** để phát hiện đối tượng và **DeepSORT** để theo dõi đối tượng, cho phép đếm chính xác số lượng các loại phương tiện đi qua một vạch đếm được định nghĩa trước.

### Tính năng chính

- **Phát hiện đối tượng**: Sử dụng YOLOv5 để phát hiện 3 loại phương tiện: xe buýt (bus), xe hơi (car), xe máy (motorbike)
- **Theo dõi đối tượng**: Sử dụng DeepSORT để theo dõi từng đối tượng qua các khung hình
- **Đếm thông minh**: Chỉ đếm khi đối tượng cắt qua vạch đếm, tránh đếm trùng lặp
- **Hiển thị real-time**: Bounding box, ID tracking, và counter được hiển thị trực tiếp
- **Xuất kết quả**: Video output với thống kê chi tiết

### Công nghệ sử dụng

- **YOLOv5**: Mô hình phát hiện đối tượng state-of-the-art
- **DeepSORT**: Thuật toán tracking đối tượng
- **OpenCV**: Xử lý video và computer vision
- **PyTorch**: Deep learning framework
- **Python**: Ngôn ngữ lập trình chính

## Yêu cầu hệ thống

### Phần cứng tối thiểu
- **CPU**: Intel i5 hoặc AMD Ryzen 5 trở lên
- **RAM**: 8GB trở lên
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị GTX 1060 trở lên)
- **Ổ cứng**: 10GB trống

### Phần mềm
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS
- **Python**: 3.8 trở lên
- **CUDA**: 11.0+ (nếu sử dụng GPU)

## Cài đặt chi tiết

### Bước 1: Chuẩn bị môi trường

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv vehicle_counting_env
source vehicle_counting_env/bin/activate  # Linux/macOS
# hoặc
vehicle_counting_env\Scripts\activate     # Windows

# Cập nhật pip
pip install --upgrade pip
```

### Bước 2: Cài đặt PyTorch

```bash
# Cho GPU với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cho CPU only (nếu không có GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Bước 3: Cài đặt các thư viện cần thiết

```bash
# OpenCV cho xử lý video
pip install opencv-python

# DeepSORT cho tracking
pip install deep-sort-realtime

# Ultralytics cho YOLOv5
pip install ultralytics

# Các thư viện hỗ trợ
pip install numpy pandas matplotlib seaborn
```

### Bước 4: Cài đặt YOLOv5

```bash
# Clone YOLOv5 repository (nếu chưa có)
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Cấu trúc dự án

```
DACK_NMXLAS/
├── main.py                # File chính chạy hệ thống đếm xe
├── NKKN-VoThiSau.mp4      # Video đầu vào
├── yolo11n.pt             # Model YOLO (nếu có)
├── train.py               # Script training (nếu cần)
├── sample_data.ipynb      # Notebook phân tích dữ liệu mẫu
├── custom_data.yaml       # File cấu hình dữ liệu (nếu có)
├── runs/                  # Kết quả training, inference (weights, hình ảnh, thống kê...)
│   └── detect/
│       └── train/
│           ├── ...        # Các file kết quả, hình ảnh, thống kê
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── train_data/            # Dữ liệu training/validation/test
│   ├── images/            # Ảnh gốc
│   │   ├── train/         # Ảnh training
│   │   ├── val/           # Ảnh validation
│   │   └── test/          # Ảnh test
│   └── labels/            # Nhãn tương ứng
│       ├── train/         # Nhãn training
│       ├── val/           # Nhãn validation
│       ├── test/          # Nhãn test
│       ├── *.cache*       # File cache tăng tốc
├── yolov5su.pt
```

## Dữ liệu training

### Thống kê dataset
- **Tổng số ảnh**: 144 ảnh
- **Training set**: 105 ảnh (70%)
- **Validation set**: 45 ảnh (30%)
- **Classes**: 4 loại phương tiện (bus, car, motorbike, truck)

### Nguồn dữ liệu
- Video gốc: `NKKN-VoThiSau.mp4`
- Được extract thành các frame và annotate thủ công
- Sử dụng Roboflow để tạo dataset

## Cách sử dụng

### Bước 1: Chuẩn bị dữ liệu

1. **Thay thế video đầu vào**:
   - Đặt video cần xử lý vào thư mục gốc với tên `NKKN-VoThiSau.mp4`
   - Hoặc thay đổi đường dẫn trong `main.py`:

```python
video_path = "path/to/your/video.mp4"
```

2. **Kiểm tra model**:
   - Đảm bảo file `best.pt` có trong đường dẫn: `yolov5/runs/train/exp/weights/best.pt`
   - Model đã được train với 3 classes: bus, car, motorbike

### Bước 2: Chạy hệ thống

```bash
python main.py
```

### Bước 3: Điều chỉnh tham số (tùy chọn)

Trong file `main.py`, bạn có thể điều chỉnh các tham số sau:

```python
# Đường dẫn đến model và video
model_path = r"D:\Hoc_Tap\HK243\NM_XLAS\DACK_NMXLAS\yolov5\runs\train\exp\weights\best.pt"
video_path = r"D:\Hoc_Tap\HK243\NM_XLAS\DACK_NMXLAS\NKKN-VoThiSau.mp4"
output_path = "output.mp4"

# Danh sách class names (không thay đổi nếu dùng model đã train)
class_names = ['bus', 'car', 'motorbike']

# Vạch đếm - điều chỉnh tọa độ cho phù hợp với video
count_line = [[1, 237], [797, 239]]  # [x1, y1], [x2, y2]

# Ngưỡng confidence (0.0 - 1.0)
conf_threshold = 0.5

# Tham số DeepSORT
max_age = 30  # Số frame tối đa để track một đối tượng
```

## Giải thích thuật toán

### 1. Phát hiện đối tượng (YOLOv5)
```python
# Tải model YOLOv5 đã train
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Phát hiện đối tượng trong frame
results = model(frame)
```

### 2. Tracking đối tượng (DeepSORT)
```python
# Khởi tạo tracker
tracker = DeepSort(max_age=30)

# Cập nhật tracks
tracks = tracker.update_tracks(detections, frame=frame)
```

### 3. Đếm đối tượng
```python
# Kiểm tra giao nhau giữa đường đi và vạch đếm
def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# Đếm khi đối tượng cắt qua vạch
if intersect(prev_center, center, count_line[0], count_line[1]):
    sl_xe[label] += 1
```

## Training model (nếu cần retrain)

### Bước 1: Chuẩn bị dữ liệu

**Tạo file config** `data.yaml`:
```yaml
train: train_data/images/train
val: train_data/images/val
nc: 4
names: ['bus', 'car', 'motorbike','truck']
```

### Bước 2: Training

```bash
cd yolov5

# Training với model pretrained
python train.py --data ../data.yaml --weights yolov5s.pt --epochs 100 --batch-size 16 --img 640

# Training từ scratch (nếu cần)
python train.py --data ../data.yaml --weights '' --epochs 100 --batch-size 16 --img 640
```

### Bước 3: Đánh giá model

```bash
# Validation
python val.py --data ../data.yaml --weights runs/train/exp/weights/best.pt

# Inference test
python detect.py --source ../NKKN-VoThiSau.mp4 --weights runs/train/exp/weights/best.pt
```

## Kết quả và Output

### Video output
- **File**: `output.mp4`
- **Nội dung**: 
  - Bounding box cho từng đối tượng
  - ID tracking
  - Vạch đếm hồng
  - Counter real-time ở góc phải
#### 1. Model không load được
```bash
# Kiểm tra đường dẫn
ls yolov5/runs/train/exp/weights/best.pt

# Kiểm tra quyền truy cập file
chmod 644 yolov5/runs/train/exp/weights/best.pt
```

#### 2. Video không đọc được
```bash
# Kiểm tra codec
ffmpeg -i NKKN-VoThiSau.mp4

# Convert video nếu cần
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

#### 3. DeepSORT import error
```bash
# Cài đặt lại deep-sort-realtime
pip uninstall deep-sort-realtime
pip install deep-sort-realtime
```

### Tối ưu hiệu suất

#### 1. Tối ưu GPU
```python
# Sử dụng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

#### 2. Frame skipping (cho video dài)
```python
# Xử lý 1 frame mỗi N frame
frame_skip = 2
if frame_count % frame_skip != 0:
    continue
```

#### 3. Tỉ lệ phân giải
```python
# Giảm độ phân giải để tăng tốc
scale_factor = 0.5
frame = cv2.resize(frame, (int(w*scale_factor), int(h*scale_factor)))
```

## Đánh giá hiệu suất
### Benchmark
- **Hardware**: NVIDIA RTX 4050
- **Video**: 1920x1080, 30 FPS
- **Performance**: ~20 FPS với độ chính xác >75%
- Đếm số lượng còn một chút sai sót vì mô hình không nhận diện được hoặc bị che mất 
## Mở rộng và cải tiến trong tương lai

### 1. Thêm classes mới
```python
# Thêm class mới vào class_names
class_names = ['bus', 'car', 'motorbike', 'truck', 'bicycle']

# Retrain model với dataset mới
```

### 2. Tạo ra vạch đếm cho từng làn đường
```python
# Đếm nhiều vạch
count_lines = [
    [[1, 237], [797, 239]],  # Vạch 1
    [[1, 400], [797, 402]],  # Vạch 2
]
```

### 3. Phát hiện hướng di chuyển của phương tiện
```python
# Phát hiện hướng di chuyển
def get_direction(prev_center, current_center):
    dx = current_center[0] - prev_center[0]
    dy = current_center[1] - prev_center[1]
    return "left" if dx < 0 else "right"
```

Dự án này được phát triển cho mục đích học tập và là đồ án cuối kỳ của môn Nhập Môn Xử Lý Ảnh Số.

## Tài liệu tham khảo

- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Huấn luyện YOLOv5 trên dữ liệu tùy chỉnh](https://docs.ultralytics.com/vi/yolov5/tutorials/train_custom_data/)
- [Yolov5AnimalCamera- KhuongDuy25](https://github.com/KhuongDuy25/Yolov5AnimalCamera/tree/main)


