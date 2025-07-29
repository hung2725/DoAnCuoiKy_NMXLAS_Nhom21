import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === CẤU HÌNH ===
model_path = r"C:\Users\T.Hung\Desktop\DACK_NMXLAS\runs\detect\train\weights\best.pt"
video_path = r"C:\Users\T.Hung\Desktop\DACK_NMXLAS\NKKN-VoThiSau.mp4"
output_path = "output.mp4"
class_names = ['bus', 'car', 'motorbike']
count_line = [[1, 237], [797, 239]]  # Vạch đếm

# === TẢI MÔ HÌNH YOLO ===
model = YOLO(model_path)
tracker = DeepSort(max_age=30)

# === VIDEO ===
cap = cv2.VideoCapture(video_path)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# === BIẾN ĐẾM ===
counted_ids = set()
sl_xe = defaultdict(int)
prev_centers = {}

# === GIAO NHAU ĐOẠN THẲNG ===
def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# === XỬ LÝ VIDEO ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # [0] để lấy kết quả dự đoán đầu tiên

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id < len(class_names) and conf >= 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1
            label = class_names[cls_id]
            detections.append(([x1, y1, w_box, h_box], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        label = track.get_det_class()

        if label in class_names:
            cx, cy = int((l + r) / 2), int((t + b) / 2)
            center = (cx, cy)

            if track_id in prev_centers:
                prev_center = prev_centers[track_id]

                if intersect(prev_center, center, count_line[0], count_line[1]) and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    sl_xe[label] += 1
                    print(f'Đếm: {label} ID:{track_id}')

            prev_centers[track_id] = center

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ID:{track_id}', (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Vẽ vạch đếm
    cv2.line(frame, tuple(count_line[0]), tuple(count_line[1]), (255, 0, 255), 2)

    # Hiển thị số lượng
    y_offset = 30
    for cls in class_names:
        cv2.putText(frame, f'{cls}: {sl_xe[cls]}', (w - 180, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    out.write(frame)
    cv2.imshow("Vehicle Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# === GIẢI PHÓNG ===
cap.release()
out.release()
cv2.destroyAllWindows()

# === KẾT QUẢ CUỐI ===
print("\nKẾT QUẢ CUỐI:")
for cls in class_names:
    print(f"{cls}: {sl_xe[cls]}")
