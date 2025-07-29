import cv2
import torch
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# đường dẫn đến file đã train dữ liệu best.pt và video gốc
model_path = r"C:\Users\T.Hung\Desktop\DACK_NMXLAS\runs\detect\train\weights\best.pt"
video_path = r"C:\Users\T.Hung\Desktop\DACK_NMXLAS\NKKN-VoThiSau.mp4"
output_path = "output.mp4"

class_names = ['bus', 'car', 'motorbike']
count_line = [[1, 237], [797, 239]]  # Vạch đếm


model = YOLO(model_path)  # dùng model yolov5 của mình đã huấn luyện
tracker = DeepSort(max_age=30) # khởi tạo DeepSORT để theo dõi đối tượng

# Khởi tạo video capture
cap = cv2.VideoCapture(video_path)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

counted_ids = set() #biến chứa các ID đã đếm tránh đếm lại cùng 1 xe
sl_xe = defaultdict(int) # biến đếm xe
prev_centers = {}  #tọa độ tâm trước đó của từng khung hình dùng để kiểm tra xem có cắt vạch đếm hay chưa

# kiểm tra giao nhau giữa hai đoạn thẳng
def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# xử lý từng khung hình
while True:
    ret, frame = cap.read()
    if not ret:
        break # dừng lại nếu video lỗi hoặc là video hết

    results = model(frame)[0]  # lấy kết quả đầu tiên

    detections = []
    #tạo danh sách phát hiện đối tượng duyệt qua từng detections của mô hình
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        # lọc những đối tượng thuộc các lớp mong muốn (bus, car, motorbike)
        if cls_id < len(class_names) and conf >= 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1
            label = class_names[cls_id]
            detections.append(([x1, y1, w_box, h_box], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame) # cập nhật các đối tượng đang theo dõi dựa trên phát hiện mới
    
    for track in tracks: # lặp qua từng đối tượng dduocnwgj theo dõi
        if not track.is_confirmed():
            continue #bỏ qua track chưa được xác nhận
        #lấy ID, tọa độ bounding box, nhãn lớp của đối tượng tracl
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        label = track.get_det_class()
        
        if label in class_names:
            cx, cy = int((l + r) / 2), int((t + b) / 2)
            center = (cx, cy)
            #tính tâm của bounding box
            if track_id in prev_centers:
                prev_center = prev_centers[track_id]

                # nếu đã có tâm trước đó lấy ra để kiểm tra cắt vạch
                if intersect(prev_center, center, count_line[0], count_line[1]) and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    sl_xe[label] += 1
                    print(f'Đếm: {label} ID:{track_id}')

            # cập nhật center
            prev_centers[track_id] = center

            # vẽ bounding box
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ID:{track_id}', (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # vẽ line đếm
    cv2.line(frame, tuple(count_line[0]), tuple(count_line[1]), (255, 0, 255), 2)

    # hiển thị đếm
    y_offset = 30
    for cls in class_names:
        cv2.putText(frame, f'{cls}: {sl_xe[cls]}', (w - 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
    out.write(frame)
    cv2.imshow("Vehicle Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nKẾT QUẢ CUỐI:")
for cls in class_names:
    print(f"{cls}: {sl_xe[cls]}")
