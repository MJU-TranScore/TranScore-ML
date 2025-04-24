import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

# 1. YOLO 모델 로드
model = YOLO("/content/train/weights/best.pt")

# 2. 테스트 이미지 폴더
image_folder = "/content"
image_list = os.listdir(image_folder)

# 3. 오선 5줄 추출 함수
def extract_5lines_in_staff(staff_crop):
    gray = cv2.cvtColor(staff_crop, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    filtered = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    if lines is None:
        return []
    y_coords = [y1 for x1, y1, x2, y2 in lines[:, 0] if abs(y1 - y2) < 3]
    if len(y_coords) < 5:
        return []
    data = np.array(y_coords, dtype=np.float32).reshape(-1, 1)
    _, _, centers = cv2.kmeans(data, 5, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 10, cv2.KMEANS_PP_CENTERS)
    return sorted([float(c[0]) for c in centers])

# 4. pitch 계산 함수
def pitch_from_relative_y(y_relative, staff_lines, margin_ratio=0.15):
    L1, L2, L3, L4, L5 = staff_lines
    gap = (L5 - L1) / 4
    margin = gap * margin_ratio

    positions = [
        ("G5", -np.inf, L1 - gap),
        ("F5", L1 - gap, (L1 + L2) / 2 - margin),
        ("E5", (L1 + L2) / 2 - margin, (L1 + L2) / 2 + margin),
        ("D5", (L1 + L2) / 2 + margin, (L2 + L3) / 2 - margin),
        ("C5", (L2 + L3) / 2 - margin, (L2 + L3) / 2 + margin),
        ("B4", (L2 + L3) / 2 + margin, (L3 + L4) / 2 - margin),
        ("A4", (L3 + L4) / 2 - margin, (L3 + L4) / 2 + margin),
        ("G4", (L3 + L4) / 2 + margin, (L4 + L5) / 2 - margin),
        ("F4", (L4 + L5) / 2 - margin, (L4 + L5) / 2 + margin),
        ("E4", (L4 + L5) / 2 + margin, L5 + margin),
        ("D4", L5 + margin, L5 + gap),
        ("C4", L5 + gap, np.inf)
    ]
    for note, low, high in positions:
        if low <= y_relative < high:
            return note
    return "Unknown"

# 5. 메인 루프
for img_name in image_list:
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    print(f"
=== [Image] {img_name}")
    results = model.predict(img, conf=0.25)[0]
    vis = img.copy()

    staff_boxes, note_boxes, head_boxes = [], [], []

    for box in results.boxes:
        cls = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == "staff_line":
            staff_boxes.append((x1, y1, x2, y2))
        elif cls in ["quarter_note", "eighth_note", "sixteenth_note"]:
            note_boxes.append((cls, x1, y1, x2, y2))
        elif cls == "note_head":
            head_boxes.append((x1, y1, x2, y2))

    for i, (sx1, sy1, sx2, sy2) in enumerate(staff_boxes):
        crop = img[sy1:sy2, sx1:sx2]
        lines = extract_5lines_in_staff(crop)
        if len(lines) != 5:
            print(f"  [Staff {i}] 오선 검출 실패. Skip.")
            continue

        for idx, (cls_name, qx1, qy1, qx2, qy2) in enumerate(note_boxes):
            qcx, qcy = (qx1 + qx2) // 2, (qy1 + qy2) // 2
            if not (sx1 <= qcx <= sx2 and sy1 <= qcy <= sy2):
                continue

            note_pitches = []
            for (hx1, hy1, hx2, hy2) in head_boxes:
                hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                if qx1 - 5 <= hcx <= qx2 + 5 and qy1 - 5 <= hcy <= qy2 + 5:
                    rel_y = hcy - sy1
                    pitch = pitch_from_relative_y(rel_y, lines)
                    note_pitches.append(pitch)

            if not note_pitches:
                rel_y = qcy - sy1
                pitch = pitch_from_relative_y(rel_y, lines)
                note_pitches = [pitch]

            short_cls = {
                "quarter_note": "quarter",
                "eighth_note": "eighth",
                "sixteenth_note": "sixteenth"
            }[cls_name]

            color_map = {
                "quarter": ((0, 255, 0), (255, 0, 0)),
                "eighth": ((0, 128, 255), (255, 105, 180)),
                "sixteenth": ((255, 0, 255), (128, 0, 255))
            }
            box_color, text_color = color_map[short_cls]

            cv2.rectangle(vis, (qx1, qy1), (qx2, qy2), box_color, 2)
            label = f"{short_cls}: {', '.join(note_pitches)}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (qx1, qy1 - th - 8), (qx1 + tw, qy1), (255, 255, 255), -1)
            cv2.putText(vis, label, (qx1, qy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    plt.figure(figsize=(12, 6))
    plt.title(f"Final Mapping - {img_name}")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
#gpt 수정버전