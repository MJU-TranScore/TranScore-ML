import cv2
import numpy as np
class StafflineUtils:
    # 음정 계산을 위한 오선의 좌표를 찾아서 리스트로 반환하는 함수
    @staticmethod 
    def extract_5lines(staff_crop):
        gray = cv2.cvtColor(staff_crop, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        _, threshed = cv2.threshold(lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(threshed, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        if lines is None: return []
        y_coords = [y1 for x1, y1, x2, y2 in lines[:, 0] if abs(y1 - y2) < 3]
        if len(y_coords) < 5: return []
        data = np.array(y_coords, dtype=np.float32).reshape(-1, 1)
        _, _, centers = cv2.kmeans(data, 5, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 10, cv2.KMEANS_PP_CENTERS)
        return sorted([float(c[0]) for c in centers])
    
    # 해당 보표에서 탐지할 영역 선택
    @staticmethod  
    def calculate_pitch_range_area(staff_lines):
        y_top = staff_lines[0]
        y_bottom = staff_lines[4]
        gap = (y_bottom - y_top) / 4  # 줄 간 간격

        # G3는 아래로 줄 3개 (칸 3칸) 더 내려감 → 총 3 * (gap/2)
        # D6는 위로 줄 2개 (칸 2칸) 더 올라감 → 총 2 * (gap/2)
        pitch_top_y = y_top - (gap / 2) * 5     # D6
        pitch_bottom_y = y_bottom + (gap / 2) * 5  # G3

        return int(pitch_top_y), int(pitch_bottom_y)
    
    # 해당 staff_line에서 처리할 객체들만 추리고 정렬하는 함수
    @staticmethod 
    def get_objects_in_staff_area(staff_df, staff_x1, staff_x2, pitch_y_top, pitch_y_bottom):
        """
        주어진 staff_df에서 지정된 보표 영역에 포함되는 객체를 복사하고,
        왼쪽에서 오른쪽 (x_center 오름차순),
        같은 x라면 위에서 아래 (y_center 오름차순)으로 정렬한다.
        """
        # 조건에 맞는 행만 추출
        filtered_df = staff_df[
            (staff_df["x2"] >= staff_x1) &
            (staff_df["x1"] <= staff_x2) &
            (staff_df["y2"] >= pitch_y_top) &
            (staff_df["y1"] <= pitch_y_bottom)
        ].copy()

        # 정렬: x_center → y_center
        filtered_df.sort_values(by=["x_center", "y_center"], inplace=True)
        filtered_df.reset_index(drop=True, inplace=True)

        return filtered_df

    #fallback 로직 (음자리표로 staff_line 추정)
    @staticmethod
    def fallback_staffline_from_clef(clef_row, image):
        h, w = image.shape[:2]

        clef_y1 = int(clef_row["y1"])
        clef_y2 = int(clef_row["y2"])
        clef_height = clef_y2 - clef_y1

        # Clef type에 따른 확장 비율 설정
        if clef_row["class_name"] == "clef_F":  # 낮은 음자리표 (Bass Clef)
            up_ratio = 0.05    # y1 기준 위쪽으로 5%
            down_ratio = 0.30  # y2 기준 아래쪽으로 30%
        elif clef_row["class_name"] == "clef_G":  # 높은 음자리표 (Treble Clef)
            up_ratio = 0.05    # y1 기준 위쪽으로 15%
            down_ratio = 0.05  # y2 기준 아래쪽으로 15%
        else:
            # 기본값 (예비 처리용)
            up_ratio = 0.10
            down_ratio = 0.10

        # 확장된 crop 영역 계산
        y1_pad = max(0, int(clef_y1 - clef_height * up_ratio))
        y2_pad = min(h, int(clef_y2 + clef_height * down_ratio))

        # 이미지 crop
        staff_crop = image[y1_pad:y2_pad, 0:w]

        # 오선 감지
        local_staff_lines = StafflineUtils.extract_5lines(staff_crop)

        # 성공 시, 원본 이미지 좌표로 보정해서 반환
        if len(local_staff_lines) == 5:
         staff_lines_global = [y + y1_pad for y in local_staff_lines]
         print(f"[🟡 fallback] Clef 기반 오선 Y좌표: {staff_lines_global}")
         return staff_lines_global
        else:
         print("[❌ fallback] Clef 기반 오선 감지 실패")
         return []
