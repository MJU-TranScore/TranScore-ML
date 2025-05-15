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