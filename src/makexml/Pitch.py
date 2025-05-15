from music21 import note
import cv2
import pandas as pd
import numpy as np
from src.makexml.IntervalPreset import IntervalPreset

class Pitch:

    # 특정 음표 머리의 pitch를 계산하는 함수. 임시표까지 처리. 
    # 현재는 사전에 정의한 부분에 음표머리의 중심좌표가 존재하면 음이 있다고 판단하는데 정확도가 떨어짐
    # 향후 음표 머리의 좌표로부터 가장 가까운 줄/칸에 있는 음으로 찾는 로직으로 개선 예정 
    @staticmethod
    def find_pitch_from_y(staff_df, head, staff_lines, measiter, margin_ratio=0.35):
        ACCIDENTAL_CLASSES = {
            "accidental_sharp": 1,
            "accidental_flat": -1,
            "accidental_natural": 0
        }
        """
        y_center: 중심 y좌표 (note_head 기준)
        staff_lines: 오선 5줄의 y 좌표 (float 5개)
        interval_list: MIDI 번호 리스트 (길이 19)

        return: 해당 위치의 MIDI pitch (int), or None
        """
        interval_list = measiter.interval_list

        if len(staff_lines) != 5 or len(interval_list) != 19:
            return None

        # 줄 사이 간격 계산
        L1, L2, L3, L4, L5 = staff_lines
        gap = (L5 - L1) / 4 / 2

        x1, y1, x2, y2 = head["x1"], head["y1"], head["x2"], head["y2"]
        x_center = head["x_center"]
        y_center = head["y_center"]

        # 기준 생성
        margin = gap * margin_ratio
        positions = [
            (18, L1-5*gap),
            (17, L1-4*gap),
            (16, L1-3*gap),
            (15, L1-2*gap),
            (14, L1-1*gap),
            (13, L1),
            (12, L1+gap),
            (11, L2),
            (10, L2+gap),
            (9, L3),
            (8, L3+gap),
            (7, L4),
            (6, L4+gap),
            (5, L5),
            (4, L5+1*gap),
            (3, L5+2*gap),
            (2, L5+3*gap),
            (1, L5+4*gap),
            (0, L5+5*gap),
        ]
        # 가장 가까운 위치 계산
        pitch_centers = [(pitch, position) for pitch, position in positions]
        pitch_idx, closest_y = min(pitch_centers, key=lambda p: abs(p[1] - y_center))

        n = note.Note()

        # 임시표 처리
        accidental_df = staff_df[staff_df["class_name"].isin(ACCIDENTAL_CLASSES.keys())].copy()
        for _, acc in accidental_df.iterrows():
            ax_target = acc["x2"] - acc["width"] * 0.2
            ay_center = acc["y_center"]

            # y조건: 임시표가 음표 머리 영역 y 안에 있어야 함
            if not (y1 <= ay_center <= y2):
                continue

            # x조건: 임시표가 음표 머리 옆에 있을 경우
            head_width = x2 - x1
            if x1 <= ax_target <= x2:
                adjust = ACCIDENTAL_CLASSES[acc["class_name"]]
                if adjust == 1:
                    interval_list[pitch_idx] += 1
                    n.pitch.midi = interval_list[pitch_idx]
                    n.accidental = note.Accidental('sharp')
                elif adjust == -1:
                    interval_list[pitch_idx] -= 1
                    n.pitch.midi = interval_list[pitch_idx]
                    n.accidental = note.Accidental('flat')
                else:
                    temp_interval = IntervalPreset.get_interval_list(measiter.cur_clef, 0)
                    interval_list[pitch_idx] = temp_interval[pitch_idx]
                    n.pitch.midi = interval_list[pitch_idx]
                    n.accidental = note.Accidental('natural')
                return n

        n.pitch.midi = interval_list[pitch_idx]
        n.accidental = None
        return n


    # 음표 영역 안에 dot_note_head의 중심좌표가 있는지 확인하는 함수
    @staticmethod 
    def is_dotted_note(note_box, staff_df):
        """
        note_box: (x1, y1, x2, y2) → 음표 바운딩 박스
        dot_heads: [(x1, y1, x2, y2), ...] → dot_note_head 박스 리스트

        return: True(부점음표) / False(일반음표)
        """
        nx1, ny1, nx2, ny2 = note_box

        dot_df = staff_df[staff_df["class_name"] == "dot_note_head"]
        dot_centers = dot_df[["x_center", "y_center"]].values.tolist()

        for cx, cy in dot_centers:
            if nx1 <= cx <= nx2 and ny1 <= cy <= ny2:
                return True
        return False
    
    # 해당 음표의 머리로 추정되는 음표머리의 y좌표들 반환
    def find_note_head(head_fd, x1, y1, x2, y2):
        #print(x1, y1, x2, y2)
        #print(head_fd)
        hits = head_fd[
            (head_fd["x_center"] >= x1) & (head_fd["x_center"] <= x2) &
            (head_fd["y_center"] >= y1) & (head_fd["y_center"] <= y2)
        ].copy()
        #print(hits)
        if hits.empty:
            return pd.DataFrame(columns=head_fd.columns)
        x_base = hits["x_center"].min()
        base_width = hits[hits["x_center"] == x_base]["width"].iloc[0]

        # 3. 오른쪽에 붙어있는 다른 머리 찾기 (기둥 오른쪽 가능성)
        extra = head_fd[
            (head_fd["x_center"] > x_base + 2) &
            (head_fd["x_center"] <= x_base + base_width) &
            (head_fd["y_center"] >= y1) & (head_fd["y_center"] <= y2)
        ]
        return pd.concat([hits, extra]).drop_duplicates().reset_index(drop=True)