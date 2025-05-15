import pandas as pd
import json

# 현재 학습된 모델이 탐지능력이 떨어져 미리 만든 데이터를 통해 xml로 변환할 데이터를 생성 
class MakeTestData:
    CLASS_NAMES = [
        "accidental_flat", "accidental_natural", "accidental_sharp", "clef_F", "clef_G",
        "dot_eighth_rest", "dot_half_rest", "dot_note_head", "dot_quarter_rest", "eighth_note",
        "eightth_rest", "half_note", "half_rest", "harmony", "keysig_A", "keysig_Ab",
        "keysig_B", "keysig_Bb", "keysig_D", "keysig_Db", "keysig_E", "keysig_Eb", "keysig_F",
        "keysig_Fsharp", "keysig_G", "keysig_Gb", "lyrics", "measure", "measure_double",
        "note_head", "quarter_note", "quarter_rest", "sixteenth_note", "sixteenth_rest",
        "span_down", "span_up", "staff_line", "timesig_12_8", "timesig_2_2", "timesig_2_4",
        "timesig_3_4", "timesig_4_4", "timesig_6_8", "timesig_9_8", "whole_note", "whole_rest"
    ]

    # json 형식의 라벨링된 데이터를 DataFrame으로 바꾸는거
    @staticmethod
    def load_json_labels(json_path):
        #with open(json_path, 'r') as f:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        rows = []
        for box in data["boxes"]:
            class_name = box["label"]
            class_id = MakeTestData.CLASS_NAMES.index(class_name)
            x_center = float(box["x"])
            y_center = float(box["y"])
            width = float(box["width"])
            height = float(box["height"])
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            rows.append({
                "class_id": class_id,  # 정답이니까 숫자 ID는 없어도 됨
                "class_name": class_name,
                "confidence": 0.99,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x_center": x_center, "y_center": y_center,
                "width": width, "height": height
            })
        df_raw = pd.DataFrame(rows)
        df_raw_sorted = df_raw.sort_values(by=["class_name", "x_center", "y_center"])
        return df_raw_sorted
