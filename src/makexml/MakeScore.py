from fractions import Fraction
from music21 import chord,  stream, note, meter, key, clef, metadata, interval, bar
from src.makexml.ScoreInfo import ScoreInfo
from src.makexml.ScoreIterator import ScoreIterator
from src.makexml.MeasureIterator import MeasureIterator
from src.makexml.Pitch import Pitch
from src.makexml.StafflineUtils import StafflineUtils
from src.makexml.IntervalPreset import IntervalPreset
from src.makexml.MakeTestData import MakeTestData
from src.makexml.TextProcesser import TextProcesser
from src.FilePath import BASE_DIR
import random
import string
from PIL import Image
import cv2
from ultralytics import YOLO
import pandas as pd 
import os
import numpy as np

class MakeScore:
    # 학습된 모델 위치와 그걸 기반으로 한 모델 객체 
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'weights.pt')
    MODEL = YOLO(MODEL_PATH)

    # YOLO 모델로 탐지할 음표 목록과 박자 
    NOTE_DURATION_MAP = {
        "whole_note": 4.0,
        "half_note": 2.0,
        "quarter_note": 1.0,
        "eighth_note": 0.5,
        "sixteenth_note": 0.25,
    }

    # YOLO 모델로 탐지할 쉼표 목록과 박자 
    REST_DURATION_MAP = {
        "whole_rest": 4.0,
        "dot_half_rest": 3.0,
        "half_rest": 2.0,
        "dot_quarter_rest": 1.5,
        "quarter_rest": 1.0,
        "dot_eighth_rest": 0.75,
        "eightth_rest": 0.5,
        "sixteenth_rest": 0.25,
    }

    """
    # 현재 구현하지 않은 함수
    # 사용자가 입력한 파일을 png로 변환시켜주는 함수
    # 입력 형식은 pdf, jpg, jpeg 등. 반환값은 png로
    @staticmethod
    def convert_origin_to_png(origin):
        png_list = []
        return png_list
    """

    #추가한 함수
    #staff_line이 겹쳐 탐지된 경우, y좌표 비슷한 줄끼리 병합하여
    #하나의 줄로 만든다. x1=0, x2=image_width로 강제 확장
    @staticmethod
    def merge_staff_lines(df: pd.DataFrame, image_width: int, y_threshold: int = 10) -> pd.DataFrame:
        staff_lines = df[df["class_name"] == "staff_line"].copy().reset_index(drop=True)
        others = df[df["class_name"] != "staff_line"].copy()

        merged = []
        used = [False] * len(staff_lines)

        for i in range(len(staff_lines)):
            if used[i]:
                continue

            y1_i = staff_lines.loc[i, "y1"]
            y2_i = staff_lines.loc[i, "y2"]
            conf_i = staff_lines.loc[i, "confidence"]
            class_id_i = staff_lines.loc[i, "class_id"]

            group = [staff_lines.loc[i]]
            used[i] = True

            for j in range(i + 1, len(staff_lines)):
                if used[j]:
                    continue
                y1_j = staff_lines.loc[j, "y1"]
                y2_j = staff_lines.loc[j, "y2"]

                if abs(y1_i - y1_j) < y_threshold and abs(y2_i - y2_j) < y_threshold:
                    group.append(staff_lines.loc[j])
                    used[j] = True

            y1_avg = float(np.mean([g["y1"] for g in group]))
            y2_avg = float(np.mean([g["y2"] for g in group]))
            x1 = 0
            x2 = image_width
            conf = max([g["confidence"] for g in group])

            merged.append({
                "class_id": class_id_i,
                "class_name": "staff_line",
                "confidence": conf,
                "x1": x1,
                "y1": y1_avg,
                "x2": x2,
                "y2": y2_avg,
                "x_center": (x1 + x2) / 2,
                "y_center": (y1_avg + y2_avg) / 2,
                "width": x2 - x1,
                "height": y2_avg - y1_avg
            })

        df_merged = pd.DataFrame(merged)
        result_df = pd.concat([others, df_merged], ignore_index=True)
        result_df = result_df.sort_values(by=["class_name", "x_center", "y_center"]).reset_index(drop=True)
        return result_df


    # 모델의 예측 결과를 pandas dataframe으로 변환시켜주는 함수 
    def convert_result_to_df(result):
        rows = []
        boxes = result.boxes 

        image_width = result.orig_img.shape[1]

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            class_name = result.names[class_id]  

            row = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(boxes.conf[i]),
                "x1": float(boxes.xyxy[i][0]),
                "y1": float(boxes.xyxy[i][1]),
                "x2": float(boxes.xyxy[i][2]),
                "y2": float(boxes.xyxy[i][3])
            }
            row["x_center"] = (row["x1"] + row["x2"]) / 2
            row["y_center"] = (row["y1"] + row["y2"]) / 2
            row["width"] = row["x2"] - row["x1"]
            row["height"] = row["y2"] - row["y1"]
            rows.append(row)

        df = pd.DataFrame(rows)
        # 중복되는 staff_line 제거 
        df = MakeScore.merge_staff_lines(df, image_width=image_width)
        # 클래스 이름으로 정렬하고 좌표기준으로 정렬
        df_sorted = df.sort_values(by=["class_name", "x_center", "y_center"]) 

        return df_sorted       


    # 이미지 리스트를 모델을 통해 탐지하는 함수 
    @staticmethod
    def detect_object(img_list):
        detection_results = []

        # 5. 메인 루프
        for img in img_list: # list에 들어있는 이미지마다 반복문으로 실행
            if img is None: # 이미지가 없으면
                continue # 이번 반복은 끝내고 다음으로

            #print(f"\n=== [Image] {img_name}") # 어떤 이미지를 처리중인지 로그 출력
            results = MakeScore.MODEL.predict(img, conf=0.25)[0] # 모델로 예측
            detection_results.append(results)

        return detection_results


    # 이미지 리스트를 받아 pandas dataframe으로 변환시켜주는 함수 
    @staticmethod
    def imgs_to_df(img_list):
        df_list = []
        vis_list = img_list.copy() # 원본 이미지 복사헤서 사용 
        detection_results = MakeScore.detect_object(vis_list)

        for result in detection_results:
            df_list.append(MakeScore.convert_result_to_df(result))

        return df_list

        
    # 이미지를 입력하면 학습된 모델을 통해 musicxml로 변환하는 함수
    # 현재는 이미지를 받는데 향후 다양한 형식을 지원할 예정  
    @staticmethod
    def make_score(origin):
        # 0. 악보 파일 형식 변환
        
        #img_list = MakeScore.convert_origin_to_png(origin)
        # 현재는 그냥 이미지로
        # 원본 이미지 사본
        vis_list = origin.copy()
        object_dfs = MakeScore.imgs_to_df(vis_list)
            
        
        # 이 경우 정상작동하므로 테스트를 위해 막아놓는다. 다시 테스트트
        # 테스트를 위해 모든 요소가 다 탐지되었다고 가정한 기존에 만든 깨끗한 데이터로 실행 
        """
        vis = cv2.imread("./testfiles/떳다떳다비행기.png")
        vis_list = []
        vis_list.append(vis)
        json_path = "./testfiles/떳다떳다비행기_annotation_data.json"
        print("MakeTestData type:", type(MakeTestData))
        object_df = MakeTestData.load_json_labels(json_path)
        object_dfs = []
        object_dfs.append(object_df)
        """

        # 변환
        score = MakeScore.convert_df_to_score(object_dfs, vis_list)

        return score

    # 각 이미지와 이미지에서 탐지된 객체들을 페이지별로 리스트 형태로 건내줌
    # 그러면 이걸 Score 객체로 변환시켜줌 
    @staticmethod # DataFrame
    def convert_df_to_score(object_dfs, vis_list):
        # 1. 악보 객체 생성
        score = stream.Score()
        scoinfo = ScoreInfo()
        scoiter = ScoreIterator()
        measiter = MeasureIterator()

        # 2. 파트(보표) 생성
        part = stream.Part() # 단일성부. 피아노 양손악보면 2번 하는 식으로 나중에 조정 

        # 3. 마디 생성
        measurenum = 1
        m = stream.Measure(number=measurenum)

        for idx, object_df in enumerate(object_dfs):
            vis = vis_list[idx]
            # 저장된 dataframe에서 보표에 대한 정보만 들고옴
            staff_df = object_df[object_df["class_name"] == "staff_line"].copy()
            staff_df = staff_df.sort_values(by="y1").reset_index(drop=True)

            # 🔁 staff_line이 감지되지 않았을 경우 → clef 기반 fallback 시도
            if staff_df.empty:
                clef_df = object_df[object_df["class_name"].isin(["clef_G", "clef_F"])]
                fallback_staff_rows = []
                for _, clef_row in clef_df.iterrows():
                    fallback_lines = StafflineUtils.fallback_staffline_from_clef(clef_row, vis)
                    if len(fallback_lines) == 5:
                        print(f"[⚠️ fallback 적용] Clef 기준으로 staff_line 대체 성공: {fallback_lines}")
                        fallback_staff_rows.append({
                            "x1": 0,
                            "x2": vis.shape[1],
                            "y1": min(fallback_lines),
                            "y2": max(fallback_lines),
                            "x_center": vis.shape[1] / 2,
                            "y_center": sum(fallback_lines) / 5,
                            "width": vis.shape[1],
                            "height": max(fallback_lines) - min(fallback_lines),
                            "class_name": "staff_line",
                            "class_id": -1,  # dummy
                            "confidence": 0.01  # 낮은 신뢰도로 표시
                        })
                if fallback_staff_rows:
                    staff_df = pd.DataFrame(fallback_staff_rows)


            # 해당 페이지의 탐지결과에서 가사 영역만 가진 dataframe과 코드 영역만 가진 dataframe
            lyrics_df = object_df[object_df["class_name"] == "lyrics"].copy()
            harmony_df = object_df[object_df["class_name"] == "harmony"].copy()

            # 들고온 보표의 개수만큼 반복문
            for staff_index in range(len(staff_df)):
                row = staff_df.iloc[staff_index]
                sx1, sy1, sx2, sy2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

                # 해당 보표의 가사만 골라내기
                if staff_index < len(staff_df) - 1: # 마지막 보표가 아닌 경우
                    next_row = staff_df.iloc[staff_index+1]
                    cur_lyrics_df = lyrics_df[
                        (lyrics_df["y_center"] > row["y2"]) &
                        (lyrics_df["y_center"] < next_row["y1"])
                        ].copy()
                else: # 마지막 보표인 경우
                    cur_lyrics_df = lyrics_df[
                        (lyrics_df["y_center"] > row["y2"])
                        ].copy()
                    
                # 박스쳐진 staff_line에 선이 5개가 안들어가있는 경우가 있어서 y좌표에 약간의 padding을 적용
                y_padding = int(row["height"] * 0.05)
                y1_pad = max(0, sy1 - y_padding)
                y2_pad = min(vis.shape[0], sy2 + y_padding)

                # 이미지에서 잘라냄
                staff_crop = vis[y1_pad:y2_pad, 0:vis.shape[1]]

                # OpenCV로 5줄 찾음
                staff_lines = StafflineUtils.extract_5lines(staff_crop)
                pitch_y_top = None
                pitch_y_bottom = None
                if len(staff_lines) == 5:
                    staff_lines_global = [y + sy1 for y in staff_lines]  # 원본 좌표로 보정
                    print(f"[✅ 보표 {staff_index + 1}] 오선 Y좌표:", staff_lines_global)

                    staff_lines_global = [int(y + sy1) for y in staff_lines]

                    pitch_y_top, pitch_y_bottom = StafflineUtils.calculate_pitch_range_area(staff_lines_global) # pitch 범위
                else:
                    print(f"[⚠️ 보표 {staff_index + 1}] 오선 인식 실패")

                # 객체 탐색
                # 탐지범위 내에 있는 객체들만 따로 추출. x, y 순서대로 정렬되어 있으므로 이걸 순차적으로 탐색
                cur_staff_df = StafflineUtils.get_objects_in_staff_area(object_df, sx1, sx2, pitch_y_top, pitch_y_bottom)
                cur_staff_note_head = cur_staff_df[cur_staff_df["class_name"] == "note_head"]

                # 탐색
                for idx, row in cur_staff_df.iterrows():
                    cid = row["class_id"] # 현재 객체의 class_id
                    cls = row["class_name"] # 현재 객체의 class_name

                    if cid in [3,4]: # 음자리표
                        print("clef: ", cid)
                        if scoiter.clef != cid:
                            scoiter.clef = cid
                            measiter.cur_clef = cid
                            measiter.interval_list = IntervalPreset.get_interval_list(measiter.cur_clef, measiter.cur_keysig)

                            if cid == 4:
                                m.append(clef.TrebleClef())
                            else:
                                m.append(clef.BassClef())

                    elif "keysig" in cls: # 조표
                        keysig = cls.split("_")[1]
                        print("keysig_index: ", cls)
                        if IntervalPreset.KEY_ORDER[measiter.cur_keysig] != keysig:
                            keysig_index = IntervalPreset.KEY_ORDER.index(keysig)
                            if keysig_index > 6:
                                keysig_index = keysig_index - 13
                            scoinfo.keysig_list.append(keysig_index)
                            scoiter.cur_keysig = keysig_index
                            measiter.cur_keysig = keysig_index
                            measiter.interval_list = IntervalPreset.get_interval_list(measiter.cur_clef, measiter.cur_keysig)
                            print(measiter.interval_list)
                            for el in m.getElementsByClass(key.KeySignature):
                                m.remove(el)
                            m.insert(0, key.KeySignature(keysig_index))
                            m.append(key.KeySignature(keysig_index))

                    elif "timesig" in cls: # 박자표
                        parts = cls.split("_")
                        if scoiter.cur_timesig[0] != parts[1] or scoiter.cur_timesig[1] != parts[2]:
                            scoiter.cur_timesig[0] = parts[1]
                            scoiter.cur_timesig[1] = parts[2]
                            measiter.measure_length = Fraction(int(parts[1])) * Fraction(4, int(parts[2]))
                            m.append(meter.TimeSignature(f'{parts[1]}/{parts[2]}'))

                    elif cls in MakeScore.REST_DURATION_MAP: # 쉼표
                        r = note.Rest()
                        r.duration.quarterLength = MakeScore.REST_DURATION_MAP[cls]
                        m.append(r)
                        #print(cls)

                    elif cls in MakeScore.NOTE_DURATION_MAP: # 음표
                        duration = MakeScore.NOTE_DURATION_MAP[cls]
                        c = chord.Chord()
                        # 점 음표 확인
                        note_box = (row["x1"], row["y1"], row["x2"], row["y2"])
                        #print(note_box)
                        if Pitch.is_dotted_note(note_box, cur_staff_df):
                            duration *= 1.5
                            print("dot",cls)
                        else:
                            print(cls)

                        # pitch 계산
                        head_df = Pitch.find_note_head(cur_staff_note_head, row["x1"], pitch_y_top, row["x2"], pitch_y_bottom)
                        print("음표탐지시도 완료")
                        if head_df.empty:
                            print("탐지된 음표 없음")
                            continue  # 또는 적절히 skip
                        pitches = []
                        for _, head in head_df.iterrows():
                            n = Pitch.find_pitch_from_y(cur_staff_df, head, staff_lines_global, measiter)
                            if isinstance(n, note.Note):
                                pitches.append(n)
                        if pitches:                            
                            c.pitches = [n.pitch for n in pitches]
                            c.duration.quarterLength = duration
                            for i, note_obj in enumerate(pitches):
                                if hasattr(note_obj, "accidental") and note_obj.accidental is not None and note_obj.accidental.displayStatus:
                                    c.notes[i].accidental = note_obj.accidental
                                    c.notes[i].accidental.displayStatus = True

                            # 가사 확인
                            lyrics_list = TextProcesser.find_text_list(cur_lyrics_df, row["x1"], row["x2"])

                            lyrics_data = []
                            for _, lyric in lyrics_list.iterrows():
                                x1, y1, x2, y2 = int(lyric["x1"]), int(lyric["y1"]), int(lyric["x2"]), int(lyric["y2"])
                                pad_x = lyric["width"] * 0
                                pad_y = lyric["height"] * 0 
                                y_max, x_max = vis.shape[:2]
                                crop_x1 = max(x1-pad_x,0)
                                crop_x2 = min(x2+pad_x,x_max)
                                crop_y1 = max(y1-pad_y,0)
                                crop_y2 = min(y2+pad_y,y_max)
                                lyrics_crop = vis[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
                                crop_pil = Image.fromarray(cv2.cvtColor(lyrics_crop, cv2.COLOR_BGR2RGB))
                                # OCR 수행
                                text = TextProcesser.detect_text(crop_pil)
                                # 결과 저장
                                lyrics_data.append(text)
                            for i, lyric in enumerate(lyrics_data):
                                print(f"탐지된 가사: {lyric}")
                                lyric_obj = note.Lyric()
                                lyric_obj.text = lyric
                                lyric_obj.number = i + 1
                                #c.notes[0].lyrics.append(lyric_obj)
                                c.addLyric(lyric)
                            m.append(c)
                            print(c)

                    elif cls in ["measure", "measure_double", "measure_final"]:
                        if cls == "measure_double":
                          m.rightBarline = bar.Barline("light-light")
                        elif cls == "measure_final":
                          m.rightBarline = bar.Barline("light-heavy") 

                        part.append(m)
                        measurenum += 1
                        m = stream.Measure(number=measurenum)
                        measiter.interval_list = IntervalPreset.get_interval_list(measiter.cur_clef, measiter.cur_keysig)
                    
                    """
                    elif cls in ["measure", "double_measure"]:
                        part.append(m)
                        measurenum += 1
                        m = stream.Measure(number=measurenum)
                        measiter.interval_list = IntervalPreset.get_interval_list(measiter.cur_clef, measiter.cur_keysig)
                    """


        part.append(m)
        measurenum += 1
        m = stream.Measure(number=measurenum)
        score.append(part)

        return score

    # 키를 변환하는 함수 
    # Score 객체와 변환할 값을 정수로 받아서 키를 변환
    # 범위는 -7 ~ +7까지지
    @staticmethod
    def change_key(score, diff): 
        if diff > 7 or diff < -7:
            return score
        
        if diff == 0:
            return score
        else:
            change = {
                -7: "-P5",
                -6: "-D5",
                -5: "-P4",
                -4: "-M3",
                -3: "-m3",
                -2: "-M2",
                -1: "-m2",
                1: "m2",
                2: "M2",
                3: "m3",
                4: "M3",
                5: "P4",
                6: "D5",
                7: "P5"
            }
            interval_str  = change[diff]
            intv = interval.Interval(interval_str)
            new_score = score.transpose(intv)
            print("키 변환 완료")
            return new_score
        
    # Score 객체를 받은 파일 이름으로 musicXML로 만들어주는 함수
    # 이름이 없으면 이름없는 악보 + 랜덤 문자열10개로 만들어줌    
    @staticmethod
    def score_to_xml(score, filename):
        name = filename
        if filename == None:
            chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
            ran_str = ''.join(random.choices(chars, k=10))
            name = "이름 없는 악보"+ ran_str

        score.metadata = metadata.Metadata()
        score.metadata.title = name

        score.write("musicxml", fp="./convert_result/"+name+'.xml')

        # 분명 임시표 이상하게 안넣는거 만들어놨는데 이상하게 나와서 이걸로 한번 테스트
        """temp = MakeScore.change_key(score,1)
        temp = MakeScore.change_key(temp,-1)
        temp.write("musicxml", fp="./convert_result/"+name+'.xml')
        """
