import easyocr
import json
import numpy as np
from music21 import converter, note
from collections import defaultdict
from PIL import Image

class TextProcesser:
    # easyOCR reader 객체
    reader = easyocr.Reader(['ko','en'], gpu=False)
    # 특정 보표의 텍스트(가사&코드)만 담긴 DataFrame과 특정 음표의 x좌표들을 받으면 해당 음표에 해당하는 가사를 추출
    
    # 특정 음표에 해당하는 가사들을 추출 
    @staticmethod
    def find_text_list(text_df, x1, x2):
        # 전달된 가사 데이터프레임에서 x 범위 내에 있는 것만 추출
        text_list = text_df[
            (text_df["x_center"] >= x1 ) &
            (text_df["x_center"] < x2)
        ].copy()

        # 추출 후 y좌표순으로 정렬 
        text_list = text_list.sort_values(by="y_center").reset_index(drop=True)
        return text_list

    # 전달받은 이미지의 텍스트를 추출 
    @staticmethod
    def detect_text(img):        
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        results = TextProcesser.reader.readtext(img, detail=0)

        if results :
            return "".join(results).strip()
        return ""
    
    # 전달받은 musicxml에서 가사를 추출하여 json으로 변환하기
    # 현재는 mxl을 score로 변환하고 추출하는 방식인데 추후 파라미터를 score로 받을 수도 있음 
    @staticmethod
    def get_lyrics_json_from_mxl(mxl_path): 
        score = converter.parse(mxl_path)
        notes = list(score.recurse().notes)

        # 절 별로 가사 수집
        lyrics_by_verse = defaultdict(list)
        
        # 절 번호 저장용 
        all_verse_numbers = set()
        # 먼저 모든 절 번호 수집
        for n in notes:
            for lyric in n.lyrics:
                number = int(lyric.number) if lyric.number else 1
                all_verse_numbers.add(number)

        max_verse = max(all_verse_numbers) if all_verse_numbers else 1
        verse_list = list(range(1, max_verse + 1))

        # 음표 순서대로 각 절의 가사 채우기
        for n in notes:
            # 현재 음표에서 실제 있는 절들만 수집
            verse_to_text = {int(lyric.number) if lyric.number else 1: lyric.text.strip() for lyric in n.lyrics}

            for verse in verse_list:
                lyrics_by_verse[verse].append(verse_to_text.get(verse, ""))  # 없는 절은 빈칸

        #print(lyrics_by_verse)
        return dict(lyrics_by_verse)


