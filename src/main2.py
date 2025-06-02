import subprocess
import os 
import cv2
from src.FilePath import BASE_DIR
from src.makexml.MakeScore import MakeScore
from src.exception import *
# from src.makexml.TextProcesser import TextProcesser
# 로직엔 문제가 없는 것 같지만 객체탐지 성능이 떨어져 제대로된 악보가 나오지 않는다

def main():
    print("메인 실행")

    
    # 악보 만들기
    img_path = os.path.join(BASE_DIR, 'testfiles', '달리기.pdf')  # 테스트용 이미지 경로
    try:
        img_list = ImageConverter.file_path_to_imglist(img_path)
    except NotImageError as e:
        print("파일 형식이 알맞지 않습니다")

    try:
        score, scoinfo = MakeScore.make_score(img_list)  # 실제 악보를 이용해 시도
        print(scoinfo.get_keysig_list())
    except EmptyDataFrameError as e:
        print("예외 발생", e)
        return
    except EmptyImageError as e:
        print("예외 발생", e)
        return
    except Exception as e:
        print("예상치 못한 예외 발생", e)
        return

        


    # 악보이름
    # score = MakeScore.make_score(None) # 현재는 미리 만들어진 데이터셋을 이용해 변환
    name = "곰세마리_20250531-1"
    
    # score 객체를 musicxml로 변환환
    MakeScore.score_to_xml(score, name)
    """
    # 기존 musicxml 파일을 가지고 가사를 추출하여 json 형식의 문자열로 변환 
    mxl_path = os.path.join(BASE_DIR, 'convert_result', '곰세마리_20250529-1.xml')
    lyrics_json = TextProcesser.get_lyrics_json_from_mxl(mxl_path)
    print(lyrics_json) # json 출력 확인 
    """

    # 만들어진 xml을 pdf로 변환환
    input_path = os.path.join(BASE_DIR, 'convert_result', name+".xml")
    # 출력 PDF 경로
    output_path = os.path.join(BASE_DIR, 'convert_result', name+".pdf")
    # MuseScore 실행 파일 경로
    #mscore_path = "../squashfs-root/bin/mscore4portable" # 이게 변환하는거. squashfs-root 이게 리눅스용 musescore4를 cli에서 실행할 수 있도록 변환?시킨거
    mscore_path = "../../Program Files/MuseScore 4/bin/MuseScore4.exe"
    # 출력 mp3 경로
    #output_path_mp3 = os.path.join(BASE_DIR, 'convert_result', name+".mp3")

    #성공적으로 됨. 현재는 다른것을테스트하기 위해 잠시 막아놓기



    # 키 변환
    diff = 5 # 변환시킬 만큼
    new_score_name = name+"키변환"+str(diff)
    new_score = MakeScore.change_key(score, diff) # 두번째 파라미터가 변경할 key. 현재 -2 ~ 2까지만만
    MakeScore.score_to_xml(new_score,new_score_name)
    
    
    input_path2 = os.path.join(BASE_DIR, 'convert_result', new_score_name+".xml")
    output_path2 = os.path.join(BASE_DIR, 'convert_result', new_score_name+".pdf")
    #mscore_path2 = "../squashfs-root/bin/mscore4portable" 

    result2 = subprocess.run(
        [mscore_path, input_path2, "-o", output_path2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    

    #정상적으로 잘 되었는지 확인
    result = subprocess.run(
        [mscore_path, input_path, "-o", output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print("PDF 변환 완료:", output_path)
    else:
        print("오류 발생:")
        print(result.stderr.decode())




if __name__ == "__main__":
    main()
