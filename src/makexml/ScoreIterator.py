class ScoreIterator:
    def __init__(self):
        self.__cur_keysig = 0 # 기본 C키 가정
        self.__cur_timesig = [0,0] # 기본은 없음 
        self.__clef = -1 # 기본은 -1로 없앰앰
    
    # 멤버변수값 초기화 
    def set_cur_score_info(self, keysig, timesig, clef):
        self.__cur_keysig = keysig
        self.__cur_timesig = timesig
        self.__clef = clef

    # 현재 키 설정
    def set_cur_keysig(self, keysig):
        self.__cur_keysig = keysig

    # 현재 박자표 설정
    def set_cur_timesig(self, timesig):
        self.__cur_timesig = timesig
    
    # 현재 음자리표 설정
    def set_cur_clef(self, clef):
        self.__clef = clef

    # 음자리표 반환
    def get_cur_clef(self):
        return self.__clef
    
    # 박자표 반환
    def get_cur_timesig(self):
        return self.__cur_timesig
    
    # 조표 반환 
    def get_cur_keysig(self):
        return self.__cur_keysig
    
    # 박자표 비교
    def compare_timesig(self, timesig):
        if self.__cur_timesig == timesig:
            return True
        else:
            return False 
