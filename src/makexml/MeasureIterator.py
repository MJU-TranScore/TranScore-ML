from src.makexml.ScoreIterator import ScoreIterator
from fractions import Fraction
from src.makexml.IntervalPreset import IntervalPreset
class MeasureIterator:
    def __init__(self):
        self.__cur_keysig = 0 # 기본은 C로 가정 
        self.__measure_length = -1 # 기본은 없음
        self.__remain_measure_length = -1 # 기본은 없음
        self.__interval_list = []
        self.__cur_clef = -1 

    def set_cur_measinfo(self, keysig, timesig, interval_list, clef):
        self.__cur_keysig = keysig
        self.__measure_length = MeasureIterator.calc_measure_length(timesig)
        self.__remain_measure_length = self.measure_length
        self.__interval_list = interval_list
        self.__cur_clef = clef
    
    def set_cur_keysig(self, keysig):
        self.__cur_keysig = keysig
    
    def set_cur_measure_length(self, timesig):
        self.__measure_length = MeasureIterator.calc_measure_length(timesig)
        self.__remain_measure_length = self.__measure_length

    def set_cur_interval_list(self, interval_list):
        self.__interval_list = interval_list

    def set_cur_clef(self, clef):
        self.__cur_clef = clef

    def subtract_remain_measure_length(self, duration):
        if self.__remain_measure_length > 0:
            self.__remain_measure_length -= duration
        else:
            return -1 

        return self.__remain_measure_length

    def set_measiter_from_scoiter(self, scoiter):
        clef = scoiter.get_cur_clef()
        keysig = scoiter.get_cur_keysig()
        timesig = scoiter.get_cur_timesig()

        self.__cur_clef = clef
        self.__cur_keysig = keysig
        self.__measure_length = MeasureIterator.calc_measure_length(timesig)
        self.__remain_measure_length = self.__measure_length
        self.__interval_list = IntervalPreset.get_interval_list(clef, keysig)

    def get_cur_keysig(self):
        return self.__cur_keysig

    def get_cur_remain_measure_length(self):
        return self.__remain_measure_length

    def get_interval_list(self):
        return self.__interval_list
    
    def get_cur_clef(self):
        return self.__cur_clef

    def calc_interval_list(self):
        self.__interval_list = IntervalPreset.get_interval_list(self.__cur_clef, self.__cur_keysig)

    @staticmethod
    def calc_measure_length(timesig):
        num, note = timesig[0], timesig[1]

        length = Fraction(num) * Fraction(4, note)

        return length
    
    