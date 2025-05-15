class MeasureIterator:
    def __init__(self):
        self.cur_keysig = 0
        self.measure_length = -1
        self.remain_measure_length = -1
        self.interval_list = []
        self.cur_clef = -1