class ScoreIterator:
    def __init__(self):
        self.cur_keysig = 0
        self.cur_timesig = [0,0]
        self.clef = -1