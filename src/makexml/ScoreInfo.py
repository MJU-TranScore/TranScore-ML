class ScoreInfo:
    def __init__(self):
        self.keysig_list = []
        self.timesig_list = []

    def add_keysig(self, keysig):
        self.keysig_list.append(keysig)

    def add_timesig(self, timesig):
        self.timesig_list.append(timesig)
    
    def is_keysig_empty(self):
        if not self.keysig_list:
            return True 
        else:
            return False 
