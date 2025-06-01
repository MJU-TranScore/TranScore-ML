class ScoreInfo:
    def __init__(self):
        self.__keysig_list = []
        self.__timesig_list = []

    def add_keysig(self, keysig):
        self.__keysig_list.append(keysig)

    def add_timesig(self, timesig):
        self.__timesig_list.append(timesig)
    
    def is_keysig_empty(self):
        if not self.__keysig_list:
            return True 
        else:
            return False

    def get_keysig_list(self):
        return self.__keysig_list 
