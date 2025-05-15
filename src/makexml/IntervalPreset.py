from music21 import pitch

class IntervalPreset:
    # 키 종류
    KEY_ORDER = ['C', 'G', 'D', 'A', 'E', 'B', 'Fs', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    # 각 키에 해당하는 숫자 
    KEY_PITCH_ORDER = [0, 5, 0, 7, 2, 9, 4, 0, 7, 2, 9, 4, 11]
    # 샾이 붙는 순서 
    SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    # 플랫이 붙는 순서
    FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F']

    # 기본 음자리표에 대한 음정리스트와 key를 입력하면 해당 음정리스트를 조정해서 해당 키에 맞는 음정리스트를 반환 
    @staticmethod 
    def apply_key_signature(pitchs, key):
        if key ==  0: # C키면 그냥 반환
            return pitchs
        result = pitchs.copy()

        if key > 0 and key < 7:
            for i in range(1, key + 1):
                k = IntervalPreset.KEY_PITCH_ORDER[i]
                #print(i, k)
                for idx in range(len(result)):
                    if result[idx] % 12 == k:
                        result[idx] += 1

        elif key < 0 and key > -7:
            for i in range(-1, key - 1, -1):
                k = IntervalPreset.KEY_PITCH_ORDER[i]
                #print(i, k)
                for idx in range(len(result)):
                    if result[idx] % 12 == k:
                        result[idx] -= 1

        return result
    
    # pitch와 key를 입력하면 맞은 음정리스트를 제공함  
    @staticmethod 
    def get_interval_list(clef, key):
        # clef는 음자리표. class_id를 고려하여 3이면 clef_F, 4면 clef_G
        # key는 -6 ~ 6까지의 정수
        clef_pitches = []

        if(clef == 3): # 낮은음자리표
            clef_pitches = [35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65]
        elif(clef == 4): # 높은음자리표
            clef_pitches = [55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86]
        else:
            print("error")

        return IntervalPreset.apply_key_signature(clef_pitches, key)
    
    # 특정 key의 midi로 된 음정리스트를 가지고 C4 형식으로 변환해주는 함수 
    @staticmethod
    def convert_midi_list_to_note_names(key):
        """
        각 clef_key 조합의 MIDI 리스트를 note name (ex: 'C4') 리스트로 변환
        """
        converted = {}
        for i in range(0,18):
            converted[i] = [pitch.Pitch(key[i]).nameWithOctave]
        return converted