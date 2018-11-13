class Morphology:

    def __init__(self, line):
        surface, rest = line.split("\t")
        features = rest.split(",")
        self.surface = surface
        self.part_of_speech = features[0]
        self.part_of_speech1 = features[1]
        self.part_of_speech2 = features[2]
        self.part_of_speech3 = features[3]
        self.conjugation_type = features[4]
        self.conjugation_form = features[5]
        self.base_form = features[6]
        self.reading = features[7]
        self.pronunciation = features[8]
