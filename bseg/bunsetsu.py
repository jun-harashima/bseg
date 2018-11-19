class Bunsetsu:

    def __init__(self, morps):
        self.morphologies = morps
        self.surface = "".join([morp.surface for morp in morps])

    def ispredicate(self):
        return any([morp.part_of_speech in ["動詞", "形容詞", "判定詞"]
                    for morp in self.morphologies])
