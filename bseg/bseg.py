from bseg.morphology import Morphology
from bseg.bunsetsu import Bunsetsu


class Bseg:

    def __init__(self):
        pass

    def segment(self, analysis_result):
        morps = self._generate_morphology_from(analysis_result)
        bnsts = self._generate_bunsetsu_from(morps)
        return []

    def _generate_morphology_from(self, analysis_result):
        morps = []
        for line in analysis_result.split("\n"):
            morp = Morphology(line)
            morps.append(morp)
        return morps

    def _generate_bunsetsu_from(self, morps):
        bnsts = []
        _morps = []
        for morp in morps:
            _morps.append(morp)
            if morp.part_of_speech == "助詞":
                bnst = Bunsetsu(_morps)
                bnsts.append(bnst)
                _morps = []
        bnst = Bunsetsu(_morps)
        bnsts.append(bnst)
        return bnsts
