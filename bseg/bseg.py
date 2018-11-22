from bseg.morphology import Morphology
from bseg.bunsetsu import Bunsetsu


class Bseg:

    def __init__(self):
        pass

    def segment(self, analysis_result):
        morps = self._construct_morphology_from(analysis_result)
        bnsts = self._construct_bunsetsu_from(morps)
        return bnsts

    def _construct_morphology_from(self, analysis_result):
        return [Morphology(line) for line in analysis_result.split("\n")]

    def _construct_bunsetsu_from(self, morps):
        bnsts = []
        _morps = []
        for morp in morps:
            _morps.append(morp)
            if not morp.isfunction():
                continue
            bnst = Bunsetsu(_morps)
            bnsts.append(bnst)
            _morps = []
        bnst = Bunsetsu(_morps)
        bnsts.append(bnst)
        return bnsts
