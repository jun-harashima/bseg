from bseg.morphology import Morphology


class Bseg:

    def __init__(self):
        pass

    def segment(self, analysis_result):
        self._transform(analysis_result)
        return []

    def _transform(self, analysis_result):
        morps = []
        for line in analysis_result.split("\n"):
            morp = Morphology(line)
            morps.append(morp)
        return morps
