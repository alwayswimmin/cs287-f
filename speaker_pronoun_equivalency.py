from knowledge_graph import TokenEquivalency
from speaker_labeler import SpeakerLabeler
from util import is_first_person_pronoun

class SpeakerPronounEquivalency(TokenEquivalency):
    def __init__(self, verbose=False):
        super(SpeakerPronounEquivalency, self).__init__(verbose=verbose)
        self.sl = dict()
    def register(self, doc):
        self.sl[doc] = SpeakerLabeler(doc)
    def __call__(self, token):
        if is_first_person_pronoun(token):
            return self.sl[token.doc].speaker(token)
        return list()
