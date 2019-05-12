import spacy
import util

class SpeakerLabeler:
    def __init__(self, doc, use_last=True):
        self.doc = doc
        self.quote_begin = [-1] * len(doc)
        self.quote_actor = dict()
        beginning_of_current_quote = -1
        for token in doc:
            if token.is_quote:
                if beginning_of_current_quote == -1:
                    self.quote_actor[token.i] = list()
                    beginning_of_current_quote = token.i
                else:
                    self.quote_begin[token.i] = beginning_of_current_quote
                    beginning_of_current_quote = -1
                    continue
            self.quote_begin[token.i] = beginning_of_current_quote
        last_quote_actor = list()
        for qb in self.quote_actor:
            i = qb 
            while i < len(doc) and self.quote_begin[i] == qb:
                # if this token is in a quote, it is a ccomp, and its head is
                # not in a quote, then we expect its head to be a verb to which
                # the actor(s) is the speaker.
                if doc[i].dep_ == "ccomp" and \
                        self.quote_begin[doc[i].head.i] == -1:
                    self.quote_actor[qb].extend(util.get_actors(doc[i].head))
                i += 1
            if use_last and len(self.quote_actor[qb]) == 0:
                self.quote_actor[qb] = last_quote_actor
            else:
                last_quote_actor = self.quote_actor[qb]
        self.quote_actor[-1] = list()
    def in_quote(self, w):
        if isinstance(w, spacy.tokens.token.Token):
            w = w.i
        return self.quote_begin[w] != -1
    def speaker(self, w):
        if isinstance(w, spacy.tokens.token.Token):
            w = w.i
        return self.quote_actor[self.quote_begin[w]]
