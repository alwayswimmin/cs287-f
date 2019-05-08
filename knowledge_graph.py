import spacy
import neuralcoref
import numpy as np
from termcolor import colored

import warnings
warnings.filterwarnings("ignore")

class KnowledgeGraph:

    def __init__(self, nlp, verbose=False):
        self.nlp = nlp
        self.verbose = verbose
        self.relations = list()
        self.noun_threshold = 0.8
        self.verb_threshold = 0.9
        self.entailment = 0
        self.dissimilar_verbs = 1
        self.missing_dependencies = 2
        self.contradiction = 3

    def get_actors(self, verb):
        actors = []
        for child in verb.children:
            if child.dep_ == "nsubj":
                actors.append(child)
            elif child.dep_ == "agent":
                # passive, look for true actor
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        actors.append(grandchild)
        if verb.dep_ == "acl":
            if verb.text[-3:] == "ing":
                actors.append(verb.head)
        return actors

    def get_acteds(self, verb):
        acteds = []
        for child in verb.children:
            if child.dep_ == "dobj" or child.dep_ == "nsubjpass":
                acteds.append(child)
        if verb.dep_ == "acl":
            if verb.text[-3:] != "ing":
                acteds.append(verb.head)
        return acteds

    def get_relation(self, verb):
        actors = self.get_actors(verb)
        acteds = self.get_acteds(verb)
        return verb, actors, acteds

    def add_verb(self, verb):
        self.relations.append(self.get_relation(verb))

    def is_generic(self, token):
        return token.pos_ == "PRON" or token.pos_ == "DET"

    def get_valid_cluster_tokens(self, noun, use_generic=False):
        tokens = list()
        if (noun.pos_ == 'PRON' or noun.pos_ == 'DET') and noun.head.dep_ == 'relcl':
            # the head is the verb of the relative clause
            # the head of the verb should be the noun this thing refers to
            if self.verbose:
                print("found relative clause, replacing", noun, "with", noun.head.head)
            noun = noun.head.head
        for cluster in noun._.coref_clusters:
            for span in cluster:
                for token in span:
                    if use_generic or not self.is_generic(token):
                        if self.verbose and self.is_generic(token):
                            print(colored("warning:", "yellow"), "using generic token", noun)
                        tokens.append(token)
        if len(tokens) == 0:
            if use_generic or not self.is_generic(noun):
                if self.verbose and self.is_generic(noun):
                    print(colored("warning:", "yellow"), "using generic token", noun)
                tokens.append(noun)
        return tokens 

    def noun_similarity(self, n1, n2):
        tokens1 = self.get_valid_cluster_tokens(n1)
        tokens2 = self.get_valid_cluster_tokens(n2)
        if len(tokens1) == 0 or len(tokens2) == 0:
            tokens1 = self.get_valid_cluster_tokens(n1, True)
            tokens2 = self.get_valid_cluster_tokens(n2, True)
        maximum_similarity = 0
        maximum_pair = None
        for token1 in tokens1:
            for token2 in tokens2:
                token_similarity = token1.similarity(token2)
                if token_similarity > maximum_similarity:
                    maximum_similarity = token_similarity
                    maximum_pair = token1, token2
        if maximum_similarity > self.noun_threshold:
            return maximum_similarity, ("best match:", maximum_similarity, maximum_pair)
        return maximum_similarity, ("best match:", maximum_similarity, maximum_pair)

    def noun_intersect_setminus(self, supset, subset):
        contained_nouns = []
        missing_nouns = []
        for n in subset:
            contained = False
            for n2 in supset:
                r = self.noun_similarity(n, n2)
                if self.verbose:
                    print(n, n2, r)
                if r[0] > self.noun_threshold:
                    contained = True
                    contained_nouns.append((n, n2, r[1]))
                    continue
            if not contained:
                missing_nouns.append(n)
        return contained_nouns, missing_nouns

    def verb_similarity(self, v1, v2):
        verb_similarity = v1.similarity(v2)
        if v1.lemma_ == v2.lemma_:
            verb_similarity = max(0.95, verb_similarity)
        return verb_similarity

    # returns (result, proof)
    def implied_relation(self, premise, hypothesis):
        verb_similarity = self.verb_similarity(premise[0], hypothesis[0])
        if verb_similarity < self.verb_threshold:
            return self.dissimilar_verbs, hypothesis
        actor_actor = self.noun_intersect_setminus(premise[1], hypothesis[1])
        acted_acted = self.noun_intersect_setminus(premise[2], hypothesis[2])
        actor_acted = self.noun_intersect_setminus(premise[1], hypothesis[2])
        acted_actor = self.noun_intersect_setminus(premise[2], hypothesis[1])
        contained_deps = actor_actor[0] + acted_acted[0]
        missing_deps = actor_actor[1] + acted_acted[1]
        contradiction_deps = actor_acted[0] + acted_actor[0]
        if len(missing_deps) == 0:
            return self.entailment, ("verb similarity:", verb_similarity,
                    "contained dependences:", contained_deps)
        if len(contradiction_deps) > 0:
            return self.contradiction, ("verb similarity:", verb_similarity,
                    "contradictory dependences:", contradiction_deps)
        return self.missing_dependencies, ("verb similarity:",
                verb_similarity, "missing dependencies:", missing_deps)

    def query_relation(self, hypothesis):
        missing_dependencies = []
        contradiction = []
        for premise in self.relations:
            r = self.implied_relation(premise, hypothesis)
            if r[0] == self.entailment:
                return r[0], (premise, r[1])
            elif r[0] == self.missing_dependencies:
                missing_dependencies.append((premise, r[1]))
            elif r[0] == self.contradiction:
                contradiction.append((premise, r[1]))
        if len(contradiction) > 0:
            return self.contradiction, contradiction
        return self.missing_dependencies, missing_dependencies

    # returns (result, proof)
    def query_verb(self, verb):
        return self.query_relation(self.get_relation(verb))

