import spacy
import neuralcoref
import numpy as np
import util
from termcolor import colored
from BERT.nli_classification import bert_nli_classification

import warnings
warnings.filterwarnings("ignore")

class TokenEquivalency:
    def __init__(self, verbose=False):
        self.verbose = verbose
    # return a list of tokens considered equivalent
    def __call__(self, token):
        return list()

class CompoundEquivalency(TokenEquivalency):
    def __init__(self, verbose=False):
        super(CompoundEquivalency, self).__init__(verbose=verbose)
    def __call__(self, token):
        equiv = list()
        for child in token.children:
            if child.dep_ == 'compound':
                equiv.append(child)
        return equiv

class KnowledgeGraph:
    entailment = 0
    missing_verb = 1
    # could not find a verb that compares favorably
    missing_dependencies = 2
    # generic missing dependency fallback
    contradiction = 3
    # a contradiction in grammar: a noun is claimed to be the subject when it
    # is really the object, or vice versa.
    missing_actors = 4
    # actors are missing, but all acteds are found
    missing_acteds = 5
    # acteds are missing, but all actors are found
    invalid_simplification = 6
    # invalid simplification: a subject, verb pair and a verb, object pair is
    # collapsed to a subject, verb, object tuple, but that tuple is unattested.
    entailment_bert = 7
    # entailed, but requiring BERT support, which is sometimes shaky.
    contradiction_bert = 8
    # verb contradiction: (A, V, B) is compared against (A, V2, B) in the
    # source document, but V2 and V are deemed to be contradictory by BERT

    def __init__(self, nlp, equivalencies=list(), use_bert=False,
                 use_acomps_attrs=True, verbose=False):
        self.nlp = nlp
        self.use_bert = use_bert
        self.verbose = verbose
        self.equivalencies = equivalencies
        self.use_acomps_attrs = use_acomps_attrs
        self.relations = list()
        self.noun_threshold = 0.8
        self.verb_threshold = 0.9

    def get_relation(self, verb):
        actors = util.get_actors(verb)
        acteds = util.get_acteds(verb) 
        if self.use_acomps_attrs and verb.lemma_ == "be":
            acteds.extend(util.get_acomps_attrs(verb))
        return verb, actors, acteds

    def add_verb(self, verb):
        self.relations.append(self.get_relation(verb))

    def add_document(self, doc):
        for token in doc:
            if token.pos_ == "VERB":
                self.add_verb(token)

    def get_sub_cluster(self, noun, use_generic=False):
        tokens = list()
        if util.is_generic(noun) and noun.head.dep_ == 'relcl':
            # the head is the verb of the relative clause
            # the head of the verb should be the noun this thing refers to
            if self.verbose:
                print("found relative clause, replacing", noun, "with", 
                        noun.head.head)
            noun = noun.head.head
        for cluster in noun._.coref_clusters:
            # for span in cluster:
            #     for token in span:
            if util.is_cluster_root(noun, cluster):
                for token in util.get_cluster_roots(cluster):
                    if use_generic or not util.is_generic(token):
                        if self.verbose and util.is_generic(token):
                            print(colored("warning:", "yellow"), 
                                    "using generic token", noun)
                        tokens.append(token)
        if len(tokens) == 0:
            if use_generic or not util.is_generic(noun):
                if self.verbose and util.is_generic(noun):
                    print(colored("warning:", "yellow"), 
                            "using generic token", noun)
                tokens.append(noun)
        return tokens 

    def get_cluster(self, noun, use_generic=False):
        cluster = self.get_sub_cluster(noun, use_generic)
        for equivalency in self.equivalencies:
            for token in equivalency(noun):
                cluster.extend(self.get_sub_cluster(token, use_generic))
        return cluster

    def noun_similarity(self, n1, n2):
        tokens1 = self.get_cluster(n1)
        tokens2 = self.get_cluster(n2)
        if len(tokens1) == 0 or len(tokens2) == 0:
            tokens1 = self.get_cluster(n1, True)
            tokens2 = self.get_cluster(n2, True)
        maximum_similarity = 0
        maximum_pair = None
        for token1 in tokens1:
            for token2 in tokens2:
                token_similarity = token1.similarity(token2)
                if token_similarity > maximum_similarity:
                    maximum_similarity = token_similarity
                    maximum_pair = token1, token2
        if maximum_similarity > self.noun_threshold:
            return maximum_similarity, ("best match:", maximum_similarity,
                    maximum_pair)
        return maximum_similarity, ("best match:", maximum_similarity,
                maximum_pair)

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
    def implied_relation(self, premise, hypothesis,
                         ignore_verb_dissimilarity=False):
        verb_similarity = self.verb_similarity(premise[0], hypothesis[0])
        if not ignore_verb_dissimilarity and \
                verb_similarity < self.verb_threshold:
            return self.missing_verb, ("verb similarity:", verb_similarity,
                    "missing verb", hypothesis)
        actor_actor = self.noun_intersect_setminus(premise[1], hypothesis[1])
        acted_acted = self.noun_intersect_setminus(premise[2], hypothesis[2])
        actor_acted = self.noun_intersect_setminus(premise[1], hypothesis[2])
        acted_actor = self.noun_intersect_setminus(premise[2], hypothesis[1])
        contained_deps = actor_actor[0] + acted_acted[0]
        missing_actors = actor_actor[1] 
        missing_acteds = acted_acted[1]
        contradiction_deps = actor_acted[0] + acted_actor[0]
        if len(missing_actors) == 0 and len(missing_acteds) == 0:
            return KnowledgeGraph.entailment, ("verb similarity:",
                    verb_similarity, "contained dependences:", contained_deps)
        if len(contradiction_deps) > 0:
            return KnowledgeGraph.contradiction, ("verb similarity:",
                    verb_similarity,
                    "contradictory dependences:", contradiction_deps)
        if len(missing_actors) == 0:
            return KnowledgeGraph.missing_acteds, ("verb similarity:",
                    verb_similarity, "missing acteds:", missing_acteds)
        if len(missing_acteds) == 0:
            return KnowledgeGraph.missing_actors, ("verb similarity:",
                    verb_similarity, "missing actors:", missing_actors)
        return KnowledgeGraph.missing_dependencies, ("verb similarity:",
                verb_similarity, "missing dependencies:", missing_actors +
                missing_acteds)

    def query_relation(self, hypothesis):
        missing_deps = []
        missing_actors = []
        missing_acteds = []
        contradiction_deps = []
        best_verb_similarity = 0.0
        closest_verb_premise = None
        entailed_without_verb = []
        for premise in self.relations:
            r = self.implied_relation(premise, hypothesis)
            if r[0] == KnowledgeGraph.entailment:
                return r[0], [(premise, r[1])]
            elif r[0] == KnowledgeGraph.missing_dependencies:
                missing_deps.append((premise, r[1]))
            elif r[0] == KnowledgeGraph.missing_actors:
                missing_actors.append((premise, r[1]))
            elif r[0] == KnowledgeGraph.missing_acteds:
                missing_acteds.append((premise, r[1]))
            elif r[0] == KnowledgeGraph.contradiction:
                contradiction_deps.append((premise, r[1]))
            elif r[0] == KnowledgeGraph.missing_verb:
                if r[1][1] > best_verb_similarity:
                    closest_verb_premise = premise
                    best_verb_similarity = r[1][1]
            r = self.implied_relation(premise, hypothesis,
                                      ignore_verb_dissimilarity=True)
            if r[0] == KnowledgeGraph.entailment:
                entailed_without_verb.append((premise, r[1]))
        if len(contradiction_deps) > 0:
            return KnowledgeGraph.contradiction, contradiction_deps
        if len(missing_actors) > 0 and len(missing_acteds) > 0:
            return KnowledgeGraph.invalid_simplification, \
                    missing_actors + missing_acteds
        if len(missing_actors) > 0:
            return KnowledgeGraph.missing_actors, missing_actors
        if len(missing_acteds) > 0:
            return KnowledgeGraph.missing_acteds, missing_acteds
        if len(entailed_without_verb) > 0 and self.use_bert:
            print(entailed_without_verb)
            hypothesis_sent = util.get_containing_phrase(hypothesis)
            contradiction_bert = list()
            for premise, proof in entailed_without_verb:
                premise_sent = util.get_containing_phrase(premise)
                logits = bert_nli_classification(premise_sent,
                                                 hypothesis_sent)
                if logits.argmax() == 1: # entailment
                    return KnowledgeGraph.entailment_bert, \
                            [(premise, proof, logits)]
                if logits.argmax() == 0: # contradiction
                    contradiction_bert.append((premise, proof, logits))
            if len(contradiction_bert) is not 0:
                return KnowledgeGraph.contradiction_bert, contradiction_bert
        if len(entailed_without_verb) > 0:
            return KnowledgeGraph.missing_verb, entailed_without_verb
        # uncomment this to instead return the closest verb in the event that
        # no actual verb to which we can compare is found.
        # if len(missing_deps) > 0:
        #     return KnowledgeGraph.missing_dependencies, missing_deps
        # return KnowledgeGraph.missing_verb, [(closest_verb_premise, r[1])]
        return KnowledgeGraph.missing_dependencies, missing_deps

    # returns (result, proof)
    def query_verb(self, verb):
        return self.query_relation(self.get_relation(verb))

