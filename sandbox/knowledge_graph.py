import sys
import spacy
import neuralcoref
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

class KnowledgeGraph:

    def __init__(self):
        self.relations = list()
        self.noun_threshold = 0.9
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
        return actors

    def get_acteds(self, verb):
        acteds = []
        for child in verb.children:
            if child.dep_ == "dobj" or child.dep_ == "nsubjpass":
                acteds.append(child)
        return acteds

    def get_relation(self, verb):
        actors = self.get_actors(verb)
        acteds = self.get_acteds(verb)
        return verb, actors, acteds

    def add_verb(self, verb):
        self.relations.append(self.get_relation(verb))

    def is_not_generic(self, span):
        for word in span:
            if word.pos_ != "PRON":
                return True
        return False

    def get_valid_cluster_objects(self, noun):
        spans = list()
        # spans.append(noun)
        for cluster in noun._.coref_clusters:
            for span in cluster:
                if self.is_not_generic(span):
                    spans.append(span)
        return spans 

    def noun_same(self, n1, n2):
        # noun_similarity = n1.similarity(n2)
        # if noun_similarity > self.noun_threshold:
        #     return True, ("nouns match:", noun_similarity)
        spans1 = self.get_valid_cluster_objects(n1)
        spans2 = self.get_valid_cluster_objects(n2)
        maximum_similarity = 0
        maximum_pair = None
        for span1 in spans1:
            for span2 in spans2:
                try:
                    span_similarity = span1.similarity(span2)
                except:
                    continue
                if span_similarity > maximum_similarity:
                    maximum_similarity = span_similarity
                    maximum_pair = span1, span2
        if maximum_similarity > self.noun_threshold:
            return True, ("best match:", maximum_similarity, maximum_pair)
        return False, ("best match:", maximum_similarity, maximum_pair)

    def noun_intersect_setminus(self, supset, subset):
        contained_nouns = []
        missing_nouns = []
        for n in subset:
            contained = False
            for n2 in supset:
                r = self.noun_same(n, n2)
                print(r)
                if r[0]:
                    contained = True
                    contained_nouns.append((n, n2, r[1]))
                    continue
            if not contained:
                missing_nouns.append(n)
        return contained_nouns, missing_nouns

    # returns (result, proof)
    def implied_relation(self, premise, hypothesis):
        verb_similarity = premise[0].similarity(hypothesis[0])
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

def test(src, gen):
    print("source:", src_line[:50])
    print("summary:", gen_line[:50])
    src = nlp(src)
    gen = nlp(gen)
    print("clusters:", src._.coref_clusters)
    kg = KnowledgeGraph()
    for token in src:
        if token.pos_ == "VERB":
            kg.add_verb(token)
    contained = 0
    total = 0
    for token in gen:
        if token.pos_ == "VERB":
            total += 1
            relation = kg.get_relation(token)
            r = kg.query_relation(relation)
            if r[0] == kg.entailment:
                print("contained |", relation, "|", r[1])
                contained += 1
            elif r[0] == kg.missing_dependencies:
                print("missing |", relation, "|", r[1])
            elif r[0] == kg.contradiction:
                print("contradiction |", relation, "|", r[1])
    if total == 0:
        return 0.0
    return 100.0 * contained / total

def clean_src(s):
    s = s.split()
    # remove everything from "-lrb-" to "-rrb-"
    s2 = []
    in_paren = False
    for ixw, w in enumerate(s):
        if(w=="-lrb-"):
            in_paren=True
        elif(w=='-rrb-'):
            in_paren=False
        elif(w=="-lsb-" or w=="-rsb-"):
            continue
        elif(len(w) > 1 and w[0] == '\''):
            s2[-1] = s2[-1]+w
        elif not in_paren and not (w == '<t>' or w == '</t>'):
            s2.append(w)
    return ' '.join(s2)

def clean_gen(s):
    s = s.split()
    # remove everything from "-lrb-" to "-rrb-"
    s2 = []
    in_paren = False
    for w in s:
        if(w=="-lrb-"):
            in_paren=True
        elif(w=='-rrb-'):
            in_paren=False
        elif(w=="-lsb-" or w=="-rsb-"):
            continue
        elif(len(w) > 1 and w[0] == '\''):
            s2[-1] = s2[-1]+w
        elif not in_paren and not(w == '<t>' or w == '</t>'):
            s2.append(w)
    return ' '.join(s2)

if __name__ == "__main__":
    line_num = 0
    if len(sys.argv) > 1:
        line_num = int(sys.argv[1])
    scores = []
    i = 0
    with open("data/test.txt.src.tagged.shuf.400words") as src:
        with open("data/bottom_up_cnndm_015_threshold.out") as gen:
            for src_line, gen_line in zip(src, gen):
                i += 1
                if line_num > 0 and not i == line_num:
                    continue
                if line_num == 0 and i >= 10:
                    break
                src_line = clean_src(src_line)
                gen_line = clean_gen(gen_line)
                score = test(src_line, gen_line)
                print("score:", score)
                scores.append(score)
    
    # sns.set()
    # ax = sns.distplot(scores)
    # plt.show()
