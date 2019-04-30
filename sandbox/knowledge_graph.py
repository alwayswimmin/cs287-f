import sys
import spacy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

class KnowledgeGraph:

    def __init__(self):
        self.relations = list()
        self.noun_threshold = 0.9
        self.verb_threshold = 0.8
        self.entailment = 0
        self.dissimilar_verbs = 1
        self.missing_dependencies = 2
        self.contradiction = 3

    def add_relation(self, verb):
        self.relations.append(verb)

    # returns (result, proof)
    def implied_relation(self, premise, hypothesis):
        if premise.similarity(hypothesis) < self.verb_threshold:
            return self.dissimilar_verbs, hypothesis
        contained_deps = []
        missing_deps = []
        contradiction = None
        for child in hypothesis.children:
            if child.dep_ == "nsubj":
                actor = None
                for child2 in premise.children:
                    if child.similarity(child2) > self.noun_threshold:
                        if child2.dep_ == "nsubj":
                            actor = child2
                        elif child2.dep_ == "dobj" or child2.dep_ == "nsubjpass":
                            contradiction = child2
                if actor is not None:
                    contained_deps.append(actor)
                else:
                    missing_deps.append(child)
            if child.dep_ == "dobj" or child.dep_ == "nsubjpass":
                acted = None
                for child2 in premise.children:
                    if child.similarity(child2) > self.noun_threshold:
                        if child2.dep_ == "nsubj":
                            contradiction = child2
                        elif child2.dep_ == "dobj" or child2.dep_ == "nsubjpass":
                            acted = child2
                if acted is not None:
                    contained_deps.append(acted)
                else:
                    missing_deps.append(child)
        if len(missing_deps) == 0:
            return self.entailment, contained_deps
        if contradiction is not None:
            return self.contraddiction, contradiction
        return self.missing_dependencies, missing_deps

    # returns (result, proof)
    def query_relation(self, verb):
        missing_dependencies = []
        contradiction = []
        for premise in self.relations:
            r = self.implied_relation(premise, verb)
            if r[0] == self.entailment:
                return r
            elif r[0] == self.missing_dependencies:
                missing_dependencies.append((premise, r[1]))
            elif r[0] == self.contradiction:
                contradiction.append((premise, r[1]))
        if len(contradiction) > 0:
            return self.contradiction, contradiction
        return self.missing_dependencies, missing_dependencies

def test(src, gen):
    src = nlp(src)
    gen = nlp(gen)
    kg = KnowledgeGraph()
    for token in src:
        if token.pos_ == "VERB":
            kg.add_relation(token)
    contained = 0
    total = 0
    for token in gen:
        if token.pos_ == "VERB":
            total += 1
            r = kg.query_relation(token)
            if r[0] == kg.entailment:
                contained += 1
            elif r[0] == kg.missing_dependencies:
                print("missing |", token, "|", r[1])
            elif r[0] == kg.contradiction:
                print("contradiction |", token, "|", r[1])
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
                print("source:", src_line[:50])
                print("summary:", gen_line[:50])
                print("score:", score)
                scores.append(score)
    
    # sns.set()
    # ax = sns.distplot(scores)
    # plt.show()
