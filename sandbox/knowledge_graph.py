import spacy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_lg")

class KnowledgeGraph:
    def __init__(self):
        self.relations = list()
        self.noun_threshold = 0.9
        self.verb_threshold = 0.8
    def add_relation(self, verb):
        self.relations.append(verb)
    def query_relation(self, verb):
        contradiction = None
        for verb2 in self.relations:
            if verb.similarity(verb2) < self.verb_threshold:
                continue
            actor_validated = True
            acted_validated = True
            noun_swapped = False
            for child in verb.children:
                if child.dep_ == "nsubj":
                    actor_validated = False
                    for child2 in verb2.children:
                        if child2.dep_ == "nsubj":
                            if child.similarity(child2) < self.noun_threshold:
                                continue
                            else:
                                actor_validated = True
                        if child2.dep_ == "dobj" or child2.dep_ == "nsubjpass":
                            if child.similarity(child2) < self.noun_threshold:
                                continue
                            else:
                                noun_swapped = True
                if child.dep_ == "dobj" or child.dep_ == "nsubjpass":
                    acted_validated = False
                    for child2 in verb2.children:
                        if child2.dep_ == "nsubj":
                            if child.similarity(child2) < self.noun_threshold:
                                continue
                            else:
                                noun_swapped = True
                        if child2.dep_ == "dobj" or child2.dep_ == "nsubjpass":
                            if child.similarity(child2) < self.noun_threshold:
                                continue
                            else:
                                acted_validated = True
            if actor_validated and acted_validated:
                return True, verb2
            if noun_swapped:
                contradiction = verb2
        return False, contradiction

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
            if r[0]:
                contained += 1
            else:
                print("missing |", token, "|", r[1])
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

scores = []
i = 0
with open("data/test.txt.src.tagged.shuf.400words") as src:
    with open("data/bottom_up_cnndm_015_threshold.out") as gen:
        for src_line, gen_line in zip(src, gen):
            src_line = clean_src(src_line)
            gen_line = clean_gen(gen_line)
            score = test(src_line, gen_line)
            print("source:", src_line[:50])
            print("summary:", gen_line[:50])
            print("score:", score)
            scores.append(score)
            i += 1
            if i == 2:
                break

# sns.set()
# ax = sns.distplot(scores)
# plt.show()
