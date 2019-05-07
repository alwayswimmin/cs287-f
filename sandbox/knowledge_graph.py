import argparse
import sys
import spacy
import neuralcoref
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored

import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp, greedyness=0.50, max_dist=500)

print_scores = False
verbose = False
draw = False

class KnowledgeGraph:

    def __init__(self):
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

    def is_generic(self, token):
        return token.pos_ == "PRON" or token.pos_ == "DET"

    def get_valid_cluster_tokens(self, noun):
        tokens = list()
        if (noun.pos_ == 'PRON' or noun.pos_ == 'DET') and noun.head.dep_ == 'relcl':
            # the head is the verb of the relative clause
            # the head of the verb should be the noun this thing refers to
            if verbose:
                print("found relative clause, replacing", noun, "with", noun.head.head)
            noun = noun.head.head
        for cluster in noun._.coref_clusters:
            for span in cluster:
                for token in span:
                    if not self.is_generic(token):
                        tokens.append(token)
        if len(tokens) == 0:
            if self.is_generic(noun) and verbose:
                print(colored("warning:", "yellow"), "using generic token", noun)
            tokens.append(noun)
        return tokens 

    def noun_same(self, n1, n2):
        tokens1 = self.get_valid_cluster_tokens(n1)
        tokens2 = self.get_valid_cluster_tokens(n2)
        maximum_similarity = 0
        maximum_pair = None
        for token1 in tokens1:
            for token2 in tokens2:
                token_similarity = token1.similarity(token2)
                if token_similarity > maximum_similarity:
                    maximum_similarity = token_similarity
                    maximum_pair = token1, token2
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
                if verbose:
                    print(n, n2, r)
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
    if verbose:
        print("source:", src_line[:50])
        print("summary:", gen_line[:50])
    src = nlp(src)
    gen = nlp(gen)
    if verbose:
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
                if verbose:
                    print("contained |", relation, "|", r[1])
                contained += 1
            elif r[0] == kg.missing_dependencies:
                if verbose:
                    print(colored("missing", "red"), "|", relation, "|", r[1])
            elif r[0] == kg.contradiction:
                if verbose:
                    print(colored("contradiction", "red"), "|", relation, "|", r[1])
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
    parser = argparse.ArgumentParser(description='Analyze Bottom-Up Abstraction Outputs.')
    parser.add_argument('indices', metavar='i', type=int, nargs='+', default=-1,
                        help='indices to test')
    parser.add_argument('--print_scores', dest='print_scores',
                        action='store_const', const=True, default=False,
                        help='score prints (default: False)')
    parser.add_argument('--draw', dest='draw', action='store_const',
                        const=True, default=False,
                        help='draw histogram (default: False)')
    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help='verbose prints (default: False)')
    args = parser.parse_args()
    indices = args.indices
    print_scores = args.print_scores
    verbose = args.verbose
    draw = args.draw

    scores = []
    with open("data/test.txt.src.tagged.shuf.400words") as src:
        with open("data/bottom_up_cnndm_015_threshold.out") as gen:
            for i, (src_line, gen_line) in enumerate(zip(src, gen)):
                if -1 not in indices and i not in indices:
                    continue
                src_line = clean_src(src_line)
                gen_line = clean_gen(gen_line)
                score = test(src_line, gen_line)
                if print_scores:
                    print(i, "score:", score)
                scores.append(score)
    
    if draw:
        sns.set()
        ax = sns.distplot(scores)
        plt.show()
