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
neuralcoref.add_to_pipe(nlp)

print_scores = False
verbose = False
draw = False


class KnowledgeGraph:
    
    def __init__(self):
        self.relations = list()
        self.noun_threshold = 0.9
        self.verb_threshold = 0.9
        self.weak_threshold = 0.5
        self.entailment = 0
        self.entailment_dissimilar_verbs = 0.5
        self.dissimilar_verbs = 1
        self.missing_dependencies = 2
        self.contradiction = 3

    # ==========================================
    # 1) adding to KnowledgeGraph relations 
    # ==========================================
    def add_verb(self, verb):
        self.relations.append(self.get_relation(verb))
        
    ##### extracting relations from sentence #####
    def get_relation(self, verb):
        # get all equivalent verbs
        verb_cluster = self.get_verb_cluster(verb)
        actors = []
        acteds = []
        
        # get all actors/acteds of verbs in equivalencies
        for verb in verb_cluster:
            actors += self.get_actors(verb)
            acteds += self.get_acteds(verb)
        return verb_cluster, actors, acteds
    
    # =========================================
    # 2) looks through verb's children for
    # verb equivalencies (xcomp)
    # =========================================
    def get_verb_cluster(self, verb):
        verb_cluster = [verb]
        for child in verb.children:
            if child.dep_ == "xcomp":# or child.dep_ == "ccomp":
                verb_cluster.append(child)
        return verb_cluster
        
    def get_actors(self, verb):
        actors = []
        for child in verb.children:
            # child is a nominative subject
            if child.dep_ == "nsubj":
                actors.append(child)
            # child is something like "by"
            elif child.dep_ == "agent":  
                # passive, look for true actor
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        actors.append(grandchild)
        return actors

    def get_acteds(self, verb):
        acteds = []
        for child in verb.children:
            #child is direct object or passive subject
            if child.dep_ == "dobj" or child.dep_ == "nsubjpass":
                acteds.append(child)
        return acteds

    # =========================================
    # 3) checking hypothesis relation against 
    # premise's KnowledgeGraph relations
    # =========================================
    def query_relation(self, hypothesis):
        missing_dependencies = []
        contradiction = []
        for premise in self.relations:
            r = self.implied_relation(premise, hypothesis)

            # once we find that hypothesis is contained, accept this relation as verified
            # if the verb similarity is too low, we make note of this but still mark it as entailed
            if r[0] == self.entailment:
                return r[0], [(premise,r[1])]
            elif r[0] == self.missing_dependencies:
                missing_dependencies.append((premise, r[1]))
            elif r[0] == self.contradiction:
                contradiction.append((premise, r[1]))
        if len(contradiction) > 0:
            return self.contradiction, contradiction
        return self.missing_dependencies, missing_dependencies
    
    # check if a hypothesis is verified by a premise 
    # returns (result, proof)
    def implied_relation(self, premise, hypothesis):
        # premise[0] and hypothesis[0] is a list (verb cluster)
        verb_similarity, best_pair = self.verb_same(premise[0], hypothesis[0])
        if verb_similarity < self.verb_threshold:
            return self.dissimilar_verbs, hypothesis
        
        # check setminus of premise \ hypothesis
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

    
    # ========================
    # verb helper functions
    # ========================
    # v1 comes from premise/source, v2 comes from hypothesis/output
    def verb_same(self, v1_cluster, v2_cluster):
        maximum_similarity = 0
        maximum_pair = None
        for v1 in v1_cluster:
            for v2 in v2_cluster:
                similarity = v1.similarity(v2)
                if(similarity > maximum_similarity):
                    maximum_similarity = similarity
                    maximum_pair = v1, v2
        return maximum_similarity, maximum_pair
    

    # ========================
    # noun helper functions
    # ========================
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

    def noun_same(self, n1, n2):
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
            return True, ("best match:", maximum_similarity, maximum_pair)
        return False, ("best match:", maximum_similarity, maximum_pair)
    
    def get_valid_cluster_tokens(self, noun, use_generic=False):
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
                    if use_generic or not self.is_generic(token):
                        if verbose and self.is_generic(token):
                            print(colored("warning:", "yellow"), "using generic token", noun)
                        tokens.append(token)
        if len(tokens) == 0:
            if use_generic or not self.is_generic(noun):
                if verbose and self.is_generic(noun):
                    print(colored("warning:", "yellow"), "using generic token", noun)
                tokens.append(noun)
        return tokens 

    def is_generic(self, token):
        return token.pos_ == "PRON" or token.pos_ == "DET"


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



def visualize(src0, gen0, important_relations):
    colors = {'contained':lambda text: '\033[0;32m' + text + '\033[0m', 
              'missing':lambda text: '\033[0;33m' + text + '\033[0m', 
              'contradiction':lambda text: '\033[0;31m' + text + '\033[0m'}

    colored_src = src0
    colored_gen = gen0
    for order in ['missing', 'contained', 'contradiction']:
        for relation_tuple in important_relations:
            result = relation_tuple[0]
            relation = relation_tuple[1]
            proof = relation_tuple[2]
            if not(result == order):
                continue
            # color output doc
            verbs = relation[0]
            actors = relation[1]
            acteds = relation[2]
            for verb in verbs:
                colored_gen[verb.i] = colors[result](verb.text)
            for a in actors:
                colored_gen[a.i] = colors[result](a.text)
            for a in acteds:
                colored_gen[a.i] = colors[result](a.text)

            # color source doc
            for p in proof:
                for verb in p[0][0]:
                    colored_src[verb.i] = colors[result](verb.text)
                for a in p[0][1]:
                    colored_src[a.i] = colors[result](a.text)
                for a in p[0][2]:
                    colored_src[a.i] = colors[result](a.text)

    colored_src = ' '.join(colored_src)
    colored_gen = ' '.join(colored_gen)

    return colored_src, colored_gen

def test(src, gen):
#     print("source:", src_line[:100])
#     print("summary:", gen_line[:100])
    src = nlp(src)
    gen = nlp(gen)
#     print("clusters:", src._.coref_clusters)
    kg = KnowledgeGraph()

    # put all actors/acteds for each verb into knowledge graph
    for ixt, token in enumerate(src):
        if token.pos_ == "VERB":
            kg.add_verb(token)
    important_relations = []
    contained = 0
    missing = 0
    contradiction = 0
    total = 0
    
    for token in gen:
        # ignore xcomp verbs "tried TO EAT" since will later be added to verb cluster
        # still adds was/has/is/aux verbs though
        if token.pos_ == "VERB" and not(token.dep_=='xcomp'):# or token.dep_=='ccomp'):
            relation = kg.get_relation(token)
            # skip those relations with no actors/acteds
            if (len(relation[1]) + len(relation[2]) == 0):
                continue
            
            total += 1
            r = kg.query_relation(relation)
            if r[0] == kg.entailment:
                contained += 1
                important_relations.append(('contained', relation, r[1]))
                if(verbose):
                    print("contained |", relation, "|", r[1])
#             elif r[0] == kg.entailment_dissimilar_verbs:
#                 missing += 1
#                 important_relations.append(('contained-noverb', relation, r[1]))
#                 if(verbose):
#                     print("contained-noverb |", relation, "|", r[1])
            elif r[0] == kg.missing_dependencies:
                missing += 1
                important_relations.append(('missing', relation, r[1]))
                if(verbose):
                    print(colored("missing", "yellow"), "|", relation, "|", r[1])
            elif r[0] == kg.contradiction:
                contradiction += 1
                important_relations.append(('contradiction', relation, r[1]))
                if(verbose):
                    print(colored("contradiction", "red"), "|", relation, "|", r[1])
    
    important_relations = sorted(important_relations)
    colored_src, colored_gen = visualize([word.text for word in src], [word.text for word in gen], important_relations)
    
    if total == 0:
        return important_relations, (0.0, 0.0, 0.0), (colored_src, colored_gen)
    return important_relations, (100.0 * contained / total, 
                                 100.0 * missing / total, 
                                 100.0 * contradiction / total), (colored_src, colored_gen)

# returns average number of tokens copied = max copy length / unique phrases copied
def avg_copy_length(src,gen):
    src = src.split()
    gen = gen.split()
    substrings = {}
    for ixgw,word in enumerate(gen):
        substrings[ixgw] = []
    
    avg_length = 0
    num_copied = 0
    ixgw = 0
    while(ixgw < len(gen)):
        gen_word = gen[ixgw]
        max_js = []
        src_ixs = []
        for ixsw, src_word in enumerate(src):
            j = 0
            while(ixgw+j <= len(gen) and ixsw+j <= len(src) and src[ixsw:ixsw+j] == gen[ixgw:ixgw+j]):
                j += 1
            if(len(max_js) == 0 or j > max_js[0]):
                max_js = [j]
                src_ixs = [ixsw]
            elif(j == max_js[0]):
                max_js.append(j)
                src_ixs.append(ixsw)
        substrings[ixgw] = ([gen[ixgw:ixgw+max_j-1] for max_j in max_js], src_ixs)
        ixgw += 1
        
    for ixgw,gen_word in enumerate(gen):
#         substr = substrings[ixgw][0]
#         src_ix = substrings[ixgw][1]
        contained = False
        for src_ix in substrings[ixgw][1]:
            if ixgw > 0 and src_ix-1 in substrings[ixgw-1][1]:
                contained=True
                break
        
        if not contained:
            if(len(substrings[ixgw][0])>0):
                num_copied += 1
#                 print(substrings[ixgw])
#                 print(len(substrings[ixgw][0][0]))
                avg_length += len(substrings[ixgw][0][0])
    avg_length /= num_copied
    
    return avg_length 


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

    line_num = 0
    scores = []
    avg_lengths = []
    with open("data/bottom-up/test.txt.src.tagged.shuf.400words") as src:
        with open("data/bottom-up/test.txt.tgt.tagged.shuf.noslash") as tgt:
            with open("data/bottom-up/bottom_up_cnndm_015_threshold.out") as gen:
                for i, (orig_src_line, tgt_line, gen_line) in enumerate(zip(src, tgt, gen)):
                    if line_num > 0 and not i == line_num:
                        continue
                    if line_num == 0 and i >= 40:
                        break
                    orig_src_line = clean_src(orig_src_line)
                    tgt_line = clean_src(tgt_line)
                    src_line = tgt_line + ' ' + orig_src_line
                    gen_line = clean_gen(gen_line)
                    important_relations, score, (colored_src, colored_gen) = test(src_line, gen_line)
                    print("===========================================================================================")
                    print(f"Src {i}:"%{i:i}, colored_src)
                    print("===========================================================================================")
                    print(f"Summary {i}:"%{i:i}, colored_gen)
                    print("Score:", score)
                    avg_length = avg_copy_length(orig_src_line, gen_line)
                    print("Avg copy length:", avg_length)
                    print("===========================================================================================")
                    print("===========================================================================================")
                    scores.append(score)


                
            

    np.save("experiments/bottom-up/scores", scores)
    np.save("experiments/bottom-up/avg_lengths", avg_lengths)
    
    if draw:
        sns.set()
        ax = sns.distplot(scores)
        plt.show()