import argparse
import sys
import spacy
import neuralcoref
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from knowledge_graph import KnowledgeGraph

def test(nlp, src, gen, verbose=False):
    if verbose:
        print("source:", src_line[:50])
        print("summary:", gen_line[:50])
    src = nlp(src)
    gen = nlp(gen)
    if verbose:
        print("clusters:", src._.coref_clusters)
    kg = KnowledgeGraph(verbose)
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

def clean(s):
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
        elif(len(s2) > 0 and len(w) > 1 and w[0] == '\''):
            s2[-1] = s2[-1]+w
        elif not in_paren and not (w == '<t>' or w == '</t>'):
            s2.append(w)
    return ' '.join(s2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Summary Outputs.')
    parser.add_argument('--document', dest='document', metavar='d', type=str, 
                        default='data/bottom-up/test.txt.src.tagged.shuf.400words', help='source document path')
    parser.add_argument('--summary', dest='summary', metavar='s', type=str, 
                        default='data/bottom-up/bottom_up_cnndm_015_threshold.out', help='generated summary path')
    parser.add_argument('--indices', dest='indices', metavar='i', type=int, nargs='*',
                        default=[], help='indices to test')
    parser.add_argument('--language-model', dest='lm', metavar='m', type=str,
                        default='en_core_web_lg',
                        help='language model (default: en_core_web_lg)')
    parser.add_argument('--print-scores', dest='print_scores',
                        action='store_const', const=True, default=False,
                        help='score prints (default: False)')
    parser.add_argument('--draw-histogram', dest='draw', action='store_const',
                        const=True, default=False,
                        help='draw histogram (default: False)')
    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help='verbose prints (default: False)')
    args = parser.parse_args()
    document = args.document
    summary = args.summary
    indices = args.indices
    print_scores = args.print_scores
    verbose = args.verbose
    draw = args.draw
    language_model = args.lm

    nlp = spacy.load(language_model)
    neuralcoref.add_to_pipe(nlp, greedyness=0.50, max_dist=500)

    scores = []
    with open(document) as src:
        with open(summary) as gen:
            for i, (src_line, gen_line) in enumerate(zip(src, gen)):
                if len(indices) > 0 and i not in indices:
                    continue
                src_line = clean(src_line)
                gen_line = clean(gen_line)
                score = test(nlp, src_line, gen_line, verbose)
                if print_scores:
                    print(i, "score:", score)
                scores.append(score)
    
    if draw:
        sns.set()
        ax = sns.distplot(scores)
        plt.show()
