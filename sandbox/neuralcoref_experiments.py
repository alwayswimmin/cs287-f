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

def print_corefs(src, gen):
    src = nlp(src)
    gen = nlp(gen)
    print(src._.coref_clusters)
    print(gen._.coref_clusters)

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
                print_corefs(src_line, gen_line)
                print("source:", src_line[:50])
                print("summary:", gen_line[:50])
    
    # sns.set()
    # ax = sns.distplot(scores)
    # plt.show()
