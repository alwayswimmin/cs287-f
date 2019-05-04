import spacy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
important_dependencies = ['nsubj', 'dobj', 'nsubjpass']

def simplify(token):
    if token.pos_ == "VERB":
        return token.lemma_
    return token.text

def test(src, gen):
    src = nlp(src)
    gen = nlp(gen)
    dependencies = dict()
    for token in src:
        word, head, dep = simplify(token), simplify(token.head), token.dep_
        if dep not in dependencies:
            dependencies[dep] = dict()
        if word not in dependencies[dep]:
            dependencies[dep][word] = dict()
        if head not in dependencies[dep][word]:
            dependencies[dep][word][head] = list()
        dependencies[dep][word][head].append(token)
    contained = 0
    total = 0
    for token in gen:
        word, head, dep = simplify(token), simplify(token.head), token.dep_
        if dep not in important_dependencies:
            continue
        if dep in dependencies:
            if word in dependencies[dep]:
                if head in dependencies[dep][word]:
                    contained += 1
                else:
                    print("missing |", word, head, dep, "|", dependencies[dep][word])
            else:
                print("missing |", word, head, dep, "| word not in source document under this dependency")
        else:
            print("missing |", word, head, dep, "| dependency not in source document")
    if total == 0:
        return 0.0
    return 100 * contained / total

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

if __name__=='main':
    scores = []
    i = 0
    with open("data/test.txt.src.tagged.shuf.400words") as src:
        with open("data/bottom_up_cnndm_015_threshold.out") as gen:
            for src_line, gen_line in zip(src, gen):
                src_line = clean_src(src_line)
                gen_line = clean_gen(gen_line)
                print("source:", src_line)
                print("summary:", gen_line)
                scores.append(test(src_line, gen_line))
                i += 1
                if i == 5:
                    break

# sns.set()
# ax = sns.distplot(scores)
# plt.show()
