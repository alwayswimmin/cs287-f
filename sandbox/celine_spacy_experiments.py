import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nlp = spacy.load("en_core_web_sm")

def test(src, gen):
    src = nlp(src)
    gen = nlp(gen)
    dependencies = []
    for token in src:
        if(token.dep_ == 'nsubj' or token.dep_ == 'dobj'):
            dependencies.append((token.text, token.head.text, token.dep_))
    contained = 0.
    total = 0.
    for token in gen:
        if(token.dep_ == 'nsubj' or token.dep_ == 'dobj'):
            total += 1.
            if((token.text, token.head.text, token.dep_) in dependencies):
                contained += 1.
    return (contained, total) #100. * contained / total

def clean_src(s):
    return s

def clean_gen(s):
    s = s.split()
    if(s[0] == "-lrb-"):
        s = s[3:]
    s2 = []
    for w in s:
        if not (w == '<t>' or w == '</t>'):
            s2.append(w)
    return ' '.join(s2)

stats = []
with open("../data/test.txt.src.tagged.shuf.400words") as src:
    with open("../data/bottom_up_cnndm_015_threshold.out") as gen:
        for src_line, gen_line in zip(src, gen):
            src_line = clean_src(src_line)
            gen_line = clean_gen(gen_line)
            # print("source:", src_line[:30])
            # print("generated summary:", gen_line[:30])
            # print("score:", test(src_line, gen_line))
            stats.append(test(src_line, gen_line))

np.save(stats, "stats")

sns.distplot([contained/total if total > 0. else 0. for (contained,total) in stats])
plt.show()
