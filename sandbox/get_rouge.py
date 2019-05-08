import argparse
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rouge import Rouge
from celine_knowledge_graph import clean_src, clean_gen



if __name__ == "__main__":

    rouge_scores = []
    with open("data/bottom-up/test.txt.tgt.tagged.shuf.noslash") as tgt:
        with open("data/bottom-up/bottom_up_cnndm_015_threshold.out") as gen:
            rouge = Rouge()
            for i, (tgt_line, gen_line) in enumerate(zip(tgt, gen)):
                print(i)
                tgt_line = clean_src(tgt_line)
                gen_line = clean_gen(gen_line)
                
                rouge_scores += rouge.get_scores(gen_line, tgt_line)


    np.save("experiments/bottom-up/rouge", rouge_scores)
