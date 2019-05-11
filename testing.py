import argparse
import sys
import spacy
import neuralcoref
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from knowledge_graph import KnowledgeGraph
from annotator import Annotator
import util
from rouge import Rouge

def test(nlp, src, gen, bert=False, verbose=False):
    if verbose:
        print("source:", src_line[:50])
        print("summary:", gen_line[:50])
    src = nlp(src)
    gen = nlp(gen)
    if verbose:
        print("clusters:", src._.coref_clusters)
    kg = KnowledgeGraph(nlp, use_bert=bert, verbose=verbose)
    if verbose:
        annotator = Annotator(src, gen)
    for token in src:
        if token.pos_ == "VERB":
            kg.add_verb(token)
    contained = 0
    contained_bert = 0
    missing = 0
    contradiction = 0
    invalid_simplification = 0
    total = 0
    for token in gen:
        if token.pos_ == "VERB":
            total += 1
            relation = kg.get_relation(token)
            r = kg.query_relation(relation)
            if r[0] == KnowledgeGraph.entailment:
                if verbose:
                    print("contained |", relation, "|", r[1])
                contained += 1
            if r[0] == KnowledgeGraph.entailment_bert:
                if verbose:
                    print("contained (bert) |", relation, "|", r[1])
                contained_bert += 1
            elif r[0] == KnowledgeGraph.missing_dependencies:
                missing += 1
                if verbose:
                    print(colored("missing actor", "yellow"), "|", relation, "|", r[1])
            elif r[0] == KnowledgeGraph.missing_verb:
                missing += 1
                if verbose:
                    print(colored("missing verb", "yellow"), "|", relation, "|", r[1])
            elif r[0] == KnowledgeGraph.invalid_simplification:
                invalid_simplification += 1
                if verbose:
                    print(colored("invalid simplification", "magenta"), "|",
                            relation, "|", r[1])
            elif r[0] == KnowledgeGraph.contradiction:
                contradiction += 1
                if verbose:
                    print(colored("contradiction", "red"), "|", relation, "|", r[1])
            if verbose:
                annotator.annotate(relation, r)
    if verbose:
        annotated_document, annotated_summary = annotator.annotated()
        print("Document:", " ".join(annotated_document))
        print("Summary:", " ".join(annotated_summary))
    if total == 0:
        return 0.0
    return 100.0 * contained / total, 100.0 * contained_bert / total, 100.0 * missing / total, 100.0 * contradiction / total, 100.0 * invalid_simplification / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Summary Outputs.')
    parser.add_argument('--document', dest='document', metavar='d', type=str, 
            default='data/bottom-up/test.txt.src.tagged.shuf.400words', help='source document path (default: bottom-up)')
    parser.add_argument('--summary', dest='summary', metavar='s', type=str, 
            default='data/bottom-up/bottom_up_cnndm_015_threshold.out', help='generated summary path (default: bottom-up)')
    parser.add_argument('--reference', dest='reference', metavar='r', type=str, 
            default='data/bottom-up/test.txt.tgt.tagged.shuf.noslash', help='reference summary path (default: bottom-up)')
    parser.add_argument('--cache-dir', dest='cache_dir', metavar='c', type=str, 
            default='', help="directory for cached numpy entries (default: don't cache)")
    parser.add_argument('--indices', dest='indices', metavar='i', type=int, nargs='*',
            default=[], help='indices to test (default: everything)')
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
    parser.add_argument('--copy', dest='copy', action='store_const',
                        const=True, default=False,
                        help='calculate copy lengths (default: False)')
    parser.add_argument('--copy-only', dest='copy_only', action='store_const',
                        const=True, default=False,
                        help='calculate average copy length only. Implies copy (default: False)')
    parser.add_argument('--rouge', dest='rouge', action='store_const',
                        const=True, default=False,
                        help='calculate ROUGE scores (default: False)')
    parser.add_argument('--bert', dest='bert', action='store_const',
                        const=True, default=False,
                        help='use BERT to resolve cases where only the verb disagrees (default: False)')

    args = parser.parse_args()
    document = args.document
    summary = args.summary
    reference = args.reference
    cache_dir = args.cache_dir
    indices = args.indices
    print_scores = args.print_scores
    verbose = args.verbose
    copy = args.copy
    copy_only = args.copy_only
    rouge = args.rouge
    bert = args.bert

    if copy_only:
        copy = True
    draw = args.draw
    language_model = args.lm

    if not copy_only:
        nlp = spacy.load(language_model)
        neuralcoref.add_to_pipe(nlp, greedyness=0.50, max_dist=500)

    contained_scores = []
    contained_bert_scores = []
    missing_scores = []
    contradiction_scores = []
    invalid_simplification_scores = []
    rouge_scores = []
    average_copy_lengths = []
    r = Rouge()
    with open(document) as src:
        with open(summary) as gen:
            with open(reference) as tgt:
                for i, (src_line, gen_line, tgt_line) in enumerate(zip(src, gen, tgt)):
                    if len(indices) > 0 and i not in indices:
                        continue
                    src_line = util.clean(src_line)
                    gen_line = util.clean(gen_line)
                    tgt_line = util.clean(tgt_line)

                    if print_scores:
                        print(i, end = "\t")
                    if not copy_only:
                        score = test(nlp, src_line, gen_line, bert=bert, verbose=verbose)
                        contained, contained_bert, missing, contradiction, invalid_simplification = score
                        contained_scores.append(contained)
                        contained_bert_scores.append(contained_bert)
                        missing_scores.append(missing)
                        contradiction_scores.append(contradiction)
                        invalid_simplification_scores.append(invalid_simplification)
                        if print_scores:
                            print("score:", score, end = "\t")
                    if copy:
                        average_copy_length = util.average_copy_length(src_line, gen_line)
                        average_copy_lengths.append(average_copy_length)
                        if print_scores:
                            print("average copy length:", average_copy_length, end = "\t")
                    if rouge:
                        rouge_score = r.get_scores(gen_line, tgt_line)
                        rouge_scores += rouge_score
                        # if print_scores:
                        #     print("rouge:", rouge_score, end="\t")
                    if print_scores:
                        print()
                    


    if cache_dir:
        if not copy_only:
            np.save(cache_dir + "scores", contained_scores)
            np.save(cache_dir + "contained_bert_scores", contained_bert_scores)
            np.save(cache_dir + "missing_scores", missing_scores)
            np.save(cache_dir + "contradiction_scores", contradiction_scores)
            np.save(cache_dir + "invalid_simplification_scores",
                    invalid_simplification_scores)
        if copy:
            np.save(cache_dir + "average_copy_lengths", average_copy_lengths)
        if rouge:
            np.save(cache_dir + "rouge", rouge_scores)

    
    if draw:
        sns.set()
        ax = sns.distplot(scores)
        plt.show()
