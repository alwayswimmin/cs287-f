import argparse
import sys
import spacy
import neuralcoref
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from knowledge_graph import KnowledgeGraph, CompoundEquivalency
from annotator import Annotator
from speaker_pronoun_equivalency import SpeakerPronounEquivalency
import util
from rouge import Rouge

def test(nlp, src, gen, bert=False, print_annotations=False, print_latex=False,
         verbose=False):
    if print_annotations:
        print("source:", src_line[:50])
        print("summary:", gen_line[:50])
    src = nlp(src)
    gen = nlp(gen)
    if verbose:
        print("clusters:", src._.coref_clusters, gen._.coref_clusters)
    ce = CompoundEquivalency()
    spe = SpeakerPronounEquivalency()
    spe.register(src)
    spe.register(gen)
    kg = KnowledgeGraph(nlp, use_bert=bert, equivalencies=[ce, spe],
                        verbose=verbose)
    if print_annotations:
        annotator = Annotator(src, gen, latex=print_latex)
    kg.add_document(src)
    contained = 0
    contained_bert = 0
    missing = 0
    missing_verb = 0
    missing_actors = 0
    missing_acteds = 0
    contradiction = 0
    contradiction_bert = 0
    invalid_simplification = 0
    total = 0
    for token in gen:
        if token.pos_ == "VERB":
            total += 1
            relation = kg.get_relation(token)
            r = kg.query_relation(relation)
            if r[0] == KnowledgeGraph.entailment:
                if print_annotations:
                    print(util.format("contained", "blue", latex=print_latex),
                            "|", relation, "|", r[1])
                contained += 1
            if r[0] == KnowledgeGraph.entailment_bert:
                if print_annotations:
                    print(util.format("contained (BERT)", "blue",
                            latex=print_latex), "|", relation, "|", r[1])
                contained_bert += 1
            if r[0] == KnowledgeGraph.contradiction_bert:
                if print_annotations:
                    print(util.format("contradiction (BERT)", "red",
                            latex=print_latex), "|", relation, "|", r[1])
                contradiction_bert += 1
            elif r[0] == KnowledgeGraph.missing_dependencies:
                missing += 1
                if print_annotations:
                    print(util.format("generic missing dependency", "yellow",
                            latex=print_latex), "|", relation, "|", r[1])
            elif r[0] == KnowledgeGraph.missing_actors:
                missing_actors += 1
                if print_annotations:
                    print(util.format("missing actors", "magenta",
                            latex=print_latex), "|", relation, "|", r[1])
            elif r[0] == KnowledgeGraph.missing_acteds:
                missing_acteds += 1
                if print_annotations:
                    print(util.format("missing acteds", "magenta",
                            latex=print_latex), "|", relation, "|", r[1])
            elif r[0] == KnowledgeGraph.missing_verb:
                missing_verb += 1
                if print_annotations:
                    print(util.format("missing verb", "magenta",
                            latex=print_latex), "|", relation, "|", r[1])
            elif r[0] == KnowledgeGraph.invalid_simplification:
                invalid_simplification += 1
                if print_annotations:
                    print(util.format("invalid simplification", "magenta",
                            latex=print_latex), "|", relation, "|", r[1])
            elif r[0] == KnowledgeGraph.contradiction:
                contradiction += 1
                if print_annotations:
                    print(util.format("contradiction", "red",
                            latex=print_latex), "|", relation, "|", r[1])
            if print_annotations:
                annotator.annotate(relation, r)
    if print_annotations:
        annotated_document, annotated_summary = annotator.annotated()
        print("Document:", " ".join(annotated_document))
        print("Summary:", " ".join(annotated_summary))
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return 100.0 * contained / total, \
            100.0 * contained_bert / total, \
            100.0 * missing / total, \
            100.0 * missing_verb / total, \
            100.0 * missing_actors / total, 
            100.0 * missing_acteds / total, \
            100.0 * contradiction / total, \
            100.0 * contradiction_bert / total, \
            100.0 * invalid_simplification / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Summary Outputs.')
    parser.add_argument('--document', dest='document', metavar='d', type=str,
            default='data/bottom-up/test.txt.src.tagged.shuf.400words',
            help='source document path (default: bottom-up)')
    parser.add_argument('--summary', dest='summary', metavar='s', type=str,
            default='data/bottom-up/bottom_up_cnndm_015_threshold.out',
            help='generated summary path (default: bottom-up)')
    parser.add_argument('--reference', dest='reference', metavar='r', type=str,
            default='data/bottom-up/test.txt.tgt.tagged.shuf.noslash',
            help='reference summary path (default: bottom-up)')
    parser.add_argument('--cache-dir', dest='cache_dir', metavar='c', type=str,
            default='', 
            help="directory for cached numpy entries (default: don't cache)")
    parser.add_argument('--indices', dest='indices', metavar='i', type=int,
            nargs='*',
            default=[], help='indices to test (default: everything)')
    parser.add_argument('--language-model', dest='lm', metavar='m', type=str,
                        default='en_core_web_lg',
                        help='language model (default: en_core_web_lg)')
    parser.add_argument('--print-scores', dest='print_scores',
                        action='store_const', const=True, default=False,
                        help='score prints (default: False)')
    parser.add_argument('--print-annotations', dest='print_annotations',
                        action='store_const', const=True, default=False,
                        help='document annotation prints (default: False)')
    parser.add_argument('--print-latex', dest='print_latex',
                        action='store_const', const=True, default=False,
                        help='prints in LaTeX mode (default: False)')
    parser.add_argument('--draw-histogram', dest='draw', action='store_const',
                        const=True, default=False,
                        help='draw histogram (default: False)')
    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help='verbose prints. Implies --print-scores and '
                        '--print-annotations (default: False)')
    parser.add_argument('--copy', dest='copy', action='store_const',
                        const=True, default=False,
                        help='calculate copy lengths (default: False)')
    parser.add_argument('--no-test', dest='no_test', action='store_const',
                        const=True, default=False,
                        help='Do not run factual accuracy. Useful for running'
                        ' helper functions only. (default: False)')
    parser.add_argument('--rouge', dest='rouge', action='store_const',
                        const=True, default=False,
                        help='calculate ROUGE scores (default: False)')
    parser.add_argument('--bert', dest='bert', action='store_const',
                        const=True, default=False,
                        help='use BERT to resolve cases where only the verb'
                        'disagrees (default: False)')

    args = parser.parse_args()
    document = args.document
    summary = args.summary
    reference = args.reference
    cache_dir = args.cache_dir
    indices = args.indices
    print_scores = args.print_scores
    print_annotations = args.print_annotations
    print_latex = args.print_latex
    verbose = args.verbose
    copy = args.copy
    no_test = args.no_test
    rouge = args.rouge
    bert = args.bert

    if verbose:
        print_scores = print_annotations = True

    draw = args.draw
    language_model = args.lm

    if not no_test:
        nlp = spacy.load(language_model)
        neuralcoref.add_to_pipe(nlp, greedyness=0.50, max_dist=500)

    contained_scores = []
    contained_bert_scores = []
    missing_scores = []
    missing_verb_scores = []
    missing_actors_scores = []
    missing_acteds_scores = []
    contradiction_scores = []
    contradiction_bert_scores = []
    invalid_simplification_scores = []
    rouge_scores = []
    average_copy_lengths = []
    r = Rouge()
    with open(document) as src:
        with open(summary) as gen:
            with open(reference) as tgt:
                for i, (src_line, gen_line, tgt_line) in \
                        enumerate(zip(src, gen, tgt)):
                    if len(indices) > 0 and i not in indices:
                        continue
                    src_line = util.clean(src_line)
                    gen_line = util.clean(gen_line)
                    tgt_line = util.clean(tgt_line)

                    if print_scores:
                        print(i)
                    if not no_test:
                        score = test(nlp, src_line, gen_line, bert=bert,
                                     print_annotations=print_annotations,
                                     print_latex=print_latex, verbose=verbose)
                        contained, contained_bert, missing, missing_verb, \
                                missing_actors, missing_acteds, \
                                contradiction, contradiction_bert, \
                                invalid_simplification = score
                        contained_scores.append(contained)
                        contained_bert_scores.append(contained_bert)
                        missing_scores.append(missing)
                        missing_verb_scores.append(missing_verb)
                        missing_actors_scores.append(missing_actors)
                        missing_acteds_scores.append(missing_acteds)
                        contradiction_scores.append(contradiction)
                        contradiction_bert_scores.append(contradiction_bert)
                        invalid_simplification_scores.append(
                                invalid_simplification)
                        if print_scores:
                            print("score:", score, end = "\t")
                    if copy:
                        average_copy_length = util.average_copy_length(
                                src_line, gen_line)
                        average_copy_lengths.append(average_copy_length)
                        if print_scores:
                            print("average copy length:", average_copy_length,
                                    end = "\t")
                    if rouge:
                        rouge_score = r.get_scores(gen_line, tgt_line)
                        rouge_scores += rouge_score
                        # if print_scores:
                        #     print("rouge:", rouge_score, end="\t")
                    if print_scores:
                        print()

                    if cache_dir and (i+1) % 500 == 0:
                        if not no_test:
                            np.save(cache_dir + "scores" + str(i+1),
                                    contained_scores)
                            np.save(cache_dir + "contained_bert_scores" +
                                    str(i+1), contained_bert_scores)
                            np.save(cache_dir + "missing_scores" + str(i+1),
                                    missing_scores)
                            np.save(cache_dir + "missing_verb_scores" +
                                    str(i+1), missing_verb_scores)
                            np.save(cache_dir + "missing_actors_scores" +
                                    str(i+1), missing_actors_scores)
                            np.save(cache_dir + "missing_acteds_scores" +
                                    str(i+1), missing_acteds_scores)
                            np.save(cache_dir + "contradiction_scores" +
                                    str(i+1), contradiction_scores)
                            np.save(cache_dir + "contradiction_bert_scores" +
                                    str(i+1), contradiction_bert_scores)
                            np.save(cache_dir + "invalid_simplification_scores"
                                    + str(i+1), invalid_simplification_scores)
                        if copy:
                            np.save(cache_dir + "average_copy_lengths" +
                                    str(i+1), average_copy_lengths)
                        if rouge:
                            np.save(cache_dir + "rouge" + str(i+1),
                                    rouge_scores)
                    


    if cache_dir:
        if not no_test:
            np.save(cache_dir + "scores", contained_scores)
            np.save(cache_dir + "contained_bert_scores", contained_bert_scores)
            np.save(cache_dir + "missing_scores", missing_scores)
            np.save(cache_dir + "missing_verb_scores", missing_verb_scores)
            np.save(cache_dir + "missing_actors_scores", missing_actors_scores)
            np.save(cache_dir + "missing_acteds_scores", missing_acteds_scores)
            np.save(cache_dir + "contradiction_scores", contradiction_scores)
            np.save(cache_dir + "contradiction_bert_scores",
                    contradiction_bert_scores)
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
