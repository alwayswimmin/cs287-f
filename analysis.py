import argparse
import sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
from rouge import Rouge 
from statsmodels.nonparametric.smoothers_lowess import lowess

sns.set()

def get_ROUGE(reference, hypothesis):
	rouge = Rouge()
	scores = rouge.get_scores(hypothesis, reference)
	return scores


def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Analyze Scores and Average copy Length.')
	parser.add_argument('model', metavar='model', type=str, default="bottom-up",
                        help='name of model to analyze')
	args = parser.parse_args()
	model = args.model

	scores = np.load("experiments/"+model+"/scores.npy")
	avg_lengths = np.load("experiments/"+model+"/avg_lengths.npy")
	rouge_scores = np.load("experiments/"+model+"/rouge.npy")
	rouge1 = [rouge_scores[i]['rouge-1']['f'] for i in range(len(rouge_scores))]
	rouge2 = [rouge_scores[i]['rouge-2']['f'] for i in range(len(rouge_scores))]
	rougeL = [rouge_scores[i]['rouge-l']['f'] for i in range(len(rouge_scores))]


	contained = [scores[i][0] for i in range(len(scores))]
	missing = [scores[i][1] for i in range(len(scores))]
	contradiction = [scores[i][2] for i in range(len(scores))]
	total = [float(scores[i][0]+scores[i][1]+scores[i][2]) for i in range(len(scores))]

	frac_contained = [0 if total[i] == 0 else scores[i][0]/total[i] for i in range(len(scores))]
	frac_contradiction = [0 if total[i] == 0 else scores[i][2]/total[i] for i in range(len(scores))]
	# plt.scatter(frac_contained, rouge1, label='ROUGE-1', marker='.')
	# plt.xlabel("Fraction contained dependencies")
	# plt.ylabel("ROUGE-1")
	# # plt.scatter(contained, rouge2, label='ROUGE-2', marker='.')
	# # plt.scatter(contained, rougeL, label='ROUGE-L', marker='.')
	# plt.legend()
	# plt.show()
	df = pd.DataFrame(data=np.array([frac_contained, frac_contradiction, rouge1, rouge2, rougeL]).T, 
		columns = ['Fraction contained', 'Fraction contradiction', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])

	f, ax = plt.subplots(1,1)
	sns.lineplot(x='Fraction contained', y='ROUGE-1', data=df, ci = 'sd', ax=ax, label='ROUGE-1')
	sns.lineplot(x='Fraction contained', y='ROUGE-2', data=df, ci = 'sd', ax=ax, label='ROUGE-2')
	sns.lineplot(x='Fraction contained', y='ROUGE-L', data=df, ci = 'sd', ax=ax, label='ROUGE-L')
	plt.xlabel("Fraction of Contained Dependencies")
	plt.ylabel("ROUGE score")
	plt.title("ROUGE score vs. Contained Dependencies")
	plt.legend()
	plt.tight_layout()
	# plt.show()
	f.savefig("./figs/contained-rouge.png")

	f, ax = plt.subplots(1,1)
	sns.lineplot(x='Fraction contradiction', y='ROUGE-1', data=df, ci = 'sd', ax=ax, label='ROUGE-1')
	sns.lineplot(x='Fraction contradiction', y='ROUGE-2', data=df, ci = 'sd', ax=ax, label='ROUGE-2')
	sns.lineplot(x='Fraction contradiction', y='ROUGE-L', data=df, ci = 'sd', ax=ax, label='ROUGE-L')
	plt.xlabel("Fraction of Contradictory Dependencies")
	plt.ylabel("ROUGE score")
	plt.title("ROUGE score vs. Contradictory Dependencies")
	plt.legend()
	plt.tight_layout()
	# plt.show()
	f.savefig("./figs/contradiction-rouge.png")

	# sns.distplot(contained, label="contained")
	# sns.distplot(missing, label="missing")
	# sns.distplot(contradiction, label="contradiction")
	# plt.legend()
	# plt.title("Score distribution")
	# plt.show()

	# sns.distplot(avg_lengths)
	# plt.title("Average copy length")
	# plt.show()


	# plt.scatter(avg_lengths, contained, marker='.', label="contained")
	# # plt.scatter(avg_lengths, missing, label="missing")
	# plt.scatter(avg_lengths, contradiction, marker='.', label="contradiction")
	# plt.xlabel("Average copy length")
	# plt.ylabel("Number of dependencies")
	# plt.legend()
	# plt.show()
