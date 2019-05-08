import numpy as np
import sys


if __name__ == "__main__":
	names_file = sys.argv[1]

	f = open(names_file, "rb")
	names = f.read()
	names_list = names.split('\n')
	f.close()

	pg_articles_path = "../test_output/articles.txt"
	pg_articles_file = open(pg_articles_path, "rb")
	pg_articles = pg_articles_file.read()
	pg_articles_list = pg_articles.split('\n')

	pg_references_path = "../../pointer-generator/articles.txt"
	pg_references_file = open(pg_references_path, "rb")
	pg_references = pg_references_file.read()
	pg_references_list = pg_references.split('\n')

	all_reference = open("./reference.txt", "a+")
	all_articles = open("./articles.txt", "a+")
	all_decoded = open("./decoded.txt", "a+")


	for ixn,name in enumerate(names_list):
		print(ixn, "out of", len(names_list))
		if(len(name) > 0):
			num = name.split('.')[0]

			file = open("./rnn-ext_abs_rl_rerank/decoded/"+str(num)+".dec", "rb")
			lines = file.read()
			joined_lines = " ".join(lines.split('\n'))+'\n'
			file.close()
			all_decoded.write(joined_lines)
			# print(joined_lines)
			# print("=========")

			file = open("./reference/"+str(num)+".ref", "rb")
			lines = file.read()
			joined_lines = " ".join(lines.split('\n'))+'\n'
			file.close()
			all_reference.write(joined_lines)
			# print(joined_lines)
			# print("=========")

			for i,ref in enumerate(pg_references_list):
				if(ref[:100] == joined_lines[:100]):
					break
			if(i == len(pg_references_list)):
				print("********* ERROR **********")
				break
			all_articles.write(pg_articles_list[i]+'\n')
			# print(pg_articles_list[i])
			# print("=========")
			# print("=========")

	all_reference.close()
	all_articles.close()
	all_decoded.close()