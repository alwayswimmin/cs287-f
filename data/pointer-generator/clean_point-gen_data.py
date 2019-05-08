import numpy as np
import sys


if __name__ == "__main__":
	names_file = sys.argv[1]

	f = open(names_file, "rb")
	names = f.read()
	names_list = names.split('\n')
	f.close()

	all_reference = open("../reference.txt", "a+")
	all_articles = open("../articles.txt", "a+")
	all_decoded = open("../decoded.txt", "a+")
	for ixn,name in enumerate(names_list):
		print(ixn, "out of", len(names_list))
		if(len(name) > 0):
			num = name.split('_')[0]

			file = open("./reference/"+str(num)+"_reference.txt", "rb")
			lines = file.read()
			joined_lines = " ".join(lines.split('\n'))+'\n'
			file.close()
			all_reference.write(joined_lines)

			file = open("./articles/"+str(num)+"_article.txt", "rb")
			lines = file.read()
			joined_lines = " ".join(lines.split('\n'))+'\n'
			file.close()
			all_articles.write(joined_lines)
			
			file = open("./pointer-gen-cov/"+str(num)+"_decoded.txt", "rb")
			lines = file.read()
			joined_lines = " ".join(lines.split('\n'))+'\n'
			file.close()
			all_decoded.write(joined_lines)

	all_reference.close()
	all_articles.close()
	all_decoded.close()
