
import nltk
import nltk.text
import nltk.corpus

from nltk.tokenize import word_tokenize
nltk.download('punkt')

import gensim
print(dir(gensim))




if __name__ == '__main__':

	raw_documents = [	"Саша 12 лет вышел на улицу",
						"Паша 12 лет вышел на улицу",
						"Александр 12 лет вышел гулять",
						"Александр 18 лет вшел гулять",
						"Паша 18 лет встретил Александра 18 лет на улице",
						"Паша и Александр решили вместе гулть",
						"Павел думал выйти глять",
						"Паша и Саша вышли на улицу",
						"Паш 12 лет – Саше 12 лет",
						"Саша  - Паша играют в хоккей возле дома"]

	print("Number of documents: " + str(len(raw_documents)))
	
	gen_docs = [[w.lower() for w in word_tokenize(text)] 
			for text in raw_documents]

	print(gen_docs)



	# dictionary = gensim.corpora.Dictionary(gen_docs)
	# print(dictionary[5])
	# print(dictionary.token2id['лет'])
	# print("Number of words in dictionary:",len(dictionary))
	# for i in range(len(dictionary)):
	# 	print(i, dictionary[i])

	#corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
	#print(corpus)

	#gen_docs = nltk.FreqDist(gen_docs)
	#words = nltk.tokenize.word_tokenize(raw_documents[0])
	#fdist = nltk.FreqDist(words)
	#print(fdist.most_common(4))

	result_list = []

	for doc in gen_docs:
		fdist = nltk.FreqDist(doc)
		print(fdist.most_common(10))
		result_list.append(fdist.most_common(10))

	print(result_list)



	# query_doc_tf_idf = tf_idf[query_doc_bow]
	# print(query_doc_tf_idf)


	# tf_idf = gensim.models.TfidfModel(corpus)
	# print(tf_idf)
	# s = 0
	# for i in corpus:
	# 	s += len(i)
	# print(s)

	# sims = gensim.similarities.Similarity('~/upwork/similar/',tf_idf[corpus], num_features=len(dictionary))
	# print(sims)
	# print(type(sims))