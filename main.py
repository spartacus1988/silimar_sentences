
import nltk
import nltk.text
import nltk.corpus

from nltk.tokenize import word_tokenize
nltk.download('punkt')

import gensim
print(dir(gensim))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy




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



	# vectorization of the raw_documents
	vectorizer = TfidfVectorizer(stop_words=None)
	X = vectorizer.fit_transform(raw_documents)

	words = vectorizer.get_feature_names()
	print("words", words)


	n_clusters=3
	number_of_seeds_to_try=10
	max_iter = 300
	number_of_process=2 # seads are distributed
	model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=number_of_seeds_to_try, n_jobs=number_of_process).fit(X)

	labels = model.labels_
	# indices of preferible words in each cluster
	ordered_words = model.cluster_centers_.argsort()[:, ::-1]


	print("centers:", model.cluster_centers_)
	print("labels", labels)
	print("intertia:", model.inertia_)


	texts_per_cluster = numpy.zeros(n_clusters)
	for i_cluster in range(n_clusters):
		for label in labels:
			if label==i_cluster:
				texts_per_cluster[i_cluster] +=1 

	print("Top words per cluster:")
	for i_cluster in range(n_clusters):
		print("Cluster:", i_cluster, "texts:", int(texts_per_cluster[i_cluster])),
		for term in ordered_words[i_cluster, :10]:
			print("\t"+words[term])

	print("\n")
	print("Prediction")


	text_to_predict = "Саша 12 лет вышел на улицу"
	Y = vectorizer.transform([text_to_predict])
	predicted_cluster = model.predict(Y)[0]
	texts_per_cluster[predicted_cluster]+=1

	print(text_to_predict)
	print("Cluster:", predicted_cluster, "texts:", int(texts_per_cluster[predicted_cluster])),
	for term in ordered_words[predicted_cluster, :10]:
		print("\t"+words[term])



	# print("Number of documents: " + str(len(raw_documents)))
	
	# gen_docs = [[w.lower() for w in word_tokenize(text)] 
	# 		for text in raw_documents]

	# print(gen_docs)



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

	# result_list = []

	# for doc in gen_docs:
	# 	fdist = nltk.FreqDist(doc)
	# 	print(fdist.most_common(10))
	# 	#result_list.append(fdist.most_common(10))
	# 	result_list.append(fdist)

	# print(result_list)



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