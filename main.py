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


	n_clusters=5
	number_of_seeds_to_try=10
	max_iter = 300
	number_of_process=2 # seads are distributed
	model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=number_of_seeds_to_try, n_jobs=number_of_process).fit(X)
	print(model)


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



