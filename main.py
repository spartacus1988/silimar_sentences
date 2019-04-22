
import nltk
import nltk.text
import nltk.corpus

from nltk.tokenize import word_tokenize




if __name__ == '__main__':
	# pass
	# idx = nltk.text.ContextIndex([word.lower( ) for word in nltk.corpus.brown.words( )])
	# save = [ ]
	# for word in nltk.word_tokenize("i want to solve this problem"):
	# 	save.append(idx.similar_words(word))

	# for word in save:
	# 	print(word)

	

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
	
	docs = []

	for text in raw_documents:
		print(text)
		#for w in word_tokenize(text):
		#docs = [w.lower() for w in word_tokenize(text)]
			#docs.append(list(w.lower() for w in word_tokenize(text)))

	#print(gen_docs)