
import nltk
import nltk.text
import nltk.corpus

from nltk.tokenize import word_tokenize
nltk.download('punkt')




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
	
	docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]

	print(docs)