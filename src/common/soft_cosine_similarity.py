import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex
import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_lg')
stop_words = nlp.Defaults.stop_words


class SoftCosineSimilarity:
    def __init__(self, documents=None):
        self.word2vec_model300 = api.load('word2vec-google-news-300')
        # Convert the sentences into bag-of-words vectors.
        self.similarity_index = WordEmbeddingSimilarityIndex(self.word2vec_model300)

    def calculate(self, doc_1, doc_2):
        # Prepare a dictionary and a corpus.
        documents = [doc_1, doc_2]
        dictionary = Dictionary(documents)

        # Prepare the similarity matrix
        # similarity_matrix = SparseTermSimilarityMatrix(self.similarity_index, dictionary)

        # Convert the sentences into bag-of-words vectors.
        doc_1 = dictionary.doc2bow(doc_1)
        doc_2 = dictionary.doc2bow(doc_2)
        return cosine_similarity(doc_1, doc_2)[0][0]

