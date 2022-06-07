from typing import List
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize



class IRSystem():
    def __init__(self, corpus, queries) -> None:
        self.corpus = corpus
        self.queries = queries

    def preprocess_document(self, document):
        stopset = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = wordpunct_tokenize(document)
        clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
        final = [stemmer.stem(word) for word in clean]
        return final

    
    def create_dictionary(self, documents):
        preprocess_documents = [self.preprocess_document(document) for document in documents]
        dictionary = 