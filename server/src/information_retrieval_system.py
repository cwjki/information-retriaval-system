import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from gensim import corpora, models, similarities


class IRSystem():
    def __init__(self, corpus, queries) -> None:
        self.corpus = corpus
        self.queries = queries

    def preprocess_document(self, document):
        stopset = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = wordpunct_tokenize(document)
        clean = [token.lower() for token in tokens if token.lower()
                 not in stopset and len(token) > 2]
        final = [stemmer.stem(word) for word in clean]
        return final

    def create_dictionary(self, documents):
        pdocs = [self.preprocess_document(doc) for doc in documents]
        dictionary = corpora.Dictionary(pdocs)
        dictionary.save('vsm.dict')
        return dictionary, pdocs

    def docs_to_bows(self, dictionary: corpora.Dictionary, pdocs):
        vectors = [dictionary.doc2bow(doc) for doc in pdocs]
        corpora.MmCorpus.serialize('vsm_docs.mm', vectors)
        return vectors

    def ranking_function(self, corpus, query, query_id, mode):
        pass

    def create_query_vector(self, query, dictionary: corpora.Dictionary):
        pquery = self.preprocess_document(query)
        vquery = dictionary.doc2bow(pquery)
        return vquery

    def create_document_vector(self, corpus, model_id):
        dictionary, pdocs = self.create_dictionary(corpus)
        vdocs = self.docs_to_bows(dictionary, pdocs)
        loaded_corpus = corpora.MmCorpus('vsm_docs.mm')

        model = models.TfidfModel(loaded_corpus)

        return vdocs, model

    def initialize_model(self, corpus, queries, mode):
        query_id = 0
        if isinstance(queries, list):
            for query in queries:
                self.ranking_function(corpus, query, query_id, mode)
                query_id += 1
        else:
            self.ranking_function(corpus, queries, 1, mode)


class IR_TF_IDF(IRSystem):
    def __init__(self, corpus, queries) -> None:
        super().__init__(corpus, queries)
        self.ranking_query = dict()
        self.initialize_model(corpus, queries, 1)


class IR_Boolean(IRSystem):
    def __init__(self, corpus, queries) -> None:
        super().__init__(corpus, queries)
        self.ranking_query = dict()

        query_id = 0
        if isinstance(queries, list):
            for query in queries:
                or_set, and_set = self.preprocess_query(query)
                dict_matches = self.preprocess_operators(
                    corpus, or_set, and_set, query_id)
                query_id += 1
        else:
            or_set, and_set = self.preprocess_query(query)
            dict_matches = self.preprocess_operators(
                corpus, or_set, and_set, 1)

    def preprocess_query(self, query):
        text = re.split(r'[^\w\s]', query)
        or_set = []
        and_set = []
        for phrase in text:
            txt = re.split('or', phrase)
            if len(txt) > 1:
                or_set.append(txt)
            else:
                and_set.append(txt)
        return or_set, and_set
