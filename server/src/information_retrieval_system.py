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
                self.print_result(dict_matches)
        else:
            or_set, and_set = self.preprocess_query(query)
            dict_matches = self.preprocess_operators(
                corpus, or_set, and_set, 1)
            self.print_result(dict_matches)

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

    def preprocess_operators(self, corpus, or_set, and_set, query_id):
        or_list = [value for sublist in or_set for value in sublist]
        for or_text in or_list:
            dict_matches = self.document_matches(corpus, or_text)
        if len(and_set) > 0:
            and_list = [value for sublist in and_set for value in sublist]
            and_txt = ', '.join(and_list)
            dict_matches = self.document_matches(corpus, and_txt)
        self.ranking_query[query_id] = dict_matches.items()
        return dict_matches

    def document_matches(self, corpus, query):
        dictionary, pdocs = self.preprocess_corpus(corpus)
        vquery = self.preprocess_document(query)
        dict_matches = dict((doc,0) for doc in corpus)
        doc_number = 0
        for doc in pdocs:
            intersection_list = list(set(doc) & set(vquery))
            if len(intersection_list) == len(vquery):
                dict_matches[corpus[doc_number]] = 1
            doc_number += 1
        return dict_matches


    def preprocess_corpus(self, corpus):
        dictionary, pdocs = self.create_dictionary(corpus)
        return dictionary, pdocs

    def print_result(self, dict_matches: dict):
        for keys, values in dict_matches.items():
            print("[ Score = " + str(values) + "] ")
            print("Document = " + keys)
