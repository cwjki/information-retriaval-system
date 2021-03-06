import re
import numpy as np
from operator import itemgetter
from gensim import corpora, models, similarities
from ..preprocess_document import preproccess_document


class IRSystem():
    def __init__(self, corpus, model_id) -> None:
        self.corpus = corpus
        self.query_weight = []
        self.ranking_query = []
        self.model_id = model_id
        self.model, self.index, self.dictionary = self.compute_model_and_matrix(
            corpus)

    def compute_model_and_matrix(self, corpus):
        model, dictionary = self.create_document_vector(corpus)
        loaded_corpus = corpora.MmCorpus('src/data/vsm_docs.mm')
        index = similarities.MatrixSimilarity(
            loaded_corpus, num_features=len(dictionary))
        return model, index, dictionary

    def create_dictionary(self, documents):
        pdocs = [preproccess_document(doc) for doc in documents]
        dictionary = corpora.Dictionary(pdocs)
        dictionary.save('src/data/vsm.dict')
        return dictionary, pdocs

    def docs_to_bows(self, dictionary: corpora.Dictionary, pdocs):
        vectors = [dictionary.doc2bow(doc) for doc in pdocs]
        corpora.MmCorpus.serialize('src/data/vsm_docs.mm', vectors)
        return vectors

    def ranking_function(self, corpus, query, query_id, ranking_count):
        vquery = self.create_query_vector(query, self.dictionary)
        if self.model_id == 1:
            self.query_weight = [(w[0], 1 + np.log2(w[1])) for w in vquery]
        else:
            self.query_weight = self.model[vquery]

        sim = self.index[self.query_weight]
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        self.ranking_query[query_id] = ranking

        result = []
        for doc, score in ranking[:ranking_count]:
            if score > 0.125:
                result.append((score, corpus[doc], doc))
        return result

    def create_query_vector(self, query, dictionary: corpora.Dictionary):
        pquery = preproccess_document(query)
        vquery = dictionary.doc2bow(pquery)
        return vquery

    def create_document_vector(self, corpus):
        dictionary, pdocs = self.create_dictionary(corpus)
        vdocs = self.docs_to_bows(dictionary, pdocs)
        loaded_corpus = corpora.MmCorpus('src/data/vsm_docs.mm')

        if self.model_id == 1:      # TF model
            model = [[(w[0], 1 + np.log2(w[1])) for w in v]
                     for v in vdocs]
        elif self.model_id == 2:    # TF IDF MODEL
            model = models.TfidfModel(loaded_corpus)
        elif self.model_id == 3:    # LogEntropyModel model
            model = models.LogEntropyModel(
                loaded_corpus)

        return model, dictionary

    def initialize_model(self, corpus, queries, ranking_count):
        query_id = 0
        if isinstance(queries, list):
            for query in queries:
                print(query)
                self.ranking_function(
                    corpus, query, query_id, ranking_count)
                query_id += 1
        else:
            return self.ranking_function(corpus, queries, 1, ranking_count)

    def get_ranking_index(self, query: str, ranking_count=20):
        ranking = self.compute_ranking(query, ranking_count)
        indexes = [index for _, _, index in ranking]
        return indexes

    def compute_ranking(self, query, ranking_count=20):
        ranking = self.initialize_model(self.corpus, query, ranking_count)
        return ranking


class IR_TF(IRSystem):
    def __init__(self, corpus, queries=None) -> None:
        super().__init__(corpus, queries)
        self.ranking_query = dict()
        self.model_id = 1


class IR_TF_IDF(IRSystem):
    def __init__(self, corpus, queries=None) -> None:
        super().__init__(corpus, queries)
        self.ranking_query = dict()
        self.model_id = 2


class IR_LEM(IRSystem):
    def __init__(self, corpus, queries=None) -> None:
        super().__init__(corpus, queries)
        self.ranking_query = dict()
        self.model_id = 3


class IR_Boolean():
    def __init__(self, corpus, queries=None) -> None:
        self.corpus = corpus
        self.queries = queries
        self.ranking_query = dict()
        self.pdocs = self.compute_document_vector(corpus)

    def get_ranking_index(self, query: str, ranking_count=20):
        ranking = self.compute_ranking(query, ranking_count)
        indexes = [index for _, index, _ in ranking]
        return indexes

    def compute_ranking(self, queries, count=20):
        query_id = 0
        if isinstance(queries, list):
            for query in queries:
                or_set, and_set = self.preprocess_query(query)
                dict_matches = self.preprocess_operators(
                    self.corpus, or_set, and_set, query_id)
                query_id += 1
        else:
            or_set, and_set = self.preprocess_query(queries)
            dict_matches = self.preprocess_operators(
                self.corpus, or_set, and_set, 1)

        return self._compute_ranking(dict_matches, count)

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
        pdocs = self.pdocs
        vquery = preproccess_document(query)
        dict_matches = dict((doc, 0) for doc in corpus)
        doc_number = 0
        for doc in pdocs:
            intersection_list = list(set(doc) & set(vquery))
            if len(intersection_list) == len(vquery):
                dict_matches[corpus[doc_number]] = 1
            doc_number += 1
        return dict_matches

    def print_result(self, dict_matches: dict):
        for keys, values in dict_matches.items():
            print("[ Score = " + str(values) + "] ")
            print("Document = " + keys)

    def _compute_ranking(self, dict_matches: dict, count: int):
        ranking = []
        doc_number = 1
        for key, value in dict_matches.items():
            doc_number += 1
            if count == 0:
                break
            if value == 1:
                ranking.append((value, key, doc_number))
                count -= 1
        return ranking

    def compute_document_vector(self, corpus):
        _, pdocs = self.preprocess_corpus(corpus)
        return pdocs

    def preprocess_corpus(self, corpus):
        dictionary, pdocs = self.create_dictionary(corpus)
        return dictionary, pdocs

    def create_dictionary(self, documents):
        pdocs = [preproccess_document(doc) for doc in documents]
        dictionary = corpora.Dictionary(pdocs)
        dictionary.save('src/data/vsm.dict')
        return dictionary, pdocs
