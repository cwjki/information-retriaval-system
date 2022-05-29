from typing import Counter, List
from src.preprocess_document import preprocces_document
from src.inverted_index import InvertedIndex
from src.dataset import Dataset
from math import log10, sqrt

ALPHA = 0.5
RANKING_COUNT = 5


class VectorSpaceModel:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.index = InvertedIndex(dataset)
        self.document_vectors = [self.generate_document_vector(
            index) for index in range(len(self.dataset))]

    def compute_tf(self, document_index: int, term: str) -> float:
        term_frequency = self.index.get_term_frequency_document(
            document_index, term)
        max_frequency = self.index.get_max_term_frequency_document(
            document_index)
        return term_frequency / max_frequency

    def compute_idf(self, term: str) -> float:
        total_documents = self.index.get_total_documents()
        document_frequency = self.index.get_amount_document_with_term(term)
        return log10(total_documents / document_frequency)

    def compute_wight(self, document_index: int, term: str) -> float:
        tf = self.compute_tf(document_index, term)
        idf = self.compute_idf(term)
        return tf * idf

    def compute_query_weight(self, term: str, frequency: float, max_frequency: float) -> float:
        tf = (ALPHA + (1 - ALPHA) * frequency) / max_frequency
        idf = self.compute_idf(term)
        return tf * idf

    def generate_document_vector(self, document_index: int):
        document = self.dataset[document_index]
        terms = preprocces_document(document.text)
        return {term: self.compute_wight(document_index, term) for term in terms}

    def generate_query_vector(self, query: str):
        terms = preprocces_document(query)
        terms = self.index.remove_non_indexed_terms(terms)
        query_frequency = Counter(terms)
        max_frequecy = max(query_frequency.values())
        result = {term: self.compute_query_weight(
            term, query_frequency[term], max_frequecy) for term in terms}
        return result

    def compute_similarity(self, document_vector, query_vector) -> float:
        dot_product = self.compute_dot_product(query_vector, document_vector)
        # print("DOT PRODUCT")
        # print(dot_product)
        document_norma = self.compute_norma_vector(document_vector)
        # print("DOCUMENT NORMA")
        # print(document_norma)
        query_norma = self.compute_norma_vector(query_vector)
        # print("QUERY NORMA")
        # print(query_norma)
        try:
            return dot_product / (document_norma * query_norma)
        except:
            return 0

    def compute_ranking(self, query: str, ranking_count=RANKING_COUNT):
        scores: List[float, int] = []
        query_vector = self.generate_query_vector(query)
        for index in range(len(self.dataset)):
            document_vector = self.document_vectors[index]
            score = self.compute_similarity(document_vector, query_vector)
            scores.append((score, index))

        sorted_scores = sorted(scores, reverse=True)
        # print(sorted_scores)
        index_ranking = []
        for i in range(ranking_count):
            index_ranking.append(sorted_scores[i][1])

        # print(index_ranking)
        ranking = []
        for index in index_ranking:
            ranking.append((index, self.dataset[index]))

        return ranking

    def get_ranking_index(self, query: str, ranking_count=RANKING_COUNT):
        ranking = self.compute_ranking(query, ranking_count)
        indexes = [index for index, _ in ranking]
        return indexes

    def compute_dot_product(self, query_vector, document_vector):
        result = 0.0
        for term, weight1 in query_vector.items():
            try:
                weight2 = document_vector[term]
                result += weight1 * weight2
            except KeyError:
                continue
        return result

    def compute_norma_vector(self, vector):
        result = 0.0
        for term, weight in vector.items():
            result += (weight ** 2)
        return sqrt(result)
