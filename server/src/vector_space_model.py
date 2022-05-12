from typing import Counter, List
from src.preprocess_document import preprocces_document
from src.inverted_index import InvertedIndex
from src.dataset import Dataset
from math import log10

ALPHA = 0.4
RANKING_COUNT = 20


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
        query_frequency = Counter(terms)
        max_frequecy = max(query_frequency.values())
        result = {term: self.compute_query_weight(
            term, query_frequency[term], max_frequecy) for term in terms}
        return result

    def compute_similarity(self, document_vector, query_vector) -> float:
        return 0.9

    def compute_ranking(self, query: str):
        scores: List[float, int] = []
        query_vector = self.generate_query_vector(query)
        for index in range(len(self.dataset)):
            document_vector = self.document_vectors[index]
            score = self.compute_similarity(document_vector, query_vector)
            scores.append((score, index))

        sorted_scores = sorted(scores, reverse=True)
        index_ranking = []
        for i in range(RANKING_COUNT):
            index_ranking.append(sorted_scores[i][1])
        
        ranking = []
        for i in range(RANKING_COUNT):
            ranking.append((i, self.dataset[i]))

        return ranking
