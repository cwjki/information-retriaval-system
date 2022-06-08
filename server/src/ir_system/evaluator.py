

from sys import flags


class IREvaluator():
    def __init__(self, relevance_docs, ranking_query, continue_eval, only_query_id) -> None:
        self.relevance_docs = relevance_docs
        self.continue_eval = continue_eval

        query_id = 1
        if len(ranking_query) > 1:
            for query in ranking_query-1:
                q_relevants_docs = self.get_total_relevant_docs(query_id)
                ranking_query[query] = [ranking_query[query][i] for i in range(
                    len(ranking_query[query])) if ranking_query[query][i][1] > 0.0]
                self.evaluate_query(
                    ranking_query[query], q_relevants_docs, query_id)
                query_id += 1
        else:
            q_relevants_docs = self.get_total_relevant_docs(only_query_id)
            ranking_query[1] = [ranking_query[1][i] for i in range(
                len(ranking_query[1])) if ranking_query[1][i][1] > 0.0]
            self.evaluate_query(
                ranking_query[1], q_relevants_docs, only_query_id)

    def get_total_relevant_docs(self, query_id):
        relevance_query = dict()
        q_relevance_docs = []
        for doc in self.relevance_docs:
            if doc[0] == query_id:
                q_relevance_docs.append(doc[2])
            relevance_query[query_id] = q_relevance_docs
            return relevance_query

    def evaluate_query(self, ranking, q_relevance_docs, query_id):
        if self.continue_eval:
            [rr, ri] = self.relevant_doc_retrieved(
                query_id, ranking, q_relevance_docs)

            precission = self.compute_precission(rr, ri)

    def relevant_doc_retrieved(self, query, ranking, q_relevant_docs):
        rr = 0
        ri = 0
        for doc in ranking:
            if str(doc[0]) in q_relevant_docs[query]:
                rr += 1
            else:
                ri += 1
        return rr, ri

    def compute_precission(self, rr, ri):
        retrieved = rr + ri
        precission = float(rr) / float(retrieved)
        return precission

    def compute_recall(self, rr, rrr):
        recall = float(rr) / float(rrr)
        return recall