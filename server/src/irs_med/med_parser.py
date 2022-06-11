import re
from collections import defaultdict


def med_parse_collection(path):
    try:
        list_texts = re.split('.I \d*\n.W\n', open(path).read())[1:]
        return list_texts
    except IOError:
        print("No such file or directory - MED COLLECTION")


def med_parse_queries(path):
    try:
        list_texts = re.split('.I \d*\n.W\n', open(path).read())[1:]
        list_texts = [text.strip() for text in list_texts]
        return list_texts
    except IOError:
        print("No such file or directory - MED COLLECTION")


def med_parse_rel(file):
    relations = defaultdict(lambda: list())
    with open(file, 'r') as f:
        for line in f.readlines():
            query, _, document, _ = map(lambda x: int(float(x)), line.split())
            relations[query-1].append(document-1)
    return relations
