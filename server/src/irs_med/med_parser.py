import re
import csv


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


def process_queries(self, document):
    lines = document.split('\n')
    lines = list(map(lambda x: x.strip(), lines))


def med_parse_relevances(path):
    with open(path, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        relevances = []
        for row in spamreader:
            relevances.append(row)
    return relevances
