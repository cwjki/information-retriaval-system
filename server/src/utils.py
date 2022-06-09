import pickle
import re


def med_parse(path):
    try:
        list_texts = re.split('.I \d*\n.W\n', open(path).read())[1:]
        return list_texts
    except IOError:
        print("No such file or directory - MED COLLECTION")


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))
