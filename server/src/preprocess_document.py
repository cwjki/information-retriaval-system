from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
from string import punctuation
from typing import List

POS_TRANSFORM = {
    'V': wordnet.VERB,
    'J': wordnet.ADJ,
    'R': wordnet.ADV
}


def preprocces_document(document: str) -> List[str]:
    # lemmatize terms
    terms = word_tokenize(document, language='english')
    terms_lemmatized = lemmatize(terms)
    # remove stopswords
    remove_term = set(stopwords.words())
    terms_without_stopwords = [
        term for term in terms_lemmatized if not term in remove_term]
    # remove punctuation
    terms = [term for term in terms_without_stopwords if not term in punctuation]
    return terms


def lemmatize(terms: List[str]) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    terms_pos_tag = pos_tag(terms)
    terms_lemmatized = [lemmatizer.lemmatize(term, pos=POS_TRANSFORM.get(
        tag[0], wordnet.NOUN)) for term, tag in terms_pos_tag]
    return terms_lemmatized
