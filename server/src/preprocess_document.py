from nltk.tokenize import word_tokenize
from typing import List


def preprocces_document(document: str) -> List[str]:
    terms = word_tokenize(document, language='english')
    return terms



