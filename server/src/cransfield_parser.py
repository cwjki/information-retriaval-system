import re



class CransfieldParser:

    def parse(file):
        with open(file, 'r') as f:
            data = ''.join(f.readlines())
        documents = re.split('.I \d', data)
        documents = [document.strip() for document in documents]
        documents = [document for document in documents if not document == '']
    
    def __parse_document(document):
        lines = document.split('\n')
        lines = list(map(lambda x: x.strip(), lines))
        separators = []
        for separator in ['.T', '.A', '.B', '.W', '.X']:
            try:
                index = lines.index(separator)
                separators.append((index, separator))
            except ValueError:
                pass

    def __create_document(lines, separators):
        
        
