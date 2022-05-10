from pathlib import Path
from flask import Flask
from src.cranfield_parser import CranfieldParser

path = Path(__file__).parent


CRAN_COLLECTION = str(path) + "/cranfield_collection/cran.all.1400"
CRAN_QUERIE = str(path) + '/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/cranfield_collection/cranqrel'

app = Flask(__name__)


@app.route("/")
def home():
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    queries = cranfieldParser.parse(CRAN_QUERIE)
    return documents


def main():
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    print(documents)
    queries = cranfieldParser.parse(CRAN_QUERIE)


if __name__ == "__main__":
    app.run()
    main()
