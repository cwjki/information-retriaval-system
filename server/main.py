from pathlib import Path
from flask import Flask, request, jsonify, render_template
from src.vector_space_model import VectorSpaceModel
from src.cranfield_parser import CranfieldParser

path = Path(__file__).parent


CRAN_COLLECTION = str(path) + "/cranfield_collection/cran.all.1400"
CRAN_QUERIE = str(path) + '/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/cranfield_collection/cranqrel'


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/query", methods=['POST'])
def query():
    query = request.json['data']
    # ranking = vector_space_model.compute_ranking(query)

    return jsonify(ranking)


if __name__ == "__main__":
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    queries = cranfieldParser.parse(CRAN_QUERIE)

    # vector_space_model = VectorSpaceModel(documents)

    app.run(debug=True)
