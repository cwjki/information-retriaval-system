from pathlib import Path
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from src.metrics_evaluator import Evaluator
from src.vector_space_model import VectorSpaceModel
from src.cranfield_parser import CranfieldParser

path = Path(__file__).parent

CRAN_COLLECTION = str(path) + "/cranfield_collection/cran.all.1400"
CRAN_QUERIE = str(path) + '/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/cranfield_collection/cranqrel'


app = Flask(__name__)
Bootstrap(app)


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        ranking = vector_space_model.compute_ranking(query)

        return render_template('results.html', content=[query, ranking])
    else:
        return render_template('index.html')


@app.route("/evaluate", methods=['GET'])
def evaluate():
    evaluator = Evaluator(documents, queries, relations, vector_space_model)
    metrics = evaluator.evaluate()
    return render_template('evaluate.html', content=[metrics])


if __name__ == "__main__":
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    queries = cranfieldParser.parse(CRAN_QUERIE)
    relations = cranfieldParser.parse_cranqrel(CRAN_QREL)
    vector_space_model = VectorSpaceModel(documents)

    app.run(debug=True)
