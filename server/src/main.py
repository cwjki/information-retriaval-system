from pathlib import Path
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from vsm_manual.metrics_evaluator import Evaluator
from vsm_manual.vector_space_model import VectorSpaceModel
from vsm_manual.cranfield_parser import CranfieldParser

path = Path(__file__).parent

CRAN_COLLECTION = str(path) + "/collections/cranfield_collection/cran.all.1400"
CRAN_QUERIE = str(path) + '/collections/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/collections/cranfield_collection/cranqrel'


app = Flask(__name__)
Bootstrap(app)


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        ranking = vector_space_model.compute_ranking(query)
        count = len(ranking)
        return render_template('results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/evaluate", methods=['GET'])
def evaluate():
    evaluator = Evaluator(documents, queries, relations, vector_space_model)
    metrics = evaluator.evaluate()
    return render_template('evaluate.html', content=[metrics])


if __name__ == "__main__":
    print(path)
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    queries = cranfieldParser.parse(CRAN_QUERIE)
    relations = cranfieldParser.parse_cranqrel(CRAN_QREL)
    vector_space_model = VectorSpaceModel(documents)

    app.run(debug=True)
