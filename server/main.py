from pathlib import Path
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from src.vsm_manual.metrics_evaluator import Evaluator
from src.vsm_manual.vector_space_model import VectorSpaceModel
from src.vsm_manual.cranfield_parser import CranfieldParser
from src.utils import save_model, load_model

path = Path(__file__).parent

CRAN_QUERIE = str(path) + '/src/collections/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/src/collections/cranfield_collection/cranqrel'
CRAN_COLLECTION = str(
    path) + "/src/collections/cranfield_collection/cran.all.1400"

VSM_DIR = str(path) + '/src/data/vsm.vsm'


app = Flask(__name__)
Bootstrap(app)


@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/vsm", methods=['GET', 'POST'])
def vsm():
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

    # VECTOR SPACE MODEL with Cranfield Collection
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    queries = cranfieldParser.parse(CRAN_QUERIE)
    relations = cranfieldParser.parse_cranqrel(CRAN_QREL)

    try:
        vector_space_model = load_model(VSM_DIR)
    except OSError:
        vector_space_model = VectorSpaceModel(documents)
        save_model(vector_space_model, VSM_DIR)

    app.run(debug=True)
