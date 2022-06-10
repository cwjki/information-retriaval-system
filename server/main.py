from crypt import methods
from pathlib import Path
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from src.vsm_manual.metrics_evaluator import Evaluator
from src.vsm_manual.vector_space_model import VectorSpaceModel
from src.vsm_manual.cranfield_parser import CranfieldParser
from src.utils import save_model, load_model, med_parse
from src.ir_system.models import IR_Boolean, IR_TF_IDF


path = Path(__file__).parent

MED_COLLECTION = str(path) + '/src/collections/med_collection/MED.ALL'
MED_QUERY = str(path) + '/src/collections/med_collection/MED.QRY'
MED_REL = str(path) + '/src/collections/med_collection/MED.REL'

CRAN_COLLECTION = str(
    path) + '/src/collections/cranfield_collection/cran.all.1400'
CRAN_QUERY = str(path) + '/src/collections/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/src/collections/cranfield_collection/cranqrel'

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


@app.route("/boolean", methods=['GET', 'POST'])
def boolean():
    if request.method == 'POST':
        query = request.form['query']
        ranking = boolean_model.compute_ranking(query)
        count = len(ranking)
        return render_template('boolean_results.html', content=[query, ranking, count])

    else:
        return render_template('index.html')


@app.route("/tf-idf", methods=['GET', 'POST'])
def tf_idf():
    if request.method == 'POST':
        query = request.form['query']
        ranking = tf_idf_model.compute_ranking(query)
        count = len(ranking)
        print(ranking) 
        return render_template('boolean_results.html', content=[query, ranking, count])

    else:
        return render_template('index.html')


if __name__ == "__main__":

    # CRANFIELD Collection
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    queries = cranfieldParser.parse(CRAN_QUERY)
    relations = cranfieldParser.parse_cranqrel(CRAN_QREL)

    # MED Collection
    corpus_med = med_parse(MED_COLLECTION)

    # VECTOR SPACE MODEL with CRANFIELD collection
    try:
        vector_space_model = load_model(VSM_DIR)
    except OSError:
        vector_space_model = VectorSpaceModel(documents)
        save_model(vector_space_model, VSM_DIR)

    # BOOLEAN MODEL
    boolean_model = IR_Boolean(corpus_med)
    tf_idf_model = IR_TF_IDF(corpus_med)

    app.run(debug=True)
