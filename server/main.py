from pathlib import Path
from traceback import print_tb
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from src.vsm_cranfield.metrics_evaluator import Evaluator
from src.vsm_cranfield.vector_space_model import VectorSpaceModel
from src.vsm_cranfield.cranfield_parser import CranfieldParser
from src.utils import save_model, load_model
from src.irs_med.models import IR_LDA, IR_TF, IR_Boolean, IR_TF_IDF
from src.irs_med.med_parser import med_parse_collection, med_parse_rel
from src.irs_med.evaluator import IREvaluator


path = Path(__file__).parent

MED_COLLECTION = str(path) + '/src/collections/med_collection/MED.ALL'
MED_QUERY = str(path) + '/src/collections/med_collection/MED.QRY'
MED_REL = str(path) + '/src/collections/med_collection/MED.REL'

CRAN_COLLECTION = str(
    path) + '/src/collections/cranfield_collection/cran.all.1400'
CRAN_QUERY = str(path) + '/src/collections/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/src/collections/cranfield_collection/cranqrel'

VSM_DIR = str(path) + '/src/data/vsm.vsm'

VSM_METRICS = str(path) + '/src/data/vsm.metrics'
BOOLEAN_METRICS = str(path) + '/src/data/boolean.metrics'
TF_METRICS = str(path) + '/src/data/tf.metrics'
TF_IDF_METRICS = str(path) + '/src/data/tf_idf.metrics'
LDA_METRICS = str(path) + '/src/data/lda.metrics'


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
    return render_template('evaluate.html',
                           content=[vsm_metrics,
                                    boolean_metrics,
                                    tf_metrics,
                                    tf_idf_metrics,
                                    lda_metrics])


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
        return render_template('tf_idf_results.html', content=[query, ranking, count])

    else:
        return render_template('index.html')


if __name__ == "__main__":

    # CRANFIELD Collection
    cranfieldParser = CranfieldParser()
    documents = cranfieldParser.parse(CRAN_COLLECTION)
    queries = cranfieldParser.parse(CRAN_QUERY)
    relations = cranfieldParser.parse_cranqrel(CRAN_QREL)

    # MED Collection
    corpus_med = med_parse_collection(MED_COLLECTION)
    queries_med = med_parse_collection(MED_QUERY)
    relations_med = med_parse_rel(MED_REL)

# ---------------------------------------------------------------------------------------
    # VECTOR SPACE MODEL with CRANFIELD collection
    try:
        vector_space_model = load_model(VSM_DIR)
    except OSError:
        vector_space_model = VectorSpaceModel(documents)
        save_model(vector_space_model, VSM_DIR)

    # VECTOR SPACE MODEL EVALUATOR
    vsm_evaluator = Evaluator(
        documents, queries, relations, vector_space_model)
    try:
        vsm_metrics = load_model(VSM_METRICS)
    except OSError:
        vsm_metrics = vsm_evaluator.evaluate()
        save_model(vsm_metrics, VSM_METRICS)

# ---------------------------------------------------------------------------------------
    # BOOLEAN MODEL
    boolean_model = IR_Boolean(corpus_med)

    # BOOLEAN MODEL EVALUATOR
    boolean_evaluator = Evaluator(
        corpus_med, queries_med, relations_med, boolean_model)
    try:
        boolean_metrics = load_model(BOOLEAN_METRICS)
    except OSError:
        boolean_metrics = boolean_evaluator.evaluate()
        save_model(boolean_metrics, BOOLEAN_METRICS)

# ---------------------------------------------------------------------------------------
    # TF MODEL
    tf_model = IR_TF(corpus_med)

    # TF MODEL EVALUATOR
    tf_evaluator = Evaluator(
        corpus_med, queries_med, relations_med, tf_model)
    try:
        tf_metrics = load_model(TF_METRICS)
    except OSError:
        tf_metrics = tf_evaluator.evaluate()
        save_model(tf_metrics, TF_METRICS)

# ---------------------------------------------------------------------------------------
    # TF IDF MODEL
    tf_idf_model = IR_TF_IDF(corpus_med)

    # TF IDF MODEL EVALUATOR
    tf_idf_evaluator = Evaluator(
        corpus_med, queries_med, relations_med, tf_idf_model)
    try:
        tf_idf_metrics = load_model(TF_IDF_METRICS)
    except OSError:
        tf_idf_metrics = tf_idf_evaluator.evaluate()
        save_model(tf_idf_metrics, TF_IDF_METRICS)

# ---------------------------------------------------------------------------------------
    # LDA MODEL
    lda_model = IR_LDA(corpus_med)

    # LDA MODEL EVALUATOR
    lda_evaluator = Evaluator(
        corpus_med, queries_med, relations_med, lda_model)
    try:
        lda_metrics = load_model(LDA_METRICS)
    except OSError:
        lda_metrics = lda_evaluator.evaluate()
        save_model(lda_metrics, LDA_METRICS)

    app.run(debug=True)
