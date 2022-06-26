from pathlib import Path
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from src.metrics_evaluator import Evaluator
from src.vsm_cranfield.vector_space_model import VectorSpaceModel
from src.vsm_cranfield.cranfield_parser import CranfieldParser
from src.utils import save_model, load_model
from src.irs_med.models import IR_LEM, IR_TF, IR_Boolean, IR_TF_IDF
from src.irs_med.med_parser import med_parse_collection, med_parse_rel


path = Path(__file__).parent

MED_COLLECTION = str(path) + '/src/collections/med_collection/MED.ALL'
MED_QUERY = str(path) + '/src/collections/med_collection/MED.QRY'
MED_REL = str(path) + '/src/collections/med_collection/MED.REL'

CRAN_COLLECTION = str(
    path) + '/src/collections/cranfield_collection/cran.all.1400'
CRAN_QUERY = str(path) + '/src/collections/cranfield_collection/cran.qry'
CRAN_QREL = str(path) + '/src/collections/cranfield_collection/cranqrel'

VSM_CRAN_DIR = str(path) + '/src/data/vsm.vsm.cran'
VSM_CRAN_METRICS = str(path) + '/src/data/vsm.metrics.cran'

VSM_MED_DIR = str(path) + '/src/data/vsm.vsm.med'
VSM_MED_METRICS = str(path) + '/src/data/vsm.metrics.med'

BOOLEAN_METRICS_MED = str(path) + '/src/data/boolean.metrics.med'
BOOLEAN_METRICS_CRAN = str(path) + '/src/data/boolean.metrics.cran'

TF_METRICS_MED = str(path) + '/src/data/tf.metrics.med'
TF_METRICS_CRAN = str(path) + '/src/data/tf.metrics.cran'


TF_IDF_METRICS_MED = str(path) + '/src/data/tf_idf.metrics.med'
TF_IDF_METRICS_CRAN = str(path) + '/src/data/tf_idf.metrics.cran'

LEM_METRICS_MED = str(path) + '/src/data/lem.metrics.med'
LEM_METRICS_CRAN = str(path) + '/src/data/lem.metrics.cran'


app = Flask(__name__)
Bootstrap(app)


@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/vsm-cran", methods=['GET', 'POST'])
def vsm_cran():
    if request.method == 'POST':
        query = request.form['query']
        ranking = vector_space_model_cranfield.compute_ranking(query)
        count = len(ranking)
        return render_template('results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/vsm-med", methods=['GET', 'POST'])
def vsm_med():
    if request.method == 'POST':
        query = request.form['query']
        ranking = vector_space_model_med.compute_ranking(query)
        count = len(ranking)
        return render_template('results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/evaluate", methods=['GET'])
def evaluate():
    return render_template('evaluate.html',
                           content=[vsm_cran_metrics,
                                    vsm_med_metrics,
                                    boolean_metrics_cran,
                                    boolean_metrics_med,
                                    tf_metrics_cran,
                                    tf_metrics_med,
                                    tf_idf_metrics_cran,
                                    tf_idf_metrics_med,
                                    lem_metrics])


@app.route("/boolean-cran", methods=['GET', 'POST'])
def boolean_cran():
    if request.method == 'POST':
        query = request.form['query']
        ranking = boolean_model_cran.compute_ranking(query)
        count = len(ranking)
        return render_template('boolean_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/boolean-med", methods=['GET', 'POST'])
def boolean_med():
    if request.method == 'POST':
        query = request.form['query']
        ranking = boolean_model_med.compute_ranking(query)
        count = len(ranking)
        return render_template('boolean_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/tf-cran", methods=['GET', 'POST'])
def tf_cran():
    if request.method == 'POST':
        query = request.form['query']
        ranking = tf_model_cran.compute_ranking(query)
        count = len(ranking)
        return render_template('tf_idf_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/tf-med", methods=['GET', 'POST'])
def tf_med():
    if request.method == 'POST':
        query = request.form['query']
        ranking = tf_model_med.compute_ranking(query)
        count = len(ranking)
        return render_template('tf_idf_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/tf-idf-cran", methods=['GET', 'POST'])
def tf_idf_cran():
    if request.method == 'POST':
        query = request.form['query']
        ranking = tf_idf_model_cran.compute_ranking(query)
        count = len(ranking)
        return render_template('tf_idf_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/tf-idf-med", methods=['GET', 'POST'])
def tf_idf_med():
    if request.method == 'POST':
        query = request.form['query']
        ranking = tf_idf_model_med.compute_ranking(query)
        count = len(ranking)
        return render_template('tf_idf_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/lem-cran", methods=['GET', 'POST'])
def lem_cran():
    if request.method == 'POST':
        query = request.form['query']
        ranking = lem_model_cran.compute_ranking(query)
        count = len(ranking)
        return render_template('tf_idf_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


@app.route("/lem-med", methods=['GET', 'POST'])
def lem_med():
    if request.method == 'POST':
        query = request.form['query']
        ranking = lem_model_med.compute_ranking(query)
        count = len(ranking)
        return render_template('tf_idf_results.html', content=[query, ranking, count])
    else:
        return render_template('index.html')


if __name__ == "__main__":

    # CRANFIELD Collection
    cranfieldParser = CranfieldParser()

    cranfield_documents = cranfieldParser.parse(CRAN_COLLECTION)
    corpus_cranfield = [document.text for document in cranfield_documents]
    cranfield_queries = cranfieldParser.parse(CRAN_QUERY)
    cranfield_relations = cranfieldParser.parse_cranqrel(CRAN_QREL)

    # MED Collection
    vsm_med_documents = cranfieldParser.parse(MED_COLLECTION)
    corpus_med = med_parse_collection(MED_COLLECTION)
    queries_med = med_parse_collection(MED_QUERY)
    relations_med = med_parse_rel(MED_REL)


# ---------------------------------------------------------------------------------------
    # VECTOR SPACE MODEL with CRANFIELD collection
    try:
        vector_space_model_cranfield = load_model(VSM_CRAN_DIR)
    except OSError:
        vector_space_model_cranfield = VectorSpaceModel(cranfield_documents)
        save_model(vector_space_model_cranfield, VSM_CRAN_DIR)

    # VECTOR SPACE MODEL EVALUATOR
    vsm_cran_evaluator = Evaluator(
        cranfield_documents, cranfield_queries, cranfield_relations, vector_space_model_cranfield)
    try:
        vsm_cran_metrics = load_model(VSM_CRAN_METRICS)
    except OSError:
        vsm_cran_metrics = vsm_cran_evaluator.evaluate()
        save_model(vsm_cran_metrics, VSM_CRAN_METRICS)


# ---------------------------------------------------------------------------------------
    # VECTOR SPACE MODEL with MED collection
    try:
        vector_space_model_med = load_model(VSM_MED_DIR)
    except OSError:
        vector_space_model_med = VectorSpaceModel(vsm_med_documents)
        save_model(vector_space_model_med, VSM_MED_DIR)

    # VECTOR SPACE MODEL EVALUATOR
    vsm_med_evaluator = Evaluator(
        vsm_med_documents, queries_med, relations_med, vector_space_model_med)
    try:
        vsm_med_metrics = load_model(VSM_MED_METRICS)
    except OSError:
        vsm_med_metrics = vsm_med_evaluator.evaluate()
        save_model(vsm_med_metrics, VSM_MED_METRICS)

# ---------------------------------------------------------------------------------------
    # BOOLEAN MODEL with CRANFIELD
    boolean_model_cran = IR_Boolean(corpus_cranfield)

    # BOOLEAN MODEL EVALUATOR
    boolean_evaluator_cran = Evaluator(
        corpus_cranfield, cranfield_queries, cranfield_relations, boolean_model_cran)
    try:
        boolean_metrics_cran = load_model(BOOLEAN_METRICS_CRAN)
    except OSError:
        boolean_metrics_cran = boolean_evaluator_cran.evaluate()
        save_model(boolean_metrics_cran, BOOLEAN_METRICS_CRAN)

# ---------------------------------------------------------------------------------------
    # BOOLEAN MODEL with MED
    boolean_model_med = IR_Boolean(corpus_med)

    # BOOLEAN MODEL EVALUATOR
    boolean_evaluator_med = Evaluator(
        corpus_med, queries_med, relations_med, boolean_model_med)
    try:
        boolean_metrics_med = load_model(BOOLEAN_METRICS_MED)
    except OSError:
        boolean_metrics_med = boolean_evaluator_med.evaluate()
        save_model(boolean_metrics_med, BOOLEAN_METRICS_MED)


# ---------------------------------------------------------------------------------------
    # TF MODEL with CRANFIELD
    tf_model_cran = IR_TF(corpus_cranfield)

    # TF MODEL EVALUATOR
    tf_evaluator_cran = Evaluator(
        corpus_cranfield, cranfield_queries, cranfield_relations, tf_model_cran)
    try:
        tf_metrics_cran = load_model(TF_METRICS_CRAN)
    except OSError:
        tf_metrics_cran = tf_evaluator_cran.evaluate()
        save_model(tf_metrics_cran, TF_METRICS_CRAN)

# ---------------------------------------------------------------------------------------
    # TF MODEL with MED
    tf_model_med = IR_TF(corpus_med)

    # TF MODEL EVALUATOR
    tf_evaluator_med = Evaluator(
        corpus_med, queries_med, relations_med, tf_model_med)
    try:
        tf_metrics_med = load_model(TF_METRICS_MED)
    except OSError:
        tf_metrics_med = tf_evaluator_med.evaluate()
        save_model(tf_metrics_med, TF_METRICS_MED)

# ---------------------------------------------------------------------------------------
    # TF IDF MODEL with CRANFIELD
    tf_idf_model_cran = IR_TF_IDF(corpus_cranfield)

    # TF IDF MODEL EVALUATOR
    tf_idf_evaluator_cran = Evaluator(
        corpus_cranfield, cranfield_queries, cranfield_relations, tf_idf_model_cran)
    try:
        tf_idf_metrics_cran = load_model(TF_IDF_METRICS_CRAN)
    except OSError:
        tf_idf_metrics_cran = tf_idf_evaluator_cran.evaluate()
        save_model(tf_idf_metrics_cran, TF_IDF_METRICS_CRAN)

# ---------------------------------------------------------------------------------------
    # TF IDF MODEL with MED
    tf_idf_model_med = IR_TF_IDF(corpus_med)

    # TF IDF MODEL EVALUATOR
    tf_idf_evaluator_med = Evaluator(
        corpus_med, queries_med, relations_med, tf_idf_model_med)
    try:
        tf_idf_metrics_med = load_model(TF_IDF_METRICS_MED)
    except OSError:
        tf_idf_metrics_med = tf_idf_evaluator_med.evaluate()
        save_model(tf_idf_metrics_med, TF_IDF_METRICS_MED)

# ---------------------------------------------------------------------------------------
    # LEM MODEL with CRANFIELD
    lem_model_cran = IR_LEM(corpus_cranfield)

    # LDA MODEL EVALUATOR
    lem_evaluator_cran = Evaluator(
        corpus_cranfield, cranfield_queries, cranfield_relations, lem_model_cran)
    try:
        lem_metrics_cran = load_model(LEM_METRICS_CRAN)
    except OSError:
        lem_metrics_cran = lem_evaluator_cran.evaluate()
        save_model(lem_metrics_cran, LEM_METRICS_CRAN)

# ---------------------------------------------------------------------------------------
    # LEM MODEL with MED
    lem_model_med = IR_LEM(corpus_med)

    # LDA MODEL EVALUATOR
    lem_evaluator_med = Evaluator(
        corpus_med, queries_med, relations_med, lem_model_med)
    try:
        lem_metrics_med = load_model(LEM_METRICS_MED)
    except OSError:
        lem_metrics_med = lem_evaluator_med.evaluate()
        save_model(lem_metrics_med, LEM_METRICS_MED)

    app.run(debug=True)
