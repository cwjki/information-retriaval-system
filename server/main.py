from pathlib import Path
from pickle import TRUE
from traceback import print_tb
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from src.vsm_cranfield.metrics_evaluator import Evaluator
from src.vsm_cranfield.vector_space_model import VectorSpaceModel
from src.vsm_cranfield.cranfield_parser import CranfieldParser
from src.utils import save_model, load_model
from src.irs_med.models import IR_Boolean, IR_TF_IDF
from src.irs_med.med_parser import med_parse_collection, med_parse_queries, med_parse_relevances
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

BOOLEAN_RK = str(path) + '/src/data/boolean.rk'


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
    # CRANFIELD VSM EVAL
    # evaluator = Evaluator(documents, queries, relations, vector_space_model)

    # # COMPUTE ALL THE MED QUERIES
    # boolean_model.compute_ranking(queries_med)
    # tf_idf_model.compute_ranking(queries_med)

    # MED BOOLEAN EVAL
    boolean_evaluator = IREvaluator(
        relevance_med, boolean_ranking_queries, True, 1)
    # tf_idf_evaluator = IREvaluator(
    #     relevance_med, tf_idf_model.ranking_query, True, 1)

    # # vsm_metrics = evaluator.evaluate()
    boolean_metrics = boolean_evaluator.evaluate()
    # tf_idf_metrics = tf_idf_evaluator.evaluate()

    return render_template('evaluate.html', content=[vsm_metrics, boolean_metrics])


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
    relevance_med = med_parse_relevances(MED_REL)
    queries_med = med_parse_collection(MED_QUERY)

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

    # BOOLEAN MODEL
    boolean_model = IR_Boolean(corpus_med)
    tf_idf_model = IR_TF_IDF(corpus_med)

    # COMPUTE ALL THE MED QUERIES
    try:
        boolean_ranking_queries = load_model(BOOLEAN_RK)
    except OSError:
        print("AQUI")
        aux_queries = queries_med[:2]
        print(aux_queries)
        boolean_model.compute_ranking(aux_queries)
        print("SALIO")
        boolean_ranking_queries = boolean_model.ranking_query
        print(boolean_ranking_queries)
        save_model(boolean_ranking_queries, BOOLEAN_RK)

    # MED BOOLEAN EVAL
    boolean_evaluator = IREvaluator(
        relevance_med, boolean_model.ranking_query, True, 1)

    # tf_idf_model.compute_ranking(queries_med)

    # # MED BOOLEAN EVAL
    # boolean_evaluator = IREvaluator(
    #     relevance_med, boolean_model.ranking_query, True, 1)
    # tf_idf_evaluator = IREvaluator(
    #     relevance_med, tf_idf_model.ranking_query, True, 1)

    # # vsm_metrics = evaluator.evaluate()
    # boolean_metrics = boolean_evaluator.evaluate(boolean_model.ranking_query)
    # tf_idf_metrics = tf_idf_evaluator.evaluate()

    app.run(debug=True)
