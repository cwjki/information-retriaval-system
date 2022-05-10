from flask import Flask

from server.src.cransfield_parser import CransfieldParser

app = Flask(__name__)

@app.route("/")
def home():
    return "SRI Proyecto Final"


def main():
    cransfieldParser = CransfieldParser()
    documents = cransfieldParser.parse()

if __name__ == "__main__":
    app.run()
