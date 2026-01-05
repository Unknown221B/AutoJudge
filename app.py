from flask import Flask, render_template, request
import joblib
from scipy.sparse import hstack
from features import handcrafted_features

app = Flask(__name__)

tfidf = joblib.load("tfidf.pkl")
classifier = joblib.load("classifier.pkl")
regressor = joblib.load("regressor.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    data = {"desc": "", "inp": "", "out": ""}

    if request.method == "POST":
        desc = request.form.get("description", "").strip()
        inp = request.form.get("input_desc", "").strip()
        out = request.form.get("output_desc", "").strip()

        data = {"desc": desc, "inp": inp, "out": out}

        if desc and inp and out:
            combined = desc + " " + inp + " " + out
            X_text = tfidf.transform([combined])
            X_hand = handcrafted_features([combined])
            X_final = hstack([X_text, X_hand])

            result = {
                "class": classifier.predict(X_final)[0].capitalize(),
                "score": round(float(regressor.predict(X_final)[0]), 2)
            }

    return render_template("index.html", result=result, data=data)

if __name__ == "__main__":
    app.run(debug=True)