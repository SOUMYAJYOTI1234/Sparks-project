from flask import Flask, render_template, request

from src.student_score_prediction.pipeline.prediction_pipeline import (
    CustomData,
    PredictPipeline,
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    """Home page with prediction form and result display."""
    result = None
    hours = None

    if request.method == "POST":
        try:
            hours = float(request.form.get("hours", 0))

            data = CustomData(hours=hours)
            features_df = data.get_data_as_dataframe()

            pipeline = PredictPipeline()
            predictions = pipeline.predict(features_df)
            result = round(predictions[0], 2)

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, hours=hours)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
