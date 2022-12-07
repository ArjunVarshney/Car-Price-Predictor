from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
data = pd.read_csv("CleanData.csv")

carModel = pickle.load(open("CarPricePredictorModel.pkl", "rb"))


@app.route("/")
def index():
    companies = sorted(data["company"].unique())
    carModels = sorted(data["name"].unique())
    year = sorted(data["year"].unique(), reverse=True)
    fuelType = data["fuel_type"].unique()
    return render_template("index.html", companies=companies, carModels=carModels, years=year, fuelType=fuelType)


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    company = data["company"]
    model = data["model"]
    year = int(data["year"])
    fuelType = data["fuelType"]
    kmsDriven = int(data["kmsDriven"])

    prediction = carModel.predict(pd.DataFrame([[model, company, year, kmsDriven, fuelType]], columns=[
                                  "name", "company", "year", "kms_driven", "fuel_type"]))

    return str(prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
