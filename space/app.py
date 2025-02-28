from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import datetime
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from astroquery.jplhorizons import Horizons

app = Flask(__name__)

# NASA API KEY
NASA_API_KEY = 'DEMO_KEY'

# ルート
@app.route("/", methods=["GET", "POST"])
def home():
    date = request.form.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
    apod_data = get_apod_data(date)
    return render_template("home.html", apod_data=apod_data)

# "/about"
@app.route("/about")
def about():
    return render_template("about.html")

# "/asteroids"
@app.route("/asteroids")
def asteroids():
    return render_template("asteroids.html")

# NASA APOD（Astronomy Picture of the Day）データを取得するヘルパー関数
def get_apod_data(date):
    url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}&date={date}"
    response = requests.get(url)
    return response.json()

# NASA小惑星データを取得するヘルパー関数
def get_asteroids_data(start_date, end_date):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"
    response = requests.get(url)
    return response.json()

# 小惑星データ取得API
@app.route('/get_asteroids', methods=['POST'])
def get_asteroids():
    start_date = request.form.get("start_date", datetime.datetime.now().strftime("%Y-%m-%d"))
    end_date = request.form.get("end_date", datetime.datetime.now().strftime("%Y-%m-%d"))
    return jsonify(get_asteroids_data(start_date, end_date))

# アプリケーションのエントリーポイント
if __name__ == "__main__":
    app.run(debug=True, port=5000)