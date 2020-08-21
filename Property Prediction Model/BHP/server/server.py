# Making a python flask server that will handle all the backend of our application
# it will be using pickle file and inputs arguments files(columns) to predict models
# python flask will make use of routing and access the function which will get the data from pickle and column file
# python flask will later attach to our ui website

# SERVER
# 1) Make server file in server folder
# 2) check version control make anaconda intrepeter
# 3) make function for http response
# x) run file in command propmt
# x) make artifacte folder in server
# y)
# UTIL
# 5) make utli file which will
# 	1) load saved artifacts datacolumns, locations and model
# 	x) make a function to give access to locations
# 	2) will have price prediction function
# SERVER
# 6) access prediction function in util from server file and make url post request
# 	1) seperate each datacolumn from request
# 	2) use util
# 7) enter input values using postman and make prediction
# 8) make a function that gets all locations
# =====================================================
# 1) import flask
from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():

    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/get_location_names')
def get_location_names():
    print("SERVER get_location_name START")
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    print("SERVER response", response)

    print("SERVER get_location_name END")
    return response

@app.route('/predict1')
def predict_home_price1():
    total_sqft = 2000
    location = '1st block jayanagar'
    bhk = 2
    bath = 2
    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("starting python flask server from property home price prediction!!!")
    # util.load_save_artifacts()
    app.run()
