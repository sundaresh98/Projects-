import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def results():
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    sqft_living = float(request.form['sqft_living'])
    sqft_lot = float(request.form['sqft_lot'])
    floors = float(request.form['floors'])
    waterfront = float(request.form['waterfront'])
    view = float(request.form['view'])
    condition = float(request.form['condition'])
    sqft_above = float(request.form['sqft_above'])
    sqft_basement = float(request.form['sqft_basement'])
    yr_built = float(request.form['yr_built'])
    yr_renovated = float(request.form['yr_renovated'])


    X = np.array([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated]])
    model = pickle.load(open('model.pkl','rb'))
    Y_predict = model.predict(X)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    app.run(debug = True, port = 1010)




