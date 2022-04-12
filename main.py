from flask import Flask, render_template,request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__,template_folder='templates')
data = pd.read_csv('Finaldata_csv_bhp.csv')
with open('model.pkl', 'rb') as f:
    pipe = pickle.load(f)





@app.route('/')
def index():
    places = sorted(data['location'].unique())
    return render_template('index.html',location=places)

@app.route('/predict', methods = ['POST'])
def predict():
    place = request.form.get('location')
    bhk = request.form.get('bhk')
    bhk = float(bhk)
    bath = request.form.get('bath')
    bath = float(bath)
    sqft = request.form.get('total_sqft')
    input = pd.DataFrame([[place ,sqft, bath, bhk]], columns = ['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0]
    return str(np.round(prediction,2))



if __name__=="__main__":
    app.run(debug = True, port = 5001)