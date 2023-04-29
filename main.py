from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app= Flask(__name__)
data = pd.read_csv("combined_sorted_final.csv")
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    
    locations = sorted(data['location'].unique())

    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')  
    sqrt = request.form.get('total_sqrt')
    
    # Check if any of the inputs are missing and display an error message
    if location == '' or bhk == '' or bath == '' or sqrt == '':
        return 'Please enter all the required inputs'
    
    # Convert input values to the correct data type
    bhk = int(bhk)
    bath = int(bath)
    sqrt = float(sqrt)
    
    input=pd.DataFrame([[location,sqrt,bath,bhk]],columns=['location', 'total_sqrt','bath','bhk'])
    prediction = pipe.predict(input)
    prediction = prediction[0] * 1e5
    return str(np.round(prediction,2))





if __name__=="__main__":
    app.run(debug=True, port=5001)

