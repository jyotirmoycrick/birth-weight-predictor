from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle

app = Flask(__name__)

def convert_df(baby_data):
    gestation = baby_data['gestation']
    parity = baby_data['parity']
    age = baby_data['age']
    height = baby_data['height']
    weight = baby_data['weight']
    smoke = baby_data['smoke']

    return {
        'gestation': [float(gestation)],
        'parity': [int(parity)],
        'age': [float(age)],
        'height': [float(height)],
        'weight': [float(weight)],
        'smoke': [float(smoke)]
        }

    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_value():
    baby_data = request.form
    baby_data_df= convert_df(baby_data)
    baby_df = pd.DataFrame(baby_data_df)

    #load machine learning trained model

    with open('model.pkl','rb') as obj:
        model = pickle.load(obj)

    #make predictions on user data 
    baby_pred = model.predict(baby_df)
    prediction = float(baby_pred)
    
    response = {
        'prediction':prediction
    }

    return render_template('index.html',prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
