from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load Ridge Regressor Model and Standard Scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def helloworld():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predictData():
    if request.method == 'POST':
        # Extract data from form (fixed `.get()` syntax)
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))  
        region = float(request.form.get('Region'))
        # Scale data
        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        scaled_data = standard_scaler.transform(input_data)
        # Make prediction
        prediction = ridge_model.predict(scaled_data)
        return render_template('home.html', result=prediction[0])


    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
