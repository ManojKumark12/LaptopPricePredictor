from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and dataset
pipe = pickle.load(open('model.pkl', 'rb'))
# df = pickle.load(open('df.pkl', 'rb'))

@app.route('/')
def index():
    # Pass the data for the form inputs
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input values
    company = request.form.get('Company')
    type_name = request.form.get('Type')
    ram = int(request.form.get('Ram'))
    weight = float(request.form.get('Weight'))
    touchscreen = 1 if request.form.get('Touchscreen') == 'Yes' else 0
    ips = 1 if request.form.get('IPS') == 'Yes' else 0
    screen_size = float(request.form.get('Screen_Size'))
    resolution = request.form.get('Resolution')
    cpu = request.form.get('Cpu')
    hdd = int(request.form.get('HDD'))
    ssd = int(request.form.get('SDD'))
    gpu = request.form.get('Gpu')
    os = request.form.get('OS')

    # Process resolution and calculate ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create the input array for prediction
     # Create a DataFrame for the model input
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'IPS Panel': [ips],
        'ppi': [ppi],
        'Cpu': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu': [gpu],
        'OpSys': [os]
    })
    # query = query.reshape(1, 12)

    # Predict the price
    try:
        predicted_price = np.exp(pipe.predict(input_data)[0])
        return render_template('index.html', prediction_text=f'The predicted price of this configuration is ₹{int(predicted_price)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
if __name__ == "__main__":
    app.run(debug=True)
