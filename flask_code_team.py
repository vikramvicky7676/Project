import pandas as pd
from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime

app = Flask(__name__) 

# Load the LSTM model from the .pkl file
model = pickle.load(open('lstm_model11.pkl', 'rb'))
df = pd.read_excel("D:\\projects\\project 1st oil price predection\\DCOILWTICO (1).xls")
# Load the MinMaxScaler for data normalization
scaler = MinMaxScaler()
scaler.fit(np.zeros((1, 1)))

@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    selected_date = request.form['date']
    
    # Preprocess the input date
    input_date = pd.to_datetime(selected_date)
    date_range = pd.date_range(start='2000-01-01', end=input_date, freq='D')

    # Extract the previous 'sequence_length' prices from the dataframe
    sequence_length = 10  # Set the sequence length based on your training data
    date_indices = df[df['Date'].isin(date_range)].index[-sequence_length:]
    input_sequence = df.loc[date_indices]['Price'].values

    # Normalize the input sequence using the previously used scaler
    input_data_scaled = scaler.transform(input_sequence.reshape(-1, 1))

    # Reshape the input data for LSTM (batch_size, timesteps, input_dim)
    input_data_scaled = input_data_scaled.reshape(1, -1, 1)

    # Make the prediction
    prediction_scaled = model.predict(input_data_scaled)
    prediction_dollars = scaler.inverse_transform(prediction_scaled)

    # Convert the prediction from dollars to rupees using the conversion factor (74.50)
    conversion_factor = 74.50
    prediction_rupees = prediction_dollars * conversion_factor

    # Render the template with the predicted value in rupees
    return render_template('index.html', prediction=prediction_rupees[0][0].round(2))

if __name__ == '__main__':
    app.run(debug=True, port=8000, use_reloader=False)



# -*- coding: utf-8 -*-

