from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model and data
model_path = 'model\score_regressor_model.pkl'  # Update the path as necessary
data_path = 'data\Final_Model_Dataset.csv'  # Update the path as necessary
model = joblib.load(model_path)
data = pd.read_csv(data_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get-locations', methods=['GET'])
def get_locations():
    unique_locations = data['Location'].unique().tolist()
    return jsonify({'locations': unique_locations})

@app.route('/get-crops', methods=['GET'])
def get_crops():
    location = request.args.get('location')
    if not location:
        return jsonify({'error': 'Location parameter is required'}), 400
    unique_crops = data[data['Location'] == location]['Crop'].unique().tolist()
    return jsonify({'crops': unique_crops})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse data from request
        data = request.get_json(force=True)
        crop = data.get('Crop')
        location = data.get('Location')
        temperature = data.get('Temperature')
        precipitation = data.get('Precipitation')

        # Validate input data
        if not all([crop, location, temperature, precipitation]):
            raise ValueError("Missing data for prediction")

        # Data preparation and prediction
        prediction, premium = process_data(crop, location, temperature, precipitation)
        return jsonify({
            'Crop': crop,
            'Location': location,
            'Predicted Risk Score': prediction,
            'Insurance Premium': premium
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/get-locations-and-crops', methods=['GET'])
def get_locations_and_crops():
    unique_locations = data['Location'].unique().tolist()
    unique_crops = data['Crop'].unique().tolist()
    return jsonify({'locations': unique_locations, 'crops': unique_crops})

def process_data(crop, location, temperature, precipitation):
    # Fetch optimal conditions for the crop and location from the dataset
    optimal_conditions = data[(data['Crop'] == crop) & (data['Location'] == location)].iloc[0]
    optimal_temp_midpoint = (optimal_conditions['AvgTempRequiredMin'] + optimal_conditions['AvgTempRequiredMax']) / 2
    optimal_precip_midpoint = (optimal_conditions['PrecipitationMin'] + optimal_conditions['PrecipitationMax']) / 2

    # Calculate deviations
    temp_deviation = abs(temperature - optimal_temp_midpoint)
    precip_deviation = abs(precipitation - optimal_precip_midpoint)

    # Normalize deviations
    max_temp_deviation = data['TempDeviation'].max()
    max_precip_deviation = data['PrecipDeviation'].max()
    normalized_temp_deviation = (temp_deviation / max_temp_deviation) * 100
    normalized_precip_deviation = (precip_deviation / max_precip_deviation) * 100

    # Predict the risk score
    input_features = [[normalized_temp_deviation, normalized_precip_deviation]]
    predicted_risk_score = model.predict(input_features)[0]

    # Calculate insurance premium
    if predicted_risk_score <= 33:
        insurance_premium = 'Low'
    elif predicted_risk_score <= 66:
        insurance_premium = 'Medium'
    else:
        insurance_premium = 'High'

    return predicted_risk_score, insurance_premium

if __name__ == '__main__':
    app.run(debug=True, port=5000)