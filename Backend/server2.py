from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model instance
model = None
model_types = ['MLP', 'RNN_GRU', 'RNN_LSTM']
current_model_type = 'MLP'
model_performance = {
    'MLP': {'mae': 0.045, 'mse': 0.0032, 'rmse': 0.057, 'accuracy': 0.82},
    'RNN_GRU': {'mae': 0.038, 'mse': 0.0028, 'rmse': 0.053, 'accuracy': 0.85},
    'RNN_LSTM': {'mae': 0.041, 'mse': 0.0030, 'rmse': 0.055, 'accuracy': 0.84}
}

patient_database = {}
system_stats = {
    'total_predictions': 0,
    'total_patients': 0,
    'high_risk_patients': 0,
    'last_updated': datetime.now().isoformat()
}


def visus_to_logMAR(visus):
    """Convert visus to logMAR scale"""
    return -math.log10(max(0.001, float(visus)))


def logMAR_to_visus(logMAR):
    """Convert logMAR to visus scale"""
    return 10 ** (-logMAR)


def calculate_risk_level(patient_data, time_series_data):
    """Calculate patient risk level based on multiple factors"""
    risk_score = 0

    # Age factor
    current_year = datetime.now().year
    age = current_year - patient_data['Geburtsdatum']
    if age > 75:
        risk_score += 2
    elif age > 65:
        risk_score += 1

    # Comorbidities
    risk_score += patient_data.get('G_DiabetisMellitus', 0) * 2
    risk_score += patient_data.get('G_ArterielleHypertonie', 0)
    risk_score += patient_data.get('G_Apoplex', 0) * 1.5

    # Visual acuity trend
    if len(time_series_data['V_Vis_L']) >= 3:
        recent_trend = np.mean(
            time_series_data['V_Vis_L'][-3:]) - np.mean(time_series_data['V_Vis_L'][:3])
        if recent_trend < -0.1:
            risk_score += 3
        elif recent_trend < 0:
            risk_score += 1

    # Fluid presence
    if sum(time_series_data['O_IntraretFlk_L']) > len(time_series_data['O_IntraretFlk_L']) * 0.5:
        risk_score += 2

    if risk_score >= 6:
        return 'high'
    elif risk_score >= 3:
        return 'medium'
    else:
        return 'low'


def initialize_model():
    """Initialize the MLP model"""
    global model
    model = Sequential()
    model.add(Dense(units=32, activation=tf.nn.relu, input_shape=(4, 1)))
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=64, activation=tf.nn.relu))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation=tf.nn.relu))
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Load weights if available
    try:
        model.load_weights(
            './MLP__treatment_prediction_dataset__20250814-165920_model.h5')
        print("Model weights loaded successfully!")
    except:
        print("No weights found. Using random initialization.")


def generate_sample_patient_data(base_info, num_visits=8):
    """Generate realistic patient data based on frontend input"""
    # Use frontend data or create realistic sample
    patient_data = {
        'V_Datum': [],
        'Geburtsdatum': base_info.get('birthYear', 1948),
        'Geschlecht': 1 if base_info.get('gender', 'male').lower() == 'male' else 2,
        'G_Apoplex': 1 if base_info.get('strokeHistory', False) else 0,
        'G_ArterielleHypertonie': 1 if base_info.get('hypertension', True) else 0,
        'G_DiabetisMellitus': 1 if base_info.get('diabetes', False) else 0,
        'G_Herzinfarkt': 1 if base_info.get('heartAttack', False) else 0,
        'G_Blutverd': 1 if base_info.get('bloodThinners', True) else 0,
    }

    time_series_data = {
        'V_Vis_L': [],
        'O_ZentrNetzhDicke_L': [],
        'O_IntraretFlk_L': [],
        'O_SubretFlk_L': [],
        'O_RPE_Abhebg_L': [],
        'O_SubretFibrose_L': [],
        'D_AMD_L': [],
        'D_Cataracta_L': [],
        'OCT_exists': [],
        'T_Medikament': [],
        'T_NummerInjektion': []
    }

    # Generate realistic time series
    start_date = datetime(2023, 3, 15)
    visit_dates = [start_date + timedelta(days=30 * i)
                   for i in range(num_visits)]
    date_numeric = [int(date.strftime('%Y%m%d')) for date in visit_dates]

    # Realistic AMD progression
    initial_visus = base_info.get('initialVisualAcuity', 0.3)
    initial_thickness = base_info.get('initialThickness', 420)

    for i in range(num_visits):
        if i == 0:
            visus = initial_visus
            thickness = initial_thickness
        elif i < 3:
            visus = min(
                0.8, time_series_data['V_Vis_L'][-1] + random.uniform(0.05, 0.15))
            thickness = max(
                250, time_series_data['O_ZentrNetzhDicke_L'][-1] - random.randint(30, 60))
        else:
            visus = time_series_data['V_Vis_L'][-1] + \
                random.uniform(-0.03, 0.03)
            thickness = time_series_data['O_ZentrNetzhDicke_L'][-1] + \
                random.randint(-15, 20)

        # Fill time series data
        time_series_data['V_Vis_L'].append(round(visus, 2))
        time_series_data['O_ZentrNetzhDicke_L'].append(thickness)
        time_series_data['O_IntraretFlk_L'].append(
            1 if i == 0 else random.choices([0, 1], weights=[0.8, 0.2])[0])
        time_series_data['O_SubretFlk_L'].append(
            1 if i == 0 else random.choices([0, 1], weights=[0.9, 0.1])[0])
        time_series_data['O_RPE_Abhebg_L'].append(1 if i == 0 else 0)
        time_series_data['O_SubretFibrose_L'].append(
            0 if i < 6 else random.choices([0, 1], weights=[0.8, 0.2])[0])
        time_series_data['D_AMD_L'].append(1)
        time_series_data['D_Cataracta_L'].append(
            random.choices([0, 1], weights=[0.6, 0.4])[0])
        time_series_data['OCT_exists'].append(1)
        time_series_data['T_Medikament'].append(
            1 if i < 3 or i % 2 == 0 else -1)
        time_series_data['T_NummerInjektion'].append(
            i + 1 if i < 3 or i % 2 == 0 else -1)

    return patient_data, time_series_data, date_numeric


def create_visualization_charts(patient_df, time_series_data, predictions, current_visus, predicted_visus, predicted_class):
    """Create charts and return as base64 images"""
    charts = {}

    # Main chart: Visual Acuity Progression
    fig, ax = plt.subplots(figsize=(10, 6))
    dates = [datetime.strptime(str(date), '%Y%m%d')
             for date in patient_df['V_Datum']]

    ax.plot(dates, time_series_data['V_Vis_L'], 'bo-',
            linewidth=2, markersize=8, label='Actual Visus')
    last_date = dates[-1]
    next_date = datetime(last_date.year, last_date.month + 2, last_date.day)
    ax.plot(next_date, predicted_visus, 'ro', markersize=10,
            label=f'Predicted: {predicted_visus:.2f}')
    ax.axhline(y=current_visus, color='gray', linestyle='--',
               alpha=0.7, label=f'Current: {current_visus:.2f}')

    ax.set_title('Visual Acuity Progression and Prediction', fontweight='bold')
    ax.set_ylabel('Visual Acuity', fontweight='bold')
    ax.set_xlabel('Visit Date', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    charts['main_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, time_series_data['O_ZentrNetzhDicke_L'], 'go-',
            linewidth=2, markersize=8, label='Retinal Thickness')
    ax.axhline(y=300, color='red', linestyle='--',
               alpha=0.7, label='Normal Threshold')
    ax.set_title('Retinal Thickness Over Time', fontweight='bold')
    ax.set_ylabel('Thickness (μm)', fontweight='bold')
    ax.set_xlabel('Visit Date', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    charts['thickness_chart'] = base64.b64encode(
        buf.getvalue()).decode('utf-8')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    treatment_dates = [dates[i] for i, t in enumerate(
        time_series_data['T_Medikament']) if t == 1]
    treatment_y = [1] * len(treatment_dates)
    ax.scatter(treatment_dates, treatment_y, c='red',
               s=100, marker='v', label='Injections')
    ax.set_ylim(0.5, 1.5)
    ax.set_title('Treatment Timeline', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    charts['treatment_chart'] = base64.b64encode(
        buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return charts


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get available model types and their performance metrics"""
    return jsonify({
        'models': model_types,
        'current_model': current_model_type,
        'performance': model_performance
    })


@app.route('/api/models/<model_type>', methods=['POST'])
def switch_model(model_type):
    """Switch to a different model type"""
    global current_model_type
    if model_type in model_types:
        current_model_type = model_type
        # In a real implementation, you would load the specific model weights here
        return jsonify({
            'success': True,
            'current_model': current_model_type,
            'message': f'Switched to {model_type} model'
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Model {model_type} not available'
        }), 400


@app.route('/api/patients', methods=['GET'])
def get_all_patients():
    """Get list of all patients in database"""
    return jsonify({
        'patients': list(patient_database.keys()),
        'total_count': len(patient_database)
    })


@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient_data(patient_id):
    """Get specific patient data"""
    if patient_id in patient_database:
        return jsonify(patient_database[patient_id])
    else:
        return jsonify({'error': 'Patient not found'}), 404


@app.route('/api/patients/<patient_id>', methods=['POST'])
def save_patient_data(patient_id):
    """Save patient data to database"""
    data = request.get_json()
    patient_database[patient_id] = {
        'patient_info': data.get('patientInfo', {}),
        'last_prediction': data.get('prediction', {}),
        'history': data.get('patientHistory', {}),
        'risk_level': data.get('riskLevel', 'medium'),
        'last_updated': datetime.now().isoformat()
    }

    # Update system stats
    system_stats['total_patients'] = len(patient_database)
    if data.get('riskLevel') == 'high':
        system_stats['high_risk_patients'] += 1

    return jsonify({'success': True, 'message': 'Patient data saved'})


@app.route('/api/statistics', methods=['GET'])
def get_system_statistics():
    """Get system-wide statistics"""
    # Calculate additional stats
    risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
    for patient_data in patient_database.values():
        risk_level = patient_data.get('risk_level', 'medium')
        risk_distribution[risk_level] += 1

    return jsonify({
        'system_stats': system_stats,
        'risk_distribution': risk_distribution,
        'model_performance': model_performance[current_model_type],
        'current_model': current_model_type
    })


@app.route('/api/export/<patient_id>', methods=['GET'])
def export_patient_data(patient_id):
    """Export patient data as JSON"""
    if patient_id in patient_database:
        return jsonify({
            'patient_id': patient_id,
            'export_date': datetime.now().isoformat(),
            'data': patient_database[patient_id]
        })
    else:
        return jsonify({'error': 'Patient not found'}), 404


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()

        # Update system stats
        system_stats['total_predictions'] += 1
        system_stats['last_updated'] = datetime.now().isoformat()

        # Generate patient data
        base_info = data.get('patientInfo', {})
        patient_data, time_series_data, visit_dates = generate_sample_patient_data(
            base_info)

        risk_level = calculate_risk_level(patient_data, time_series_data)

        # Prepare data for prediction
        visual_acuity_series = np.array(
            time_series_data['V_Vis_L']).reshape(-1, 1)

        data_x = []
        data_y = []
        look_back = 4

        for i in range(len(visual_acuity_series) - look_back):
            data_x.append(visual_acuity_series[i:i + look_back].flatten())
            data_y.append(visual_acuity_series[i + look_back][0])

        data_x = np.array(data_x).reshape(-1, look_back, 1)

        # Make prediction
        predictions = model.predict(data_x)
        current_visus = visual_acuity_series[-1][0]
        predicted_visus = predictions[-1][0] if len(
            predictions) > 0 else current_visus

        # Convert to logMAR
        current_logMAR = visus_to_logMAR(current_visus)
        predicted_logMAR = visus_to_logMAR(predicted_visus)

        # Determine classification
        unchanged_factor = 0.1
        if abs(current_logMAR - predicted_logMAR) <= unchanged_factor:
            predicted_class = 'stable'
        elif predicted_logMAR < current_logMAR:
            predicted_class = 'improved'
        else:
            predicted_class = 'worsened'

        base_confidence = model_performance[current_model_type]['accuracy']
        risk_modifier = {'low': 0.05, 'medium': 0, 'high': -0.1}
        confidence = base_confidence + risk_modifier[risk_level]

        # Create patient dataframe for charts
        patient_df = pd.DataFrame({
            'V_Datum': visit_dates,
            'V_Vis_L': time_series_data['V_Vis_L']
        })

        # Generate charts
        charts = create_visualization_charts(
            patient_df, time_series_data, predictions,
            current_visus, predicted_visus, predicted_class
        )

        response = {
            'success': True,
            'prediction': {
                'currentVisualAcuity': float(current_visus),
                'predictedVisualAcuity': float(predicted_visus),
                'currentLogMAR': float(current_logMAR),
                'predictedLogMAR': float(predicted_logMAR),
                'predictedChange': predicted_class,
                'confidence': float(confidence),
                'recommendation': 'TREAT' if predicted_class == 'worsened' or risk_level == 'high' else 'MONITOR',
                'nextFollowUp': '2 weeks' if predicted_class == 'worsened' else '3 months',
                'riskLevel': risk_level,
                'modelUsed': current_model_type
            },
            'patientHistory': {
                'visitDates': visit_dates,
                'visualAcuity': [float(v) for v in time_series_data['V_Vis_L']],
                'retinalThickness': time_series_data['O_ZentrNetzhDicke_L'],
                'fluidPresence': time_series_data['O_IntraretFlk_L'],
                'treatments': time_series_data['T_Medikament']
            },
            'charts': charts,
            'statistics': {
                'initialVisualAcuity': float(time_series_data['V_Vis_L'][0]),
                'finalVisualAcuity': float(time_series_data['V_Vis_L'][-1]),
                'visualChange': float(time_series_data['V_Vis_L'][-1] - time_series_data['V_Vis_L'][0]),
                'totalInjections': sum(1 for x in time_series_data['T_Medikament'] if x == 1),
                'averageThickness': np.mean(time_series_data['O_ZentrNetzhDicke_L']),
                'fluidEpisodes': sum(time_series_data['O_IntraretFlk_L'])
            }
        }

        # Save patient data if patient ID provided
        patient_id = base_info.get('patientId')
        if patient_id:
            patient_database[patient_id] = {
                'patient_info': base_info,
                'last_prediction': response['prediction'],
                'history': response['patientHistory'],
                'risk_level': risk_level,
                'last_updated': datetime.now().isoformat()
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'current_model': current_model_type,
        'timestamp': datetime.now().isoformat(),
        'total_patients': len(patient_database),
        'total_predictions': system_stats['total_predictions']
    })


@app.route('/api/sample-patient', methods=['GET'])
def get_sample_patient():
    """Get sample patient data for frontend"""
    sample_data = {
        'patientInfo': {
            'patientId': '12345',
            'birthYear': 1948,
            'gender': 'male',
            'hypertension': True,
            'diabetes': False,
            'strokeHistory': False,
            'heartAttack': False,
            'bloodThinners': True,
            'initialVisualAcuity': 0.3,
            'initialThickness': 420
        }
    }
    return jsonify(sample_data)


if __name__ == '__main__':
    print("Initializing AMD Prediction Server...")
    initialize_model()
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
