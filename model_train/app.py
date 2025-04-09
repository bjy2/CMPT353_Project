import os
import uuid
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

import joblib



matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRAINING_FOLDER'] = 'training_data'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

GAIT_STATES = {
    'walking': 'Walking',
    'running': 'Running',
    'stairs': 'Stairs',
    'unstable': 'Unstable'
}

MODEL_PATH = os.path.join(app.config['MODEL_FOLDER'], 'gait_model.pkl')
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def mask_low_variation(signal, threshold):
    grad = np.abs(np.gradient(signal))
    mask = grad > threshold
    masked_signal = signal.copy()
    masked_signal[~mask] = np.nan
    return masked_signal

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def preprocess_data(data, cutoff=50, order=2, variation_threshold=0.19):
    
    if 'time' not in data.columns:
        raise ValueError("Missing 'time' column in data.")

    time_diff = np.diff(data['time'])
    fs = 1 / np.mean(time_diff) if len(time_diff) > 0 else 100 

    for col in ['gFx', 'gFy', 'gFz']:
        if col in data.columns:
            filtered = butter_lowpass_filter(data[col], cutoff=cutoff, fs=fs, order=order)
            filtered_masked = mask_low_variation(filtered, threshold=variation_threshold)
            data[f'{col}_filt'] = filtered_masked
            data[f'{col}_filt'] = filtered_masked
            data[f'{col}_filt'].fillna(0, inplace=True)
    
    return data


def extract_features(data):
    features = {}
    
    data = preprocess_data(data)
    vertical_acc = data['gFz_filt'].values
    horizontal_acc = np.sqrt(data['gFx_filt'].values**2 + data['gFy_filt'].values**2)
    
    features['mean_vert_acc'] = np.mean(vertical_acc)
    features['std_vert_acc'] = np.std(vertical_acc)
    features['max_vert_acc'] = np.max(vertical_acc)
    features['min_vert_acc'] = np.min(vertical_acc)
    features['mean_horiz_acc'] = np.mean(horizontal_acc)
    features['std_horiz_acc'] = np.std(horizontal_acc)
    
    vert_peaks, _ = find_peaks(vertical_acc, height=0.5, distance=30)
    horiz_peaks, _ = find_peaks(horizontal_acc, height=0.3, distance=30)
    features['vert_steps'] = len(vert_peaks)
    features['horiz_steps'] = len(horiz_peaks)
    
    if len(vert_peaks) > 1:
        peak_times = data['time'].iloc[vert_peaks].values
        step_intervals = np.diff(peak_times)
        features['step_interval_mean'] = np.mean(step_intervals)
        features['step_interval_std'] = np.std(step_intervals)
        features['cadence'] = 60 / features['step_interval_mean']
    else:
        features['step_interval_mean'] = 0
        features['step_interval_std'] = 0
        features['cadence'] = 0
    
    n = len(vertical_acc)
    yf = fft(vertical_acc)
    xf = np.linspace(0.0, 50, n//2)
    spectrum = 2.0/n * np.abs(yf[0:n//2])
    
    features['dominant_freq'] = xf[np.argmax(spectrum)]
    features['spectrum_energy'] = np.sum(spectrum**2)
    
    features['vert_acc_range'] = features['max_vert_acc'] - features['min_vert_acc']
    features['vert_acc_var'] = np.var(vertical_acc)
    features['horiz_acc_var'] = np.var(horizontal_acc)

    
    
    return features

def extract_gait_features(data):
    """Extract gait features for display"""
    features = extract_features(data)
    peaks, _ = find_peaks(data['gFz_filt'].values, height=0.5, distance=30)
    return features, peaks

def train_model(model_type='naive_bayes'):
    """Train classification model using the specified algorithm."""

    global model

    VALID_MODELS = ['naive_bayes', 'knn', 'random_forest']
    if model_type not in VALID_MODELS:
        return False, f"Invalid model type: {model_type}"

    training_files = [f for f in os.listdir(app.config['TRAINING_FOLDER']) if f.endswith('.csv')]
    if not training_files:
        return False, "No training data found"

    features_list = []
    labels = []

    for filename in training_files:
        label = filename.split('_')[0]
        # Combine upstairs/downstairs into stairs
        if label in ['upstairs', 'downstairs']:
            label = 'stairs'
        if label not in GAIT_STATES:
            continue

        filepath = os.path.join(app.config['TRAINING_FOLDER'], filename)
        data = pd.read_csv(filepath)

        try:
            features = extract_features(data)
            features_list.append(features)
            labels.append(label)
        except Exception:
            continue

    if not features_list:
        return False, "Failed to extract features from training data"

    feature_names = features_list[0].keys()
    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(labels)

    if len(y) < 8 or len(set(y)) > len(y) * 0.5:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    if model_type == 'knn':
        model = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=3)
        )
    elif model_type == 'random_forest':
        model = make_pipeline(
            SimpleImputer(strategy='mean'),
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
    else:  # default: Naïve Bayes
        model = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            GaussianNB()
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)

    return True, f"Model trained using {model_type.replace('_', ' ').title()}, test accuracy: {accuracy:.2f}"


def predict_gait_state(data):
    if model is None:
        return "Unknown (model not trained)"
    
    features = extract_features(data)
    feature_names = features.keys()
    X = pd.DataFrame([features], columns=feature_names)
    
    prediction = model.predict(X)[0]
    return GAIT_STATES.get(prediction, "Unstable")

@app.route('/')
def index():
    return render_template('index.html', gait_states=GAIT_STATES, model_exists=model is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4().hex}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            data = pd.read_csv(filepath)
            
            required_columns = ['time', 'gFx', 'gFy', 'gFz']
            if not all(col in data.columns for col in required_columns):
                os.remove(filepath)
                return render_template('error.html', 
                                     message="CSV file must contain 'time', 'gFx', 'gFy', 'gFz' columns")
            
            data = preprocess_data(data)
            features, peaks = extract_gait_features(data)
            xf, yf = frequency_analysis(data)
            plots = create_plots(data, peaks, xf, yf)
            
            avg_stride_length = float(request.form.get('stride_length', 0.7))
            walking_speed = features['cadence'] * avg_stride_length / 60  # m/s
            walking_speed_kmh = walking_speed * 3.6
            
            gait_state = predict_gait_state(data)
            
            os.remove(filepath)
            
            return render_template('results.html', 
                                 features=features,
                                 plots=plots,
                                 stride_length=avg_stride_length,
                                 walking_speed=walking_speed,
                                 walking_speed_kmh=walking_speed_kmh,
                                 gait_state=gait_state,
                                 model_exists=model is not None)
            
        except Exception as e:
            os.remove(filepath)
            return render_template('error.html', message=f"Analysis error: {str(e)}")
    
    return redirect(request.url)

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return redirect(url_for('index', _anchor='train'))
    
    file = request.files['file']
    gait_state = request.form.get('gait_state')
    if gait_state in ['upstairs', 'downstairs']:
        gait_state = 'stairs'
    
    if file.filename == '' or not gait_state:
        return redirect(url_for('index', _anchor='train'))
    
    if file and allowed_file(file.filename) and gait_state in GAIT_STATES:

        filename = f"{gait_state}_{uuid.uuid4().hex}.csv"
        filepath = os.path.join(app.config['TRAINING_FOLDER'], filename)
        file.save(filepath)
        
        if gait_state == 'running':
            data = pd.read_csv(filepath)
            try:
                features = extract_features(data)
                if features['cadence'] < 50:
                    os.remove(filepath)
                    return render_template('error.html',
                                        message="This doesn't appear to be running data (low cadence). Please verify.")
            except Exception as e:
                os.remove(filepath)
                return render_template('error.html',
                                    message=f"Error validating running data: {str(e)}")
        
        return render_template('train_success.html', 
                            gait_state=GAIT_STATES[gait_state],
                            filename=filename)
    
    return redirect(url_for('index', _anchor='train'))

@app.route('/train_model', methods=['POST'])

def train_model_route():
    selected_model = request.form.get('model_type', 'naive_bayes')
    success, message = train_model(selected_model)
    if success:
        return render_template('model_trained.html', message=message)
    else:
        return render_template('error.html', message=message)


def frequency_analysis(data):
    n = len(data)
    yf = fft(data['gFz_filt'].values)
    xf = np.linspace(0.0, 50, n//2)
    
    return xf, 2.0/n * np.abs(yf[0:n//2])

def create_plots(data, peaks, xf, yf):
    matplotlib.use('Agg')  
    plots = {}
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['gFz'], label='Raw vertical acceleration', alpha=0.5)
    plt.plot(data['time'], data['gFz_filt'], label='Filtered vertical acceleration')
    
    if len(peaks) > 0:
        plt.plot(data['time'].iloc[peaks], data['gFz_filt'].iloc[peaks], 'rx', label='Detected steps')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (g)')
    plt.title('Vertical Acceleration with Step Detection')
    plt.legend()
    plt.grid()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['acceleration_plot'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    plt.figure(figsize=(10, 4))
    if len(peaks) > 1:
        peak_times = data['time'].iloc[peaks].values
        step_intervals = np.diff(peak_times)
        plt.plot(peak_times[1:], step_intervals, 'o-')
    plt.xlabel('Time (s)')
    plt.ylabel('Step interval (s)')
    plt.title('Step Time Intervals')
    plt.grid()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['interval_plot'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    plt.figure(figsize=(10, 4))
    plt.plot(xf, yf)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Vertical Acceleration Frequency Spectrum')
    plt.grid()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['spectrum_plot'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return plots

if __name__ == '__main__':
    app.run(debug=True)