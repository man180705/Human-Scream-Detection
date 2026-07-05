from flask import Flask, render_template, request
from joblib import load
import numpy as np
import librosa

decision_tree_model = load(open('decision_tree_classifier.pkl', 'rb'))
knn_model = load(open('knn_classifier.pkl', 'rb'))
logistic_model = load(open('logistic_reg_classifier.pkl', 'rb'))
random_forest_model = load(open('random_forest_classifier.pkl', 'rb'))
svm_model = load(open('svm_classifier.pkl', 'rb'))
xgb_model = load(open('xgb_classifier.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.htm',result=None)

@app.route('/predict', methods=['POST'])
def prediction():
    audio_file = request.files['audioFile']
    audio_file.save('uploaded_audio.wav')
    audioFile = 'uploaded_audio.wav'

    RATE = 44100
    y, _ = librosa.load(audioFile, sr=RATE)

    mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=RATE)
    chroma = librosa.feature.chroma_stft(y=y, sr=RATE)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=RATE)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=RATE)
    rmse = librosa.feature.rms(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=RATE)
    onset_env = librosa.onset.onset_strength(y=y, sr=RATE)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=RATE)

    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zero_crossing_rate),
        np.mean(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(rmse),
        np.mean(spectral_rolloff),
        tempo
    ]).reshape(1, -1)

    result_dt = decision_tree_model.predict(feature_vector)
    result_knn = knn_model.predict(feature_vector)
    result_logistic = logistic_model.predict(feature_vector)
    result_randomf = random_forest_model.predict(feature_vector)
    result_svm = svm_model.predict(feature_vector)
    result_xgb = xgb_model.predict(feature_vector)

    zeros = (1 - result_dt) + (1 - result_knn) + (1 - result_logistic) + (1 - result_randomf) + (1 - result_svm) + (1 - result_xgb)
    ones = result_dt + result_knn + result_logistic + result_randomf + result_svm + result_xgb

    if ones > zeros:
        return render_template('index.htm', result=1)
    elif zeros > ones:
        return render_template('index.htm', result=0)
    else:
        ans_1 = result_svm + result_randomf + result_dt
        ans_0 = (1 - result_svm) + (1 - result_randomf) + (1 - result_dt)
        if ans_0 > ans_1:
            return render_template('index.htm', result=0)
        else:
            return render_template('index.htm', result=1)

if __name__ == '__main__':
    app.run(debug=True)
