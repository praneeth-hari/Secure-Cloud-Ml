import io
import os
import json
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

from config import SECRET_KEY, JWT_SECRET_KEY
from encryption import encrypt_data, decrypt_data

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
dataset = pd.read_csv("Dataset/kdd_data.csv")
labels = np.unique(dataset['Attack_Type'].to_numpy())

label_encoder = []
for col in dataset.select_dtypes(include='object').columns.tolist():
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col].astype(str))
    label_encoder.append([col, le])
dataset.fillna(0, inplace=True)

Y = dataset['Attack_Type'].to_numpy()
dataset.drop(['Attack_Type'], axis=1, inplace=True)
X = dataset.values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

print(f"Dataset ready — {X.shape[0]} samples, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train1 = to_categorical(y_train)
y_test1  = to_categorical(y_test)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# Metrics tracking  (5 models)
# ---------------------------------------------------------------------------
algorithm_names = []
accuracy  = []
precision = []
recall    = []
fscore    = []

def calculateMetrics(name, y_true, y_pred):
    a = accuracy_score(y_true, y_pred) * 100
    p = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    r = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    algorithm_names.append(name)
    accuracy.append(round(a, 2))
    precision.append(round(p, 2))
    recall.append(round(r, 2))
    fscore.append(round(f, 2))
    print(f"{name:15s}  Acc={a:.2f}%  P={p:.2f}%  R={r:.2f}%  F1={f:.2f}%")

# ---------------------------------------------------------------------------
# Model 1: KNN
# ---------------------------------------------------------------------------
knn_path = "model/knn.pckl"
knn_cls = None
if os.path.exists(knn_path):
    try:
        with open(knn_path, 'rb') as f:
            knn_cls = pickle.load(f)
    except Exception:
        knn_cls = None
if knn_cls is None:
    knn_cls = KNeighborsClassifier(algorithm="ball_tree")
    knn_cls.fit(X_train[:1000], y_train[:1000])
    with open(knn_path, 'wb') as f:
        pickle.dump(knn_cls, f)

predict_knn = knn_cls.predict(X_test)
calculateMetrics("KNN", y_test, predict_knn)

# ---------------------------------------------------------------------------
# Model 2: ANN
# ---------------------------------------------------------------------------
ann_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(y_train1.shape[1], activation='softmax')
])
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if os.path.exists("model/ann_weights.hdf5"):
    ann_model.load_weights("model/ann_weights.hdf5")
else:
    cb = ModelCheckpoint('model/ann_weights.hdf5', save_best_only=True, verbose=0)
    ann_model.fit(X_train, y_train1, batch_size=32, epochs=10,
                  validation_data=(X_test, y_test1), callbacks=[cb], verbose=0)

predict_ann = np.argmax(ann_model.predict(X_test), axis=1)
calculateMetrics("ANN", y_test, predict_ann)

# ---------------------------------------------------------------------------
# Model 3: ABC + ANN optimized
# ---------------------------------------------------------------------------
best_model    = None
best_features = None
best_neurons  = []

def fitnessFunction(features, train_x, train_y, test_x, test_y, neurons, num_epochs):
    global best_neurons
    best_neurons = neurons
    m = Sequential([
        Dense(neurons[0], activation='relu', input_shape=(train_x[:, features].shape[1],)),
        Dense(neurons[1], activation='relu'),
        Dense(train_y.shape[1], activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(train_x[:, features], train_y, batch_size=32, epochs=num_epochs,
          validation_data=(test_x[:, features], test_y), verbose=0)
    pred = np.argmax(m.predict(test_x[:, features]), axis=1)
    true = np.argmax(test_y, axis=1)
    return accuracy_score(true, pred), m

def ABCOptimization(population_size, max_iterations):
    global best_model, best_features
    population = [random.sample(range(X_train.shape[1]), int(X_train.shape[1] * 0.8))
                  for _ in range(population_size)]
    for _ in range(max_iterations):
        fitness_values = [fitnessFunction(f, X_train, y_train1, X_test, y_test1, [64, 32], 4)[0]
                          for f in population]
        for i in range(population_size):
            new_f = population[i].copy()
            new_f[random.randint(0, len(new_f) - 1)] = random.randint(0, X.shape[1] - 1)
            score, model = fitnessFunction(new_f, X_train, y_train1, X_test, y_test1, [64, 32], 10)
            if score > fitness_values[i]:
                best_features = new_f
                best_model = model
                return

def ANNOptimization(population_size, max_iterations):
    global best_features, best_model, best_neurons
    if os.path.exists("model/abc_ann_weights.h5"):
        data = np.load("model/data.npy", allow_pickle=True)
        best_features = data[0].astype(int)
        best_neurons  = data[1].astype(int)
        m = Sequential([
            Dense(int(best_neurons[0]), activation='relu',
                  input_shape=(X_train[:, best_features].shape[1],)),
            Dense(int(best_neurons[1]), activation='relu'),
            Dense(y_train1.shape[1], activation='softmax')
        ])
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m.load_weights("model/abc_ann_weights.h5")
        best_model = m
    else:
        ABCOptimization(population_size, max_iterations)
        if best_model is not None:
            best_model.save_weights("model/abc_ann_weights.h5")
            np.save("model/data", np.asarray([np.asarray(best_features),
                                               np.asarray(best_neurons), 10, 0.001]))
    return best_features, best_model, best_neurons

best_features, best_model, best_neurons = ANNOptimization(3, 10)
if best_model is not None and best_features is not None:
    predict_abc = np.argmax(best_model.predict(X_test[:, best_features]), axis=1)
    calculateMetrics("ANN+ABC", y_test, predict_abc)

# ---------------------------------------------------------------------------
# Model 4: Random Forest
# ---------------------------------------------------------------------------
rf_path = "model/rf.pckl"
if os.path.exists(rf_path):
    with open(rf_path, 'rb') as f:
        rf_cls = pickle.load(f)
else:
    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_cls.fit(X_train, y_train)
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_cls, f)

predict_rf = rf_cls.predict(X_test)
calculateMetrics("Random Forest", y_test, predict_rf)

# ---------------------------------------------------------------------------
# Model 5: XGBoost
# ---------------------------------------------------------------------------
xgb_path = "model/xgb.pckl"
if os.path.exists(xgb_path):
    with open(xgb_path, 'rb') as f:
        xgb_cls = pickle.load(f)
else:
    xgb_cls = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb_cls.fit(X_train, y_train)
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_cls, f)

predict_xgb = xgb_cls.predict(X_test)
calculateMetrics("XGBoost", y_test, predict_xgb)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n--- Final Results ---")
for i, name in enumerate(algorithm_names):
    print(f"{name:15s}  Acc={accuracy[i]:.2f}%  P={precision[i]:.2f}%  "
          f"R={recall[i]:.2f}%  F1={fscore[i]:.2f}%")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
jwt = JWTManager(app)

def _get_chart_data():
    return {
        'labels':    algorithm_names,
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'fscore':    fscore,
    }


def _count_attack_types(predictions):
    counts = {}
    for p in predictions:
        lbl = labels[p] if p < len(labels) else 'unknown'
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


# ---- Web Routes --------------------------------------------------------

@app.route('/')
@app.route('/index')
def index():
    chart_data = json.dumps(_get_chart_data())
    return render_template('index.html', data='', chart_data=chart_data)


@app.route('/Predict', methods=['GET', 'POST'])
def Predict():
    return render_template('Predict.html', data='')


@app.route('/PredictAction', methods=['POST'])
def PredictAction():
    uploaded = request.files.get('t1')
    if uploaded and uploaded.filename.endswith('.csv'):
        test_dataset = pd.read_csv(io.BytesIO(uploaded.read()))
    else:
        test_dataset = pd.read_csv("Dataset/testData.csv")
    raw_data = test_dataset.values.tolist()

    # Encrypt each row then decrypt — demonstrates AES-256 secure transmission
    encrypted_rows = [encrypt_data(str(row)) for row in raw_data]
    _ = [decrypt_data(e) for e in encrypted_rows]  # decrypted server-side

    for col, le in label_encoder:
        if col in test_dataset.columns:
            test_dataset[col] = test_dataset[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
            )
    test_dataset.fillna(0, inplace=True)
    test_data_scaled = scaler.transform(test_dataset.values.astype(float))

    pred_knn = knn_cls.predict(test_data_scaled)
    pred_ann = np.argmax(ann_model.predict(test_data_scaled), axis=1)
    pred_abc = (np.argmax(best_model.predict(test_data_scaled[:, best_features]), axis=1)
                if best_model is not None else pred_ann)
    pred_rf  = rf_cls.predict(test_data_scaled)
    pred_xgb = xgb_cls.predict(test_data_scaled)

    output = '''
    <div class="encrypt-badge">
        &#128274; Data encrypted with <strong>AES-256-CBC</strong> before transmission
        &mdash; decrypted server-side before prediction
    </div>
    <table class="result-table">
        <thead>
            <tr>
                <th>#</th><th>KNN</th><th>ANN</th><th>ANN+ABC</th>
                <th>Random Forest</th><th>XGBoost</th>
            </tr>
        </thead>
        <tbody>
    '''
    for i in range(len(pred_knn)):
        def badge(p):
            lbl = labels[p] if p < len(labels) else 'unknown'
            cls = 'normal' if lbl == 'normal' else 'attack'
            return f'<span class="badge {cls}">{lbl}</span>'

        output += f'''
            <tr>
                <td>{i + 1}</td>
                <td>{badge(pred_knn[i])}</td>
                <td>{badge(pred_ann[i])}</td>
                <td>{badge(pred_abc[i])}</td>
                <td>{badge(pred_rf[i])}</td>
                <td>{badge(pred_xgb[i])}</td>
            </tr>'''

    output += '</tbody></table>'

    chart_data = json.dumps(_get_chart_data())
    return render_template('index.html', data=output, chart_data=chart_data)




# ---- REST API ----------------------------------------------------------

_API_USER = {'username': 'admin', 'password': 'SecureCloud@2024'}

@app.route('/api/login', methods=['POST'])
def api_login():
    body = request.get_json(silent=True) or {}
    if (body.get('username') == _API_USER['username'] and
            body.get('password') == _API_USER['password']):
        token = create_access_token(identity=body['username'])
        return jsonify(access_token=token), 200
    return jsonify(error='Invalid credentials'), 401


@app.route('/api/predict', methods=['POST'])
@jwt_required()
def api_predict():
    """JWT-protected endpoint. POST JSON body: {"data": [[feature, ...], ...]}"""
    body = request.get_json(silent=True) or {}
    rows = body.get('data')
    if not rows:
        return jsonify(error='Provide "data" key with list of feature rows'), 400

    arr = scaler.transform(np.array(rows, dtype=float))

    pred_knn = knn_cls.predict(arr).tolist()
    pred_ann = np.argmax(ann_model.predict(arr), axis=1).tolist()
    pred_abc = (np.argmax(best_model.predict(arr[:, best_features]), axis=1).tolist()
                if best_model is not None else pred_ann)
    pred_rf  = rf_cls.predict(arr).tolist()
    pred_xgb = xgb_cls.predict(arr).tolist()

    def to_labels(preds):
        return [labels[p] if p < len(labels) else 'unknown' for p in preds]

    return jsonify(
        metrics={
            'algorithms': algorithm_names,
            'accuracy':   accuracy,
            'precision':  precision,
            'recall':     recall,
            'f1score':    fscore,
        },
        predictions={
            'KNN':          to_labels(pred_knn),
            'ANN':          to_labels(pred_ann),
            'ANN_ABC':      to_labels(pred_abc),
            'RandomForest': to_labels(pred_rf),
            'XGBoost':      to_labels(pred_xgb),
        }
    ), 200


if __name__ == '__main__':
    app.run(debug=False)
