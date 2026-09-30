"""Training pipeline for SecureCloudML.

Loads the KDD Cup dataset, preprocesses it, trains/loads the 5 comparison
models (KNN, ANN, ANN+ABC, Random Forest, XGBoost), and persists everything
`app.py` needs at inference time: the fitted scaler, label encoders, model
artifacts, and a metrics.json used to render the dashboard.

Run directly to (re)train from scratch:
    python train.py
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
try:
    from keras.utils import to_categorical
except ImportError:
    from keras.utils.np_utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

from config import DATASET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MODEL_DIR = "model"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_dataset(path: str = DATASET_PATH) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = pd.read_csv(path)
    labels = np.unique(dataset["Attack_Type"])
    return dataset, labels


def encode_categorical_columns(dataset: pd.DataFrame) -> list[list[Any]]:
    """Label-encode every object-typed column in place, returning [name, encoder] pairs."""
    label_encoders: list[list[Any]] = []
    for column in dataset.columns:
        if dataset[column].dtype == "object" or dataset[column].dtype.name in ("object", "string", "str") or pd.api.types.is_string_dtype(dataset[column]):
            encoder = LabelEncoder()
            dataset[column] = encoder.fit_transform(dataset[column].astype(str))
            label_encoders.append([column, encoder])
    dataset.fillna(0, inplace=True)
    return label_encoders


def calculate_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "fscore": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    }
    logger.info("%s -> accuracy=%.2f precision=%.2f recall=%.2f f1=%.2f",
                name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["fscore"])
    return metrics


def train_knn(x_train, y_train, x_test) -> KNeighborsClassifier:
    path = os.path.join(MODEL_DIR, "knn.pckl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    model = KNeighborsClassifier(algorithm="ball_tree")
    model.fit(x_train, y_train)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def train_random_forest(x_train, y_train) -> RandomForestClassifier:
    path = os.path.join(MODEL_DIR, "rf.pckl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(x_train, y_train)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def train_xgboost(x_train, y_train, num_classes: int) -> XGBClassifier:
    path = os.path.join(MODEL_DIR, "xgb.pckl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    model = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        num_class=num_classes,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def train_ann(x_train, y_train_categorical, x_test, y_test_categorical) -> Sequential:
    path = os.path.join(MODEL_DIR, "ann_weights.hdf5")
    model = Sequential()
    model.add(Dense(32, input_shape=(x_train.shape[1],)))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(y_train_categorical.shape[1], activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if os.path.exists(path):
        model.load_weights(path)
    else:
        checkpoint = ModelCheckpoint(filepath=path, verbose=1, save_best_only=True)
        model.fit(
            x_train, y_train_categorical, batch_size=32, epochs=10,
            validation_data=(x_test, y_test_categorical), callbacks=[checkpoint], verbose=1,
        )
    return model


def _fitness(features, x_train, y_train_cat, x_test, y_test_cat, neurons, epochs) -> tuple[float, Sequential]:
    model = Sequential()
    model.add(Dense(neurons[0], input_shape=(len(features),)))
    model.add(Dense(neurons[1], activation="relu"))
    model.add(Dense(y_train_cat.shape[1], activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train[:, features], y_train_cat, batch_size=32, epochs=epochs,
              validation_data=(x_test[:, features], y_test_cat), verbose=0)
    predict = np.argmax(model.predict(x_test[:, features], verbose=0), axis=1)
    true_label = np.argmax(y_test_cat, axis=1)
    return accuracy_score(true_label, predict), model


def train_ann_with_abc(x_train, y_train_cat, x_test, y_test_cat, population_size=3, max_iterations=10):
    """Simplified Artificial Bee Colony search over feature subsets for the ANN."""
    weights_path = os.path.join(MODEL_DIR, "abc_ann_weights.h5")
    data_path = os.path.join(MODEL_DIR, "data.npy")
    neurons = [64, 32]

    if os.path.exists(weights_path) and os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        best_features, best_neurons, best_epoch, best_lr = data[0], data[1], data[2], data[3]
        model = Sequential()
        model.add(Dense(int(best_neurons[0]), input_shape=(len(best_features),)))
        model.add(Dense(int(best_neurons[1]), activation="relu"))
        model.add(Dense(int(y_train_cat.shape[1]), activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.load_weights(weights_path)
        return best_features, model, best_neurons, best_epoch, best_lr

    n_features = x_train.shape[1]
    population = [
        random.sample(range(n_features), int(n_features * 0.8)) for _ in range(population_size)
    ]
    best_features, best_model, best_fitness = None, None, -1.0

    for _ in range(max_iterations):
        improved = False
        for features in population:
            fitness, model = _fitness(features, x_train, y_train_cat, x_test, y_test_cat, neurons, 4)
            if fitness > best_fitness:
                best_features, best_model, best_fitness = features, model, fitness

            new_features = features.copy()
            new_features[random.randint(0, len(new_features) - 1)] = random.randint(0, n_features - 1)
            new_fitness, new_model = _fitness(new_features, x_train, y_train_cat, x_test, y_test_cat, neurons, 10)
            if new_fitness > best_fitness:
                best_features, best_model, best_fitness = new_features, new_model, new_fitness
                improved = True
                break
        if improved:
            break

    best_model.save_weights(weights_path)
    data = np.asarray([np.asarray(best_features), np.asarray(neurons), 10, 0.001], dtype=object)
    np.save(os.path.join(MODEL_DIR, "data"), data)
    return best_features, best_model, neurons, 10, 0.001


def run_training(dataset_path: str = DATASET_PATH) -> dict[str, dict[str, float]]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Loading dataset from %s", dataset_path)
    dataset, labels = load_dataset(dataset_path)

    label_encoders = encode_categorical_columns(dataset)

    y = dataset["Attack_Type"].to_numpy()
    x = dataset.drop(columns=["Attack_Type"]).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )

    # Fit the scaler on the training split only to avoid leaking test-set
    # statistics into the transform (the original notebook fit it on all of X).
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    with open(os.path.join(MODEL_DIR, "scaler.pckl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "label_encoders.pckl"), "wb") as f:
        pickle.dump(label_encoders, f)
    with open(os.path.join(MODEL_DIR, "labels.pckl"), "wb") as f:
        pickle.dump(labels, f)

    metrics: dict[str, dict[str, float]] = {}

    knn = train_knn(x_train, y_train, x_test)
    metrics["KNN"] = calculate_metrics("KNN", y_test, knn.predict(x_test))

    rf = train_random_forest(x_train, y_train)
    metrics["Random Forest"] = calculate_metrics("Random Forest", y_test, rf.predict(x_test))

    xgb = train_xgboost(x_train, y_train, num_classes=len(labels))
    metrics["XGBoost"] = calculate_metrics("XGBoost", y_test, xgb.predict(x_test))

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    ann = train_ann(x_train, y_train_cat, x_test, y_test_cat)
    ann_pred = np.argmax(ann.predict(x_test, verbose=0), axis=1)
    metrics["ANN"] = calculate_metrics("ANN", y_test, ann_pred)

    best_features, abc_model, *_ = train_ann_with_abc(x_train, y_train_cat, x_test, y_test_cat)
    abc_pred = np.argmax(abc_model.predict(x_test[:, best_features], verbose=0), axis=1)
    metrics["ANN with ABC"] = calculate_metrics("ANN with ABC", y_test, abc_pred)
    with open(os.path.join(MODEL_DIR, "best_features.pckl"), "wb") as f:
        pickle.dump(best_features, f)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete. Metrics saved to model/metrics.json")
    return metrics


def ensure_trained(dataset_path: str = DATASET_PATH) -> None:
    """Train once if no artifacts exist yet; otherwise this is a no-op."""
    if not os.path.exists(os.path.join(MODEL_DIR, "metrics.json")):
        run_training(dataset_path)


def load_artifacts() -> dict[str, Any]:
    """Load every artifact `app.py` needs for inference: scaler, encoders, labels, and all 5 models."""
    def _load(name: str) -> Any:
        with open(os.path.join(MODEL_DIR, name), "rb") as f:
            return pickle.load(f)

    scaler = _load("scaler.pckl")
    label_encoders = _load("label_encoders.pckl")
    labels = _load("labels.pckl")
    best_features = _load("best_features.pckl")
    knn = _load("knn.pckl")
    rf = _load("rf.pckl")
    xgb = _load("xgb.pckl")

    num_features = scaler.n_features_in_
    num_classes = len(labels)

    ann = Sequential([
        Dense(32, input_shape=(num_features,)),
        Dense(16, activation="relu"),
        Dense(int(num_classes), activation="softmax"),
    ])
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    ann.load_weights(os.path.join(MODEL_DIR, "ann_weights.hdf5"))

    abc_data = np.load(os.path.join(MODEL_DIR, "data.npy"), allow_pickle=True)
    abc_neurons = abc_data[1]
    abc_ann = Sequential([
        Dense(int(abc_neurons[0]), input_shape=(len(best_features),)),
        Dense(int(abc_neurons[1]), activation="relu"),
        Dense(int(num_classes), activation="softmax"),
    ])
    abc_ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    abc_ann.load_weights(os.path.join(MODEL_DIR, "abc_ann_weights.h5"))

    return {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "labels": labels,
        "best_features": best_features,
        "models": {
            "KNN": knn,
            "Random Forest": rf,
            "XGBoost": xgb,
            "ANN": ann,
            "ANN with ABC": abc_ann,
        },
    }


if __name__ == "__main__":
    run_training()
