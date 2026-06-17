# SecureCloudML — Enhancing Data Transmission Security in Cloud Using Machine Learning

A Flask-based web application that detects network intrusions using 5 machine learning algorithms on the KDD Cup 99 dataset, with AES-256-CBC encryption securing all data transmissions.

---

## Project Overview

This project addresses the problem of network security in cloud environments by:
- Classifying network traffic into **5 categories**: Normal, DoS, Probe, R2L, U2R
- Comparing **5 ML models** side-by-side on every prediction
- Encrypting all data with **AES-256-CBC** before processing (demonstrating secure cloud transmission)

---

## ML Models & Accuracy

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| KNN | 96.35% | 69.49% | 67.57% | 68.48% |
| ANN | 97.52% | 69.36% | 73.12% | 71.04% |
| ANN + ABC | 99.11% | 85.83% | 95.75% | 89.49% |
| Random Forest | 99.83% | 94.60% | 92.06% | 93.28% |
| **XGBoost** | **99.92%** | **97.43%** | **97.32%** | **97.37%** |

> ABC = Artificial Bee Colony optimization for feature selection

---

## Tech Stack

- **Backend**: Python 3.13, Flask 3.1
- **ML**: scikit-learn, TensorFlow/Keras, XGBoost
- **Encryption**: AES-256-CBC (pycryptodome)
- **Frontend**: HTML5, CSS3, Chart.js
- **API**: REST + JWT authentication
- **Dataset**: KDD Cup 1999 (125,973 samples, 41 features)

---

## Project Structure

```
SecureCloudML/
├── CloudSecurity.py       # Main Flask app + ML training
├── encryption.py          # AES-256-CBC encrypt/decrypt
├── config.py              # App configuration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker support
├── docker-compose.yml
├── Dataset/
│   └── testData.csv       # Sample test data (11 rows, 5 attack types)
├── model/                 # Pre-trained model weights
├── static/
│   ├── style.css
│   └── images/
└── templates/
    ├── index.html         # Dashboard with charts
    └── Predict.html       # Prediction page
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SecureCloudML.git
cd SecureCloudML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download **KDD Cup 1999 dataset** and place it as:
```
Dataset/kdd_data.csv
```
The file must have an `Attack_Type` column with labels: `normal`, `dos`, `probe`, `r2l`, `u2r`.

### 4. Configure environment (optional)
Create a `.env` file:
```
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
AES_KEY=your-64-char-hex-key
```

### 5. Run the app
```bash
python CloudSecurity.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## Using Docker
```bash
docker-compose up --build
```

---

## Features

- **Dashboard** — Model performance metrics, bar chart, radar chart
- **Predict** — Upload a CSV file and get predictions from all 5 models simultaneously
- **AES-256-CBC** — Every data row is encrypted before prediction and decrypted server-side
- **REST API** — JWT-protected `/api/predict` endpoint for programmatic access

### REST API Usage
```bash
# Get JWT token (use credentials from your .env file)
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_api_username", "password": "your_api_password"}'

# Health check (no auth required)
curl http://localhost:5000/api/health

# Run prediction (replace <token> with token from /api/login)
curl -X POST http://localhost:5000/api/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"data": [[0, 1, 2, 0, 215, 45076, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 255, 255, 1, 0, 0, 0, 0, 0, 0, 0]]}'
```

---

## Test Data

Sample CSV for testing is provided at `Dataset/testData.csv` with 11 rows covering all attack types:

| Rows | Attack Type | Description |
|---|---|---|
| 1–3 | Normal | Standard HTTP/DNS traffic |
| 4–6 | DoS | SYN flood (S0 flag, serror_rate=1.0) |
| 7–8 | Probe | Port scan on private services |
| 9–10 | R2L | FTP login with failed attempts |
| 11 | U2R | Privilege escalation via Telnet |

---

## License

MIT License — free to use, modify, and distribute.
