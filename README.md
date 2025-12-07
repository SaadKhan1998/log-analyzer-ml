# Python-Based Log Analysis & Anomaly Detection Tool
# Python | Machine Learning | FastAPI | Pandas

A lightweight ML-powered system that processes raw application logs, extracts structured events, and identifies anomalies using **Isolation Forest**.  
Includes REST endpoints for:

- Training ML model on logs  
- Running anomaly detection  
- Retrieving structured insights  

This project is similar to microchip-level diagnostic tools and AI-assisted debugging workflows.

---

# Features

✔ Extracts structured events from raw logs  
✔ Multiple feature engineering strategies (log level, message length, frequency patterns)  
✔ ML-based anomaly detection (Isolation Forest)  
✔ FastAPI REST endpoints  
✔ Designed for scalable debugging + automated monitoring  

---

# Project Structure

log-analyzer-ml/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── utils.py
│   ├── ml/
│   │   ├── feature_extractor.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── detector.py
│   └── samples/
│       └── sample_logs.txt
│
├── requirements.txt
└── README.md


---

# Run Locally

# 1. Install dependencies
```bash
pip install -r requirements.txt

2. Start FastAPI server
uvicorn app.main:app --reload

Open Swagger UI:
 http://127.0.0.1:8000/docs

 API Endpoints
POST /train

Train model on provided log text.

POST /analyze

Detect anomalies + return summary statistics.

🛠 Future Enhancements

Transformer-based log sequence modeling

Sliding window temporal features

Log template extraction (Drain, Spell)

Scalable Kafka ingestion

Grafana dashboard for real-time monitoring

License

MIT License

Author

Muhammad Saad Majeed
AI, Data & Embedded Systems Engineer


---

#  Done!

If you want, I can also:

Add Docker support  
Add unit tests (PyTest)  
Add a lightweight frontend (React or Streamlit)  
Add GitHub Actions CI/CD  
Polish README with badges & diagrams  

Just tell me!

You said: