# PerishGuardAI

## AI-Assisted Fruit Freshness Monitoring and Food Waste Tracking

### Problem
Fruit sellers and small retailers struggle to identify spoiled fruits consistently and monitor food waste and financial losses.

### Solution
PerishGuardAI is a web application combining AI-assisted fruit freshness classification with digital inspection records and food waste tracking.

### Features
- AI Fruit Scanner: Classifies banana, mango and tomato images as fresh or spoiled.
- Batch Inspection: Reviews inspection records using batch IDs.
- Waste Tracker: Records wasted quantities, reasons and estimated financial losses.
- Dashboard: Displays inspection summaries and recorded waste.

### Technology Stack
- Python
- TensorFlow and Keras
- Streamlit
- NumPy and Pandas
- SQLite
- Pillow
- ReportLab

### AI Model
A fine-tuned CNN model supports three fruits:
- Banana
- Mango
- Tomato

### Run Locally
Install dependencies:

pip install -r requirements.txt

Start the application:

python -m streamlit run app/app.py

### Disclaimer
PerishGuardAI is an academic prototype. AI predictions require human review and do not establish food safety, hidden damage or remaining shelf life.
