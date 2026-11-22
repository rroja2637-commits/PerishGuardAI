import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3
import json
from PIL import Image
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "perishguard_cnn_finetuned.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"
DB_PATH = ROOT / "data" / "perishguard.db"

st.set_page_config(
    page_title="PerishGuardAI",
    page_icon="PerishGuardAI",
    layout="wide"
)

DB_PATH.parent.mkdir(exist_ok=True)

def connect():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                batch_id TEXT,
                fruit TEXT,
                ai_result TEXT,
                confidence REAL,
                human_decision TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS waste (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                batch_id TEXT,
                fruit TEXT,
                quantity REAL,
                unit TEXT,
                price REAL,
                reason TEXT,
                loss REAL
            )
        """)

initialize_database()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

def get_data(table):
    with connect() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY id DESC", conn)

def predict(image):
    model = load_model()
    classes = load_classes()
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    predictions = model.predict(arr, verbose=0)[0]
    index = int(np.argmax(predictions))
    label = classes[index]
    confidence = float(predictions[index]) * 100
    freshness = "Fresh" if label.startswith("F_") else "Spoiled"
    fruit = label.split("_", 1)[1]
    top_indices = np.argsort(predictions)[::-1][:3]
    top_results = [(classes[i], float(predictions[i]) * 100) for i in top_indices]
    return fruit, freshness, confidence, top_results

st.markdown("""
<style>
.stApp {
    background: #071525;
    color: #f1f5f9;
}
section[data-testid="stSidebar"] {
    background: #0d2033;
}
h1, h2, h3 {
    color: #f1f5f9 !important;
}
p, label, span {
    color: #d6e2ed;
}
div[data-testid="stMetric"] {
    background: #10283d;
    border: 1px solid #24445a;
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    background: #19a875;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}
.stButton > button:hover {
    background: #13865d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("PerishGuardAI PerishGuardAI")
st.sidebar.caption("Fruit Quality Monitoring System")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "AI Fruit Scanner", "Batch Inspection", "Waste Tracker"]
)

st.sidebar.divider()
st.sidebar.info("Supported fruits: Banana, Mango and Tomato")

if page == "Dashboard":
    st.title("PerishGuardAI Dashboard")

    st.markdown("""
    <div style="background:linear-gradient(120deg,#102b35,#174b3b);
                padding:25px;border-radius:15px;margin-bottom:22px;">
        <h2 style="color:#ffffff;margin:0;">Freshness insights at a glance</h2>
        <p style="color:#d5eee2;margin-top:8px;">
            Monitor fruit inspections, review freshness decisions,
            and keep track of recorded food waste.
        </p>
    </div>
    """, unsafe_allow_html=True)

    inspections = get_data("inspections")
    waste = get_data("waste")

    total = len(inspections)
    fresh = len(inspections[inspections["human_decision"] == "Fresh"]) if total else 0
    spoiled = len(inspections[inspections["human_decision"] == "Spoiled"]) if total else 0
    total_loss = float(waste["loss"].sum()) if len(waste) else 0

    st.subheader("Overview")

    a, b, c, d = st.columns(4)
    a.metric("Total Inspections", total)
    b.metric("Fresh Confirmations", fresh)
    c.metric("Spoiled Confirmations", spoiled)
    d.metric("Recorded Waste Loss", f"Rs.{total_loss:,.2f}")

    st.divider()

    st.subheader("Recent Inspection Records")
    if total:
        st.dataframe(inspections.head(10), width="stretch")
    else:
        st.info("No inspection records yet. Start by scanning a fruit image.")

    st.subheader("Recent Waste Records")
    if len(waste):
        st.dataframe(waste.head(10), width="stretch")
    else:
        st.info("No waste records have been added yet.")

elif page == "AI Fruit Scanner":
    st.title("📷 AI Fruit Scanner")
    st.write("Upload a fruit image to classify its visible condition.")

    uploaded = st.file_uploader(
        "Upload Banana, Mango or Tomato image",
        type=["jpg", "jpeg", "png"]
    )
    batch_id = st.text_input("Batch ID", value="BATCH-001")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded fruit image", width=350)

        if st.button("Analyze Fruit", width="stretch"):
            try:
                with st.spinner("Analyzing image..."):
                    fruit, result, confidence, top_results = predict(image)

                st.session_state["scan_result"] = {
                    "fruit": fruit,
                    "result": result,
                    "confidence": confidence,
                    "top": top_results,
                    "batch": batch_id
                }
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if "scan_result" in st.session_state:
        result = st.session_state["scan_result"]
        st.divider()
        st.subheader("AI Prediction")

        x, y, z = st.columns(3)
        x.metric("Fruit", result["fruit"])
        y.metric("AI Classification", result["result"])
        z.metric("Confidence", f'{result["confidence"]:.2f}%')

        st.write("Top 3 predictions")
        for label, score in result["top"]:
            st.write(f"**{label}** — {score:.2f}%")

        st.warning("Please review the image. AI output is a prediction, not a food-safety assessment.")
        decision = st.selectbox(
            "Human confirmation",
            ["Needs Further Inspection", "Fresh", "Spoiled"],
            key="human_confirmation"
        )

        if st.button("Save Inspection Record"):
            with connect() as conn:
                conn.execute("""
                    INSERT INTO inspections
                    (date, batch_id, fruit, ai_result, confidence, human_decision)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    result["batch"],
                    result["fruit"],
                    result["result"],
                    result["confidence"],
                    decision
                ))
            st.success("Inspection record saved successfully!")
            del st.session_state["scan_result"]

elif page == "Batch Inspection":
    st.title("📦 Batch Inspection Records")
    records = get_data("inspections")

    if len(records):
        search = st.text_input("Search by Batch ID")
        if search:
            records = records[
                records["batch_id"].str.contains(search, case=False, na=False)
            ]

        st.dataframe(records, width="stretch")

        st.download_button(
            "Download Inspection CSV",
            records.to_csv(index=False).encode("utf-8"),
            "inspection_records.csv",
            "text/csv"
        )
    else:
        st.info("No inspection records available. Use the AI Fruit Scanner first.")

elif page == "Waste Tracker":
    st.title("♻️ Fruit Waste Tracker")
    st.write("Manually record discarded fruit quantity and estimated financial loss.")

    with st.form("waste_form"):
        batch = st.text_input("Batch ID")
        fruit = st.selectbox("Fruit", ["Banana", "Mango", "Tomato"])
        quantity = st.number_input("Discarded Quantity", min_value=0.0, step=0.5)
        unit = st.selectbox("Unit", ["kg", "pieces", "boxes"])
        price = st.number_input("Estimated Price per Unit (Rs.)", min_value=0.0, step=1.0)
        reason = st.selectbox(
            "Reason for Disposal",
            ["Spoilage", "Physical Damage", "Overripe", "Unsold Stock", "Other"]
        )
        submitted = st.form_submit_button("Save Waste Record")

        if submitted:
            if quantity <= 0:
                st.error("Enter a quantity greater than zero.")
            else:
                loss = quantity * price
                with connect() as conn:
                    conn.execute("""
                        INSERT INTO waste
                        (date, batch_id, fruit, quantity, unit, price, reason, loss)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        batch, fruit, quantity, unit, price, reason, loss
                    ))
                st.success(f"Waste record saved. Estimated loss: Rs.{loss:,.2f}")

    st.divider()
    st.subheader("Waste History")
    waste = get_data("waste")

    if len(waste):
        st.dataframe(waste, width="stretch")
        st.download_button(
            "Download Waste CSV",
            waste.to_csv(index=False).encode("utf-8"),
            "waste_records.csv",
            "text/csv"
        )
    else:
        st.info("No waste records available yet.")

elif page == "Analytics":
    st.title("📊 Inspection and Waste Analytics")

    inspections = get_data("inspections")
    waste = get_data("waste")

    if len(inspections):
        st.subheader("Human-Confirmed Fruit Condition")
        st.bar_chart(inspections["human_decision"].value_counts())

        st.subheader("Inspections by Fruit")
        st.bar_chart(inspections["fruit"].value_counts())
    else:
        st.info("Add inspection records to view inspection analytics.")

    if len(waste):
        st.subheader("Estimated Waste Loss by Fruit")
        loss_by_fruit = waste.groupby("fruit")["loss"].sum()
        st.bar_chart(loss_by_fruit)

        st.subheader("Discarded Quantity by Fruit")
        quantity_by_fruit = waste.groupby("fruit")["quantity"].sum()
        st.bar_chart(quantity_by_fruit)
    else:
        st.info("Add waste records to view waste analytics.")

st.divider()
st.caption(
    "PerishGuardAI is an academic prototype. Its image classifier supports "
    "banana, mango and tomato only. Predictions require human review and do "
    "not establish food safety, hidden damage or remaining shelf life."
)



