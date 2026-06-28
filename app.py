import streamlit as st
import pandas as pd
import joblib

# Load models
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")

st.title("Student Clustering App")

# Example Inputs (adjust based on dataset)
age = st.number_input("Age", 15, 30, 20)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Income", 0, 100000, 20000)

# Add ALL features used in training
# IMPORTANT: match column names exactly

if st.button("Predict Cluster"):

    input_dict = {
        "age": age,
        "gender": gender,
        "income": income
    }

    # Step 1: Create dataframe
    input_df = pd.DataFrame([input_dict])

    # Step 2: Encode categorical columns (FIXED)
    for col in encoders:
        if col in input_df.columns:

            le = encoders[col]

            val = str(input_df[col].iloc[0]).strip().lower()
            classes = list(le.classes_)
            classes_lower = [str(c).lower() for c in classes]

            if val in classes_lower:
                matched_val = classes[classes_lower.index(val)]
                input_df[col] = le.transform([matched_val])[0]
            else:
                input_df[col] = 0   # unseen label safe handling

    # 🚨 DEBUG (optional but powerful)
    gender_map = {
    "male": 1,
    "female": 0
}

input_df["gender"] = input_df["gender"].astype(str).str.strip().str.lower()
input_df["gender"] = input_df["gender"].map(gender_map)

# Handle unknown values
input_df["gender"] = input_df["gender"].fillna(0)

    # Step 3: Add missing columns
input_df_full = pd.DataFrame(columns=features)

# Fill with zeros
input_df_full.loc[0] = 0

# Update only available inputs
for col in input_df.columns:
    if col in input_df_full.columns:
        input_df_full[col] = input_df[col]

# Now safe
input_df = input_df_full.copy()

    # 🔥 Step 5: Convert to float (NOW SAFE)
input_df = input_df.astype(float)

    # Step 6: Scale
input_scaled = scaler.transform(input_df)

    # Step 7: Predict
cluster = model.predict(input_scaled)

st.success(f"Assigned Cluster: {cluster[0]}")