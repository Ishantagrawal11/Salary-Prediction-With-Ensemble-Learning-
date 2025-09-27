import streamlit as st
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt
st.title("💼 Salary Prediction Model")
st.divider()
st.write("This app estimates the salary of a company employee based on years of experience , Department and job rate")

st.markdown("<h2 style='color:#2E86C1;'>📋 Enter Employee Details</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    years = st.number_input("🕒 Years of Experience", value=1, step=1, min_value=0)
    jobrate = st.number_input("⭐ Job Rate", value=3.5, step=0.5, min_value=0.0)

with col2:
    department = st.selectbox(
        "🏢 Select Department (for display only)",
        ["HR", "IT", "Finance", "Marketing", "Sales", "Operations"]
    )
try:
    model = joblib.load("linearmodel.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'linearmodel.pkl' not found. Please check the path.")
    st.stop()

st.divider()
predict_btn = st.button("🚀 Predict Salary")
clear_btn = st.button("🔄 Clear Inputs")

if predict_btn:
    with st.spinner("🔍 Analyzing data and predicting salary..."):
        time.sleep(1.5)
        x_input = np.array([[years, jobrate]], dtype=float)
        try:
            prediction_array = model.predict(x_input)
            if prediction_array is not None and len(prediction_array) > 0:
                prediction = float(prediction_array[0])
            else:
                st.error("❌ Model returned no prediction. Check your model.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.stop()
        st.success("✅ Prediction complete!")
        st.balloons()
        # st.snow()
        st.subheader("📊 Employee Summary")
        st.write(f"Years of Experience {years}")
        st.write(f"Job Rate {jobrate}")
        st.write(f"Department {department}")
        st.markdown(f"<h2 style='color:green;'>💰 Predicted Salary: ₹ {prediction:,.2f}</h2>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(["Predicted Salary"], [prediction], color="skyblue")
        ax.set_ylabel("Salary (₹)")
        st.pyplot(fig)
if not predict_btn and not clear_btn:
    st.info("ℹ️ Please enter details and press Predict Salary")
