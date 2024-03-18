import streamlit as st
import random_forest as rf
# import pickle
import os
import base64

from svm import SVM
from random_forest import RandomForestClassifier
from random_forest import DecisionTreeClassifier
from random_forest import DecisionNode
from functions import predict_for_rf, predict_for_svm, get_df, preprocess_for_svm

script_directory = os.path.abspath(os.path.dirname(__file__))

logo_path = os.path.join(script_directory, "logo.png")

st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px; background-color: #f2f2f2;">
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}" alt="Logo" style="height: 50px; width: 50px; margin-right: 10px;">
        </div>
        <div>
            <p style="margin: 0; font-size: 18px;">Nama: Nauval Ahmad Zidane</p>
            <p style="margin: 0; font-size: 18px;">NIM: 09021382025163</p>
        </div>
    </div>
""", unsafe_allow_html=True)


old_n_trees_model = 100

if __name__ == "__main__":
#     try:
#         models = pickle.load(open("./models.pickle", "rb"))
#         svm_model = models['svm']
#         rf_model = models['rf']
#         label_encoders = models['label_encoders']
#         std_scalers = models['std_scalers']
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         st.stop()

    st.title("KOMPARASI KINERJA ALGORITMA RANDOM FOREST DAN SUPPORT VECTOR MACHINE DALAM ANALISIS POTENSI TURNOVER KARYAWAN")

    uploaded_file = st.file_uploader("Upload Data", type=["csv"])

    st.header("Random Forest Parameter")

    n_trees = st.slider("Jumlah pohon", 10, 10000, 100)

    st.header("SVM Parameter")
    lambda_param = st.number_input("Nilai C (lambda)", min_value=0.00001, max_value=1000.00, value=0.01, format="%f")

    submitted = st.button("Submit")

    if submitted:
        if uploaded_file is None:
            st.warning("Please upload a file")
        else:
            new_rf_model = rf_model
            new_svm_model = svm_model
            scores_rf = None
            scores_svm = None
            with st.status("Loading", state='running') as status:
                df = get_df(uploaded_file, label_encoders, std_scalers)
                X = df.drop('event', axis=1).values
                y = df['event'].values

                df_svm = preprocess_for_svm(df, std_scalers)
                X_svm = df.drop('event', axis=1).values
                y_svm = df['event'].values

                scores_rf = predict_for_rf(new_rf_model, X, y, n_trees - old_n_trees_model, uploaded_file)
                scores_svm = predict_for_svm(new_svm_model, X_svm, y_svm, 10000, lambda_param, uploaded_file)
                status.update(label="Completed", state="complete")

            st.markdown("#### Random Forest Scores")
            st.table(scores_rf)

            st.markdown("#### SVM Scores")
            st.table(scores_svm)