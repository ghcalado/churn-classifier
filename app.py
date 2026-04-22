import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from src.prepare import preparar_dados

st.set_page_config(page_title="Preditor de Attrition", layout="wide")

X, y = preparar_dados()

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

modelo = RandomForestClassifier(random_state=42, class_weight='balanced')
modelo.fit(X_train, y_train)

st.title("Dashboard de Previsao de Attrition")

st.sidebar.header("Configuracoes do Funcionario")
inputs = {}
for col in X.columns:
    if X[col].nunique() > 2:
        inputs[col] = st.sidebar.slider(
            f"{col}",
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    else:
        inputs[col] = st.sidebar.selectbox(f"{col}", [0, 1])

col_main, col_metrics = st.columns([2, 1])

with col_main:
    if st.button("Executar Previsao"):
        input_df = pd.DataFrame([inputs])
        pred = modelo.predict(input_df)[0]
        prob = modelo.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"Resultado: ALTO RISCO de Saida ({prob*100:.2f}%)")
        else:
            st.success(f"Resultado: BAIXO RISCO de Saida ({prob*100:.2f}%)")

        st.subheader("Fatores Determinantes (Feature Importance)")
        importancias = pd.Series(modelo.feature_importances_, index=X.columns)
        importancias_top = importancias.sort_values(ascending=False).head(10)
        st.bar_chart(importancias_top)

with col_metrics:
    st.subheader("Performance do Modelo")
    previsoes = modelo.predict(X_test)
    acc = accuracy_score(y_test, previsoes)
    st.metric("Acuracia Geral", f"{acc:.2f}")

    st.text("Relatorio de Classificacao:")
    st.code(classification_report(y_test, previsoes))

st.divider()
st.subheader("Dados Base (Balanceamento SMOTE)")
st.write(y_res.value_counts())
