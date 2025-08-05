import streamlit as st
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
import plotly.express as px


st.set_page_config(page_title="Titanic Classifier", layout="wide")
st.title('Titanic Classifier')
st.write("## Работа с датасетом Titanic")

df = pd.read_csv("dfFM.csv")

st.subheader("🔎 Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

# Graphics
st.subheader("📊Визуализация данных")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="embarked", color="survived", barmode="group", title="Распределение выживших по городам")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.scatter(df, x="age", y="sibsp", color="survived", title="Возраст vs количество родственников")
    st.plotly_chart(fig2, use_container_width=True)

