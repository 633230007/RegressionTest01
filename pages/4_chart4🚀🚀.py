import json
import time
import requests
import streamlit as st
import pandas as pd

st.header("Show Data breast_cancer")
df=pd.read_csv("./data/breast_cancer.csv")
st.write(df.head(10))

st.header("Show Chart")
st.line_chart(
   df, x="breast_cancer", y=["interest_rate", "unemployment_rate"], color=["#FF0000", "#0000FF"]  # Optional
)