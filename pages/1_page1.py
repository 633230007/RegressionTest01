import json
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Kairung Datascience Project",
    page_icon= ":bar_chart:",
)
st.sidebar.success("เลือกรายการด้านบน.")

st.write("# การพยากรณ์ราคาหุ้น! 👋  🧑🏽‍🏫 ")
st.write("### 1.หลักการและเหตุผล")
st.write("### 2.วัตถุประสงค์")
st.balloons()