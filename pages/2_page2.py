import json
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/acd0c339-c8cb-4796-930f-5c7844bbbc1c/eiEBu3mxcA.json"
lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello, key="hello")