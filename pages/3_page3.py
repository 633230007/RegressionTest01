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

lottie_url_hello = "https://lottie.host/3f647b41-61bf-4d39-93c3-0433420604cc/8NtmEbAWmO.json"
lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello, key="hello")