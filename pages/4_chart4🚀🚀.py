import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

st.title("Student Stress Factors")
st.header("Student Stress Factors")

df=pd.read_csv('./data/Student Stress Factors.csv')
st.write(df.head(10))

#st.line_chart(df)
#st.line_chart(df, x="Sleep Quality", y="headaches", color="Student Stress Factors")
st.line_chart(df, x="Sleep Quality", y=["headaches", "Student Stress Factors"], color=["#FF0000", "#0000FF"]  # Optional
)

x=df[['Sleep Quality','headaches']]
y=df['Student Stress Factors']
pf=PolynomialFeatures(degree=3)
x_poly=pf.fit_transform(x)

x_train,x_test,y_train,y_test =train_test_split(x_poly,y,random_state=0)

modelRegress=LinearRegression()
modelRegress.fit(x_train,y_train)
x1=st.number_input("กรุณาป้อนข้อมูล Sleep Quality:")
x2=st.number_input("กรุณาป้อนข้อมูล headaches:")

if st.button("พยากรณ์ข้อมูล"):
    x_input=[[x1,x2]]
    y_predict=modelRegress.predict(pf.fit_transform(x_input))
    st.write(y_predict)
    st.button("ไม่พยากรณ์ข้อมูล")
else:
    st.button("ไม่พยากรณ์ข้อมูล")
