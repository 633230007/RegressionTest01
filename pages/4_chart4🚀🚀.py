import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

st.title("Student Stress Factors")
st.header("Student Stress Factors")

df=pd.read_csv('./data/Factors.csv')
st.write(df.head(10))

x = df.iloc[:, :-1]  # ยกเว้นคอลัมน์สุดท้าย
y = df.iloc[:, -1]   # คอลัมน์สุดท้ายเป็น target

#st.line_chart(df)
#st.line_chart(df, x="Sleep Quality", y="headaches", color="Student Stress Factors")
#st.line_chart(df, x="Sleep Quality", y=["headaches", "Student Stress Factors"], color=["#FF0000", "#0000FF"])

#x=df[['Kindly Rate your Sleep Quality', 'How many times a week do you suffer headaches']]
#y=df['Student Stress Factors']
#pf=PolynomialFeatures(degree=3)
#x_poly=pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

Dtmodel=DecisionTreeClassifier(criterion='gini')
Dtmodel.fit(x_train, y_train)

x1=st.number_input("กรุณาป้อนข้อมูล Sleep Quality:")
x2=st.number_input("กรุณาป้อนข้อมูล headaches:")
x3=st.number_input("กรุณาป้อนข้อมูล academic performance:")
x4=st.number_input("กรุณาป้อนข้อมูล study load:")
x5=st.number_input("กรุณาป้อนข้อมูล extracurricular activities:")
x6=st.number_input("กรุณาป้อนข้อมูล stress levels:")

if st.button("พยากรณ์ข้อมูล"):
    x_input=[[x1, x2, x3, x4, x5, x6]]
    y_predict=Dtmodel.predict(pf.fit_transform(x_input))
    st.write(y_predict)
    st.button("ไม่พยากรณ์ข้อมูล")
else:
    st.button("ไม่พยากรณ์ข้อมูล")
