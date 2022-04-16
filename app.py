import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import yfinance as yf 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import datetime


stocks =("AAPL","GOOG","MSFT","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS","INFY.NS","BTC-INR")
st.set_page_config(layout="wide")
def side_display():
    with st.sidebar:
        st.title("Dashboard😊")
        st.subheader("To compare the stocks of more than one company 📊")
        dropdown= st.multiselect('Pick your assets',stocks)

        start_date = st.date_input("Select Starting Date",datetime.date.today())
        end_date = st.date_input("Select Ending Date",datetime.date.today())
    start_date =None if start_date>=datetime.date.today() else start_date
    end_date =None if end_date>=datetime.date.today() else end_date
    return start_date,end_date,dropdown

@st.cache
def load_data(ticker,start,end):
    #st.download_button('Download',data.to_csv())
    data = yf.download(ticker,start,end)
    data.reset_index(inplace=True) 
    return data

def relativeret(df):
    rel  =df.pct_change()
    cumret =(1+rel).cumprod()-1
    cumret = cumret.fillna(0)
    return cumret

st.title("Stock Price Prediction ")
st.subheader("- using Random Forest Regressor🚀📈ᵇʸ ˢᵘˢʰᵃⁿᵗ ᴮⁱˢʰᵗ ")


kpi1, kpi2,kpi3= st.columns(3)
kpi1.metric(label="$OPEN",value=200.22,delta=-1.4)
kpi2.metric(label="$CLOSE",value=250.71,delta=+2.1)
kpi3.metric(label="$AAPP",value=195.63,delta=+0.2)

selected_stocks = st.selectbox("Select Dataset for prediction ",stocks)

n_years = st.slider("Years of prediction : ",10,20)
period = n_years *365

start_date,end_date,dropdown=side_display()
data_load_state = st.text("Load data................")
data = load_data(selected_stocks,start_date,end_date)
data_load_state.text("Loading done...!!")
if len(dropdown)>0:
    df = relativeret(yf.download(dropdown,start_date,end_date)['Adj Close'])
    st.line_chart(df)





full_display,details_display = st.columns(2)
full_display.subheader("Raw Data")
full_display.write(data)

details_display.subheader('Data Description')
details_display.write(data.describe())

dataploy_open,dataplot_close = st.columns(2)
def plot_open_dataset():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="Stock Open",line=dict(color='green')))
    fig.layout.update(title_text ="Open Price",xaxis_rangeslider_visible=True)
    dataploy_open.plotly_chart(fig)

def plot_close_dataset():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="Stock Close",line=dict(color='red')))
    fig.layout.update(title_text ="Close Price",xaxis_rangeslider_visible=True)
    dataplot_close.plotly_chart(fig)

plot_open_dataset()
plot_close_dataset()

raw_data_col,moving_average_col = st.columns(2)

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="Stock Close"))
    fig.layout.update(title_text ="Time series data",xaxis_rangeslider_visible=True)
    raw_data_col.plotly_chart(fig)

def moving_average(data):
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="Stock Close"))
    fig.add_trace(go.Scatter(x=data['Date'],y=ma100,name="Moving average for 100 values"))
    fig.add_trace(go.Scatter(x=data['Date'],y=ma200,name="Moving average for 200 values"))
    fig.layout.update(title_text ="Moving Average Data",xaxis_rangeslider_visible=True)
    moving_average_col.plotly_chart(fig)
def plot_pridicted_graph(data,y_test,y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= data['Date'], y=y_test, name="Original Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=y_pred, name="Predicted Price", line=dict(color='grey')))
    fig.layout.update(title_text ="Prediction graph",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data)  
moving_average(data)
df = data.drop(['Date','Adj Close'], axis=1)

x = data.iloc[:,1:4].values
y = data.iloc[:,4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regression = RandomForestRegressor(n_estimators=10, random_state=0)
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

st.subheader("Here test data and Predicted data you can see how similar they are !!")

full_display,details_display = st.columns(2)
full_display.subheader("Test Data")
full_display.write(y_test.reshape(len(y_test), 1))

details_display.subheader('Predicted Data')
details_display.write(y_pred.reshape(len(y_pred), 1))

st.subheader("R2 Score for model : ")
st.text(r2_score(y_test,y_pred))

st.subheader("want to predict close price for any open price ?")
open_input = st.text_input("Open ",198.779999)
high_input = st.text_input("High ",199.990005)
low_input = st.text_input("Low ",197.619995)

input_values = []
final_input = []
input_values.append(open_input)
input_values.append(high_input)
input_values.append(low_input)
final_input.append(input_values)

final_input=sc.transform(final_input)


st.subheader('Predicted Closing Price ')
st.write(regression.predict(final_input))

st.subheader("R2 Score for model : ")
st.text(r2_score(y_test,y_pred))

# st.subheader("prediction")
# fig2 =plt.figure(figsize=(50,12))
# plt.plot(y_test,'g',label='Original Price')
# plt.plot(y_pred,'y',label='Predicted Price')
# plt.xlabel('Date')
# plt.ylabel('Close')
# plt.legend()
# st.pyplot(fig2)

plot_pridicted_graph(data, y_test, y_pred)
