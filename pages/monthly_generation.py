
import streamlit as st
import pandas as pd

from datetime import timedelta

import matplotlib.pyplot as plt 
from pmdarima.arima import auto_arima

import csv
from PIL import Image
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.subheader('Monthly Avg power generation Forecasting for upcoming 12 months')

input_file = st.file_uploader("Upload input file", type="csv")

dataS='Gen. Date'
dataW='DATE'
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')



if input_file is not None:
    #data=input_file.getvalue().decode('utf-8').splitlines()
    data = pd.read_csv(input_file,encoding= 'unicode_escape')
    if dataS in data.columns:
            print('S')
            #data = pd.read_csv(input_file,encoding= 'unicode_escape',parse_dates=['Gen. Date'])
            data['Gen. Date']=pd.to_datetime(data['Gen. Date'])
            lastdate=data['Gen. Date'].iloc[-1]#24/11/2022
            print(lastdate)
            n=12
            for i in (lastdate + timedelta(n) for n in range(365)):
                df1=pd.DataFrame({"Gen. Date":i,"Customer Name":0,	"State":0,	"Site":0,	"Section":0,"MW":0,	"Loc. No.":0,	"Gen. (kwh) DAY":0,	"Gen. (kwh) MTD":0,	"Gen. (kwh) YTD":0,	"%PLF DAY":0,	"%PLF MTD":0,	"%PLF YTD":0,	"M/C Avail.%":0,	"GF":0	,"FM":0,	"S":0,	"U":0,	"Gen Hrs.":0,	"Opr Hrs.":0},index=[i])
                data=data.append(df1)
            data = data.set_index("Gen. Date", drop=False)
            data1=data.resample('M').mean()
            data1 = data1.loc[:, ["Gen. (kwh) DAY"]]#index
            data1 = data1.asfreq("M")
                 #Training and test set
            test_months = 12
            training_set = data1.iloc[:-test_months, :]
            test_set = data1.iloc[-test_months:, :]
            model = auto_arima(training_set,
                    start_p=0,d=1,start_q=0,max_p=5,max_d=5,max_q=5,
                    start_P=0,D=1,start_Q=0,max_P=5,max_D=5,max_Q=5,
                    m=12,seasonal=True,
                    trace=True,
                    )

            #predictions
            predictions_sarimax = pd.Series(model.predict(n_periods= test_months)).rename("Power")
            predictions_sarimax.index = test_set.index                              
            
            df=predictions_sarimax.to_frame()
            st.write(df)
            df = df.reset_index()
            #df['date']=df.index
            #predictions_sarimax.to_csv('Forecasting.csv')
            csv = convert_df(df)

            st.download_button(
            "Download",
            csv,
        "file.csv",
   "text/csv",
   key='download-csv'
)
            
            
          
            
            linechart=df.plot(y='Power',x='Gen. Date',marker = 'o',
                              xlabel="Months",ylabel='Monthly Avg. Power generation in Kwh',
                              title='Monthly Avg Power Generation for upcoming 12 months',legend=None,figsize=[10,8])
           
            fig = linechart.get_figure()
            fig.savefig("output.png")
            image = Image.open('output.png')
            st.image(image)
    elif dataW in data.columns:  
            print('w')
            #data['Gen. Date']= pd.to_datetime(data['Gen. Date'])
            #data = pd.read_csv(input_file,encoding= 'unicode_escape',parse_dates=['DATE'])
            data['DATE']=pd.to_datetime(data['DATE'])
            lastdate=data['DATE'].iloc[-1]#24/11/2022
            print(data.tail(1))
            print(lastdate)
            n=12
            for i in (lastdate + timedelta(n) for n in range(365)):
                df1=pd.DataFrame({"Sr. No":0,"DATE":i,"WEC":0,"WEC Type":0,"GENERATION":0,"O.Hrs":0,"L.Hrs":0,"MA":0,"CF":0,"GIA":0,"GA":0,"REMARKS":0,""
                                  "Unnamed: 12":0,"Customer":0, "SITE":0,"STATE":0},index=[i])
                data=data.append(df1)
            data = data.set_index("DATE", drop=False)
            data1=data.resample('M').mean()
            data1 = data1.loc[:, ["GENERATION"]]#index
            data1 = data1.asfreq("M")
                 #Training and test set
            test_months = 12
            training_set = data1.iloc[:-test_months, :]
            test_set = data1.iloc[-test_months:, :]
            
            model = auto_arima(training_set,
                    start_p=0,d=1,start_q=0,max_p=5,max_d=5,max_q=5,
                    start_P=0,D=1,start_Q=0,max_P=5,max_D=5,max_Q=5,
                    m=12,seasonal=True,
                    trace=True,
                    )

            #predictions
            predictions_sarimax = pd.Series(model.predict(n_periods= test_months)).rename("Power")
            predictions_sarimax.index = test_set.index                              
          
            df=predictions_sarimax.to_frame()
            st.write(df)
            df = df.reset_index()
            #df['date']=df.index
            csv = convert_df(df)

            st.download_button(
            "Download",
            csv,
        "file.csv",
   "text/csv",
   key='download-csv'
)
            df['date']=df.index
            linechart=df.plot(y='Power',x='DATE',marker = 'o',
                              xlabel="Months",ylabel='Monthly Avg. Power generation in Kwh',
                              title='Monthly Avg Power Generation for upcoming 12 months',legend=None,figsize=[10,8])
            fig = linechart.get_figure()
            fig.savefig("output.png")
            image = Image.open('output.png')
            st.image(image)
           
            
            
            