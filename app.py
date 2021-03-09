#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import pickle
import numpy
from sklearn.ensemble import RandomForestClassifier

# In[3]:

st.cache()
def get_data():
    url = "Placement_Data_Full_Class.csv"
    return pd.read_csv(url)
data = get_data()

st.set_option('deprecation.showPyplotGlobalUse', False)
# In[4]:



# In[6]:


st.title("Placement Prediction")
st.markdown("""This web app is based on the Placement Data set available on [`Kaggle`](https://www.kaggle.com/benroshan/factors-affecting-campus-placement).
            The original Notebook can be viewed from [`here`](https://www.kaggle.com/aakashg1999/eda-placement-aakash). [`Aakash`](https://www.kaggle.com/aakashg1999)""")


# In[7]:


st.header("Credits")
st.markdown("""The dataset below is provided by Kaggle user [`Ben Roshan D`](https://www.kaggle.com/benroshan).
            We would like to thank him for providing the dataset.
             PS : The data can be sorted according to a column by selecting that column.""")


# In[ 8]:st.markdown(data.columns.tolist())
cols = data.columns.tolist()
st_ms = st.multiselect("Columns", data.columns.tolist(), default=cols)
st.dataframe(data[st_ms].head(20))

# In[Final}
st.title("Enter Data for making a prediction")
pickle_Filename='Pickle_rfc_Model.pkl'

with open(pickle_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

dict_std={'ssc_p':11.027370,'hsc_p':11.053859,'degree_p':7.139097,
          'mba_p':5.717089,'etest_p':12.990365,'specialisation_Mkt&HR':0.499410,
          'workex_Yes':0.481662,'specialisation_Mkt&Fin':0.499410,'workex_No':0.481662}
dict_mean={'ssc_p':67.168667,'hsc_p':66.481167,'degree_p':66.201611,
           'mba_p':62.177111,'etest_p':71.917722,'specialisation_Mkt&HR':0.455556,
           'workex_Yes':0.361111,'specialisation_Mkt&Fin':0.544444,'workex_No':0.638889}
ser_mean=pd.Series(dict_mean)
ser_std=pd.Series(dict_std)
ssc=st.number_input("Enter SSC %age",30.,100.,)
hsc=st.number_input("Enter HSC %age",30.,100.,)
degree=st.number_input("Enter Graduation %age",30.,100.,)
mba=st.number_input("Enter MBA %age",30.,100.,)
etest=st.number_input("Enter Employability Test %age",30.,100.,)
specialization=st.selectbox('What is your MBA specialization?',
                            ('Marketing and HR','Marketing and Finance'))
workex=st.selectbox('Do you have Work Experience?',
                    ('Yes','No'))
finance=0
HR=0
Yes=0
No=0
if specialization[-1] is 'e':
    finance=1
    HR=0
elif specialization[-1] is 'R':
    HR=1
    finance=0
    
if workex[-1] is 's':
    Yes=1
    No=0
elif specialization[-1] is 'o':
    No=1
    Yes=0


dict_cols={'ssc_p':[ssc],'hsc_p':[hsc],'degree_p':[degree],
          'mba_p':[mba],'etest_p':[etest],'specialisation_Mkt&HR':[HR],
          'workex_Yes':[Yes],'specialisation_Mkt&Fin':[finance],'workex_No':[No]}
df=pd.DataFrame(dict_cols)
df=(df-ser_mean)/ser_std
result=Pickled_Model.predict(df)

st.header("Prediction")
result_list=result.tolist()
if result_list[0] is 0:
    st.write("Unplaced")
elif result_list[0] is 1:
    st.write("Placed!")

# In[9]:
st.cache()
st.title("Heatmap")
st.markdown("1) High correlation between status and (ssc_p, hsc_p, degree_p).")
st.markdown("2) Low correlation between status and (etest_p, mba_p).")
st.markdown("PS : The heatmap and all the graphs can be enlarged")
sns.set(style='whitegrid', palette='muted', font_scale=1.1)
data2 = data.copy()
data2['status'] = data2['status'].map({'Placed':1, 'Not Placed': 0}).astype(int)
plt.figure(figsize=(15,10))
plt.title('Heatmap')
sns.heatmap(data=data2.drop('salary', axis=1).corr(), annot=True)
st.pyplot()


# In[17]
st.title("Demonstration of the observations")
st.markdown("It was established that good performance in all fronts increases chance of placement. The same is demonstrated by the tool below.")
values_ssc = st.slider("SSC percentage", 30., 95., (50., 75.))
values_hsc = st.slider("HSC percentage", 40., 100., (50., 75.))
values_degree = st.slider("Degree percentage", 45., 95., (50., 75.))
values_mba = st.slider("MBA percentage", 30., 80., (50., 75.))

d1=data.query(f"ssc_p.between{values_ssc}", engine='python')
d1=d1.query(f"hsc_p.between{values_hsc}", engine='python')
d1=d1.query(f"degree_p.between{values_degree}", engine='python')
d1=d1.query(f"mba_p.between{values_mba}", engine='python')
st.dataframe(d1)

plt.figure(figsize=(15,10))
sns.swarmplot(x=d1['status'],y=d1['etest_p'])
st.pyplot()

plt.figure(figsize=(15,10))
sns.scatterplot(x=d1['etest_p'],y=d1['salary'],hue=d1['gender'])
st.pyplot()



