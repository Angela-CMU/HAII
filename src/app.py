import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from ucimlrepo import fetch_ucirepo 

st.header("FairPay: Empowering Fairness in Interactive Income Analysis")
st.write("Team members: Yen-Ju Wu (yenjuw@andrew.cmu.edu) and Chien-Yu Liu (chienyul@andrew.cmu.edu)")

@st.cache_data
def load_data():
  # fetch dataset 
  adult = fetch_ucirepo(id=2) 
    
  # data (as pandas dataframes) 
  X = adult.data.features 
  y = adult.data.targets 
  return X, y

def main():
  X, y = load_data()
  st.dataframe(X.head())

if __name__ == '__main__':
  main()
