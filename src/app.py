import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
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

def plot_feature_distribution(X):
  ##### selectbox #####
  feature_names = ['age', 'education', 'marital-status', 'race', 'sex', 'native-country']
  label_name = ['income']
  feature_select = st.selectbox('Feature distribution', feature_names)

  # Plot distribution of selected feature
  if feature_select:
    X_feature = X[feature_select]
    bin = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fig = plt.figure()
    plt.hist(X_feature, bins='auto', color='skyblue', edgecolor='black')
    plt.xlabel(feature_select)
    plt.ylabel('Frequency')
    plt.title(f'{feature_select} Distribution')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

def main():
  X, y = load_data()
  st.dataframe(X.head())

  plot_feature_distribution(X)

if __name__ == '__main__':
  main()
