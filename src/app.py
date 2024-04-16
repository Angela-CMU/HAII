import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import sklearn # import scikit-learn
from sklearn import preprocessing # import preprocessing utilites
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


features_cat = ['workclass', 'education', 'race', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
features_num = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

st.header("FairPay: Empowering Fairness in Interactive Income Analysis")
st.write("Team members: Yen-Ju Wu (yenjuw@andrew.cmu.edu) and Chien-Yu Liu (chienyul@andrew.cmu.edu) and Jyoshna Sarva (jsarva@andrew.cmu.edu)")

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
  feature_names = ['age', 'education', 'marital-status', 'race', 'sex']
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

def plot_two_feature_distribution(X):
    feature_names = ['age', 'education', 'marital-status', 'race', 'sex']
    feature1 = st.selectbox('Select First Feature', feature_names)
    feature2 = st.selectbox('Select Second Feature', feature_names)
    if feature1 and feature2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot distribution of first selected feature
        axes[0].hist(X[feature1], bins='auto', color='skyblue', edgecolor='black')
        axes[0].set_xlabel(feature1)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{feature1} Distribution')
        axes[0].tick_params(axis='x', rotation=90, labelsize=8)

        # Plot distribution of second selected feature
        axes[1].hist(X[feature2], bins='auto', color='salmon', edgecolor='black')
        axes[1].set_xlabel(feature2)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{feature2} Distribution')
        axes[1].tick_params(axis='x', rotation=90, labelsize=8)

        plt.tight_layout()
        st.pyplot(fig)


# models
def build_encoder(X_train):
    X_cat = X_train[features_cat]
    X_num = X_train[features_num]
    
    enc = preprocessing.OneHotEncoder()
    enc.fit(X_cat) # fit the encoder to categories in our data 
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_num)

    return enc, scaler

def preprocessing_data(X, enc, scaler):
    X_cat = X[features_cat]
    X_num = X[features_num]

    # Pre-processing categorical data using one hot encoding
    one_hot = enc.transform(X_cat) # transform data into one hot encoded sparse array format
    X_cat_proc = pd.DataFrame(one_hot.toarray(), columns=enc.get_feature_names_out()) # put the newly encoded sparse array back into a pandas dataframe so that we can use it

    # Pre-processing (scaling) numerical data
    scaled = scaler.transform(X_num)
    X_num_proc = pd.DataFrame(scaled, columns=features_num)

    X_preprocessed = pd.concat([X_num_proc, X_cat_proc], axis=1, sort=False)
    X_preprocessed = X_preprocessed.fillna(0)

    return X_preprocessed

def upsample(X_train, y_train):
    # concatenate X_train and y_train
    Xy_train = pd.concat([X_train, y_train], axis=1)
    # print('    We have', Xy_train.shape[0] , 'training data')

    # split them into whether the sample is caused by the specific reason
    Xy_train_less_than_50K = Xy_train[Xy_train['income'] == '<=50K']
    Xy_train_more_then_50K = Xy_train[Xy_train['income'] == '>50K']
    # print('    We have', Xy_train_less_than_50K.shape[0] , 'training data whose income are less than or equal to 50K')
    # print('    We have', Xy_train_more_then_50K.shape[0] , 'training data whose income are more than 50K')

    # upsample the Xy_train_more_then_50K rows
    Xy_train_more_then_50K_up = resample(Xy_train_more_then_50K, n_samples=len(Xy_train_less_than_50K), random_state=1)
    Xy_train_up = pd.concat([Xy_train_less_than_50K, Xy_train_more_then_50K_up], axis=0)
    # print('    After upsampling Xy_train_more_then_50K data, we have', Xy_train_up.shape[0] , 'training data')
    X_train_up = Xy_train_up[X_train.columns]
    y_train_up = Xy_train_up[y_train.name]

    return X_train_up, y_train_up

# helper method to print basic model metrics
def metrics(y_true, y_pred):
    st.write('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
    st.write('\nReport:\n', classification_report(y_true, y_pred, digits=4))

def train_model(X_train, X_test, y_train, y_test):
    # model = LogisticRegression(solver='lbfgs').fit(X_train, y_train) # first fit (train) the model
    model = RandomForestClassifier().fit(X_train, y_train) # first fit (train) the model
    y_pred = model.predict(X_test) # next get the model's predictions for a sample in the validation set
    metrics(y_test, y_pred) # finally evaluate performance

    return model

def build_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1) # split out into training 70% of our data
    global enc
    global scaler
    enc, scaler = build_encoder(X_train)
    
    # # preprocessing then upsample
    # # (after preprocessing the X_train_preprocessed is reset, so y_train needs to be reset as well)
    # X_train_preprocessed = preprocessing_data(X_train, enc, scaler)
    # X_train_up_preprocessed, y_train_up = upsample(X_train_preprocessed, y_train.reset_index(drop=True), death_cause)
    
    # upsample then preprocessing
    X_train_up, y_train_up = upsample(X_train, y_train)
    X_train_up_preprocessed = preprocessing_data(X_train_up, enc, scaler)
    X_test_preprocessed = preprocessing_data(X_test, enc, scaler)

    model = train_model(X_train_up_preprocessed, X_test_preprocessed, y_train_up.reset_index(drop=True), y_test.reset_index(drop=True))
    
    return model

def plot_race_and_income(df):
    #Sex Pie Chart
    sex_value_counts = df['sex'].value_counts()
    sex_df = pd.DataFrame({'sex': sex_value_counts.index, 'Count': sex_value_counts.values})
    sex_pie_chart = alt.Chart(sex_df).mark_arc().encode(
        color='sex:N',
        theta='Count:Q',
        tooltip=['sex', 'Count']
    ).properties(
        width=400,
        height=400,
        title='Pie Chart: Gender Distribution'
    )
    sex_pie_chart

    #Race Pie Chart
    race_value_counts = df['race'].value_counts()
    race_df = pd.DataFrame({'race': race_value_counts.index, 'Count': race_value_counts.values})
    race_pie_chart = alt.Chart(race_df).mark_arc().encode(
        color='race:N',
        theta='Count:Q',
        tooltip=['race', 'Count']
    ).properties(
        width=400,
        height=400,
        title='Pie Chart: Race Distribution'
    )
    race_pie_chart

    #Filters
    age_filter=st.sidebar.slider('Age:',min_value=min(df['age']),max_value=max(df['age']),value=(min(df['age']),max(df['age'])))
    df = df[(df["age"] >= age_filter[0]) & (df["age"] <= age_filter[1])]

    sex=['Male','Female']
    sex_filter = st.sidebar.selectbox('Gender:',sex)
    df = df[df['sex'] == sex_filter]

    workclass_filter=st.sidebar.multiselect('Work class:',df['workclass'].unique())
    if len(workclass_filter) != 0:
       df = df[(df['workclass'].isin(workclass_filter))]

    marital_status_filter=st.sidebar.multiselect('Marital status:',df['marital-status'].unique())
    if len(marital_status_filter) != 0:
        df = df[(df['marital-status'].isin(marital_status_filter))]

    race_filter=st.sidebar.multiselect('Race:',df['race'].unique())
    if len(race_filter) != 0:
        df = df[(df['race'].isin(race_filter))]

    st.subheader("Analyzing fairness in income")
    bar_race_income = alt.Chart(df).mark_bar().encode(
        x='race:N',
        y='count():Q',
        color='income:N'
    ).properties(
        width=800,
        height=600
    ).configure_axisX(
        labelAngle=-90
    )
    bar_race_income

    return df

def main():
  X, y = load_data()
  st.dataframe(X.head())

  plot_feature_distribution(X)
  plot_two_feature_distribution(X)
  y_series = y['income'].str.replace('.', '')
  y_series.value_counts()

  df = pd.concat([X, y_series], axis=1, sort=False)
  df = plot_race_and_income(df)

  model = build_model(df.drop(columns=['income']), df['income'])

if __name__ == '__main__':
  main()
