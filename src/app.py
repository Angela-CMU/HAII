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


def plot_feature_distribution_combined(df):
    ##### selectbox #####
    # feature_names = ['age', 'education', 'martial-status', 'race', 'sex']
    feature_names = ['age', 'workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    feature_select = st.selectbox('Feature distribution', feature_names)

    # Plot distribution of selected feature
    if feature_select == 'age':
        # Bar Chart
        # feature_value_counts = df[feature_select].value_counts()
        # feature_df = pd.DataFrame({feature_select: feature_value_counts.index, 'Count': feature_value_counts.values})
        feature_bar_chart = alt.Chart(df).mark_bar().encode(
            alt.X(feature_select+':Q', bin=True),
            y='count()',
        ).properties(
            width=500,
            height=500,
            title='Bar Chart: ' + feature_select + ' distribution'
        )
        feature_bar_chart
    elif feature_select:
        # Pie Chart
        feature_value_counts = df[feature_select].value_counts()
        feature_df = pd.DataFrame({feature_select: feature_value_counts.index, 'Count': feature_value_counts.values})
        feature_pie_chart = alt.Chart(feature_df).mark_arc(outerRadius=150).encode(
            color=feature_select+':N',
            theta='Count:Q',
            tooltip=[feature_select, 'Count']
        ).properties(
            width=600,
            height=600,
            title='Pie Chart: ' + feature_select + ' distribution'
        )
        feature_pie_chart

def plot_feature_vs_income(df):
    ##### selectbox #####
    feature_names = ['age', 'education', 'martial-status', 'race', 'sex']
    feature_select = st.selectbox('Feature distribution', feature_names, key='ratio')

    # Plot distribution of selected feature
    if feature_select == 'age':
        age_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        df_binned = df.copy()
        df_binned['age_bins'] = pd.cut(df_binned['age'], bins=age_bins)
        df_income_larger_50k = df_binned[df_binned['income'] == '>50K']

        age_value_counts = df_binned['age_bins'].value_counts()
        age_value_counts_larger_than_50k = df_income_larger_50k['age_bins'].value_counts().rename("count_larger_50k")
        age_value_counts_concat = pd.concat([age_value_counts, age_value_counts_larger_than_50k], axis=1).fillna(0)

        age_value_counts_concat['age_income_ratio'] = (age_value_counts_concat['count_larger_50k'] / age_value_counts_concat['count']) * 100
        age_value_counts_concat['index_name'] = age_value_counts_concat.index

        age_bar_chart = alt.Chart(age_value_counts_concat).mark_bar().encode(
            x='index_name:N',
            y=alt.Y('age_income_ratio:Q', title='Age Income Ratio (%)', scale=alt.Scale(domain=(0, 100))),
        ).properties(
            width=500,
            height=500,
            title='Bar Chart: ' + feature_select + ' vs. income distribution'
        )
        age_bar_chart

    elif feature_select:
        # Bar Chart
        df_income_larger_50k = df[df['income'] == '>50K']

        feature_value_counts = df[feature_select].value_counts()
        feature_value_counts_larger_than_50k = df_income_larger_50k[feature_select].value_counts().rename("count_larger_50k")
        feature_value_counts_concat = pd.concat([feature_value_counts, feature_value_counts_larger_than_50k], axis=1).fillna(0)

        feature_value_counts_concat['feature_income_ratio'] = (feature_value_counts_concat['count_larger_50k'] / feature_value_counts_concat['count']) * 100
        feature_value_counts_concat['index_name'] = feature_value_counts_concat.index

        feature_bar_chart = alt.Chart(feature_value_counts_concat).mark_bar().encode(
            x='index_name:N',
            y=alt.Y('feature_income_ratio:Q', title=feature_select + ' income Ratio (%)', scale=alt.Scale(domain=(0, 100))),
        ).properties(
            width=500,
            height=500,
            title='Bar Chart: ' + feature_select + ' vs. income distribution'
        )
        feature_bar_chart

# models
def build_encoder(X_train, features_cat, features_num):
    X_cat = X_train[features_cat]
    X_num = X_train[features_num]
    
    enc = preprocessing.OneHotEncoder()
    enc.fit(X_cat) # fit the encoder to categories in our data 
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_num)

    return enc, scaler

def preprocessing_data(X, enc, scaler, features_cat, features_num):
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

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.write('Classification report: \n')
    st.write(report_df.reset_index().rename(columns={'index': 'Report'}))

def train_model(X_train, X_test, y_train, y_test, model_type, print_report):
    if model_type == "Logistic Regression":
        model = LogisticRegression(solver='lbfgs').fit(X_train, y_train) # first fit (train) the model
    elif model_type == "Random Forest":
        model = RandomForestClassifier().fit(X_train, y_train) # first fit (train) the model

    if print_report:
        y_pred = model.predict(X_test) # next get the model's predictions for a sample in the validation set
        metrics(y_test, y_pred) # finally evaluate performance

    return model

def build_model(X, y, features_cat, features_num, model_type, print_report=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1) # split out into training 70% of our data
    enc, scaler = build_encoder(X_train, features_cat, features_num)
    
    # # preprocessing then upsample
    # # (after preprocessing the X_train_preprocessed is reset, so y_train needs to be reset as well)
    # X_train_preprocessed = preprocessing_data(X_train, enc, scaler)
    # X_train_up_preprocessed, y_train_up = upsample(X_train_preprocessed, y_train.reset_index(drop=True), death_cause)
    
    # upsample then preprocessing
    X_train_up, y_train_up = upsample(X_train, y_train)
    X_train_up_preprocessed = preprocessing_data(X_train_up, enc, scaler, features_cat, features_num)
    X_test_preprocessed = preprocessing_data(X_test, enc, scaler, features_cat, features_num)

    model = train_model(X_train_up_preprocessed, X_test_preprocessed, y_train_up.reset_index(drop=True), y_test.reset_index(drop=True), model_type, print_report)
    
    return scaler, enc, model



def get_user_inp(original_X):
    # Get User Input
    user_data_point = {
        'age' : st.sidebar.number_input('Age:',min_value=min(original_X['age']),max_value=max(original_X['age']),value=min(original_X['age'])),
        'workclass' : st.sidebar.selectbox('Workclass:',original_X['workclass'].unique()),
        # 'fnlwgt' : st.sidebar.number_input('fnlwgt:',min_value=min(original_X['fnlwgt']),max_value=max(original_X['fnlwgt']),value=min(original_X['fnlwgt'])),
        'education' : st.sidebar.selectbox('Education:',sorted(original_X['education'].unique())),
        'education-num' : st.sidebar.selectbox('Education Number:',sorted(original_X['education-num'].unique())),
        'martial-status' : st.sidebar.selectbox('Martial Status:',sorted(original_X['martial-status'].unique())),
        # 'occupation' : st.sidebar.selectbox('Occupation:',original_X['occupation'].unique()),
        # 'relationship' : st.sidebar.selectbox('Relationship:',sorted(original_X['relationship'].unique())),
        'race' : st.sidebar.selectbox('Race:',sorted(original_X['race'].unique())),
        'sex' : st.sidebar.selectbox('Gender:', original_X['sex'].unique()),
        # 'capital-gain' : st.sidebar.number_input('Capital-gain:',min_value=min(original_X['capital-gain']),max_value=max(original_X['capital-gain']),value=min(original_X['capital-gain'])),
        # 'capital-loss' : st.sidebar.number_input('Capital-loss:',min_value=min(original_X['capital-loss']),max_value=max(original_X['capital-loss']),value=min(original_X['capital-loss'])),
        'hours-per-week' : st.sidebar.number_input('Hours-per-week:',min_value=min(original_X['hours-per-week']),max_value=max(original_X['hours-per-week']),value=min(original_X['hours-per-week'])),
        'native-country' : st.sidebar.selectbox('Native-country:',original_X['native-country'].unique())
    }
    user_inp_data = pd.DataFrame([user_data_point], columns=original_X.columns)
    #    st.dataframe(user_inp_data)
    return user_inp_data
   
def main(): 
    # load data
    X, y = load_data()
    X['martial-status'] = X['marital-status']
    original_X = X.copy()

    # clean data
    y_series = y['income'].str.replace('.', '')
    y_series.value_counts()
    df = pd.concat([X, y_series], axis=1, sort=False)

    # select features to be trained
    # features_cat = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    # features_num = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_cat = ['workclass', 'education', 'martial-status', 'race', 'sex', 'native-country']
    features_num = ['age', 'education-num', 'hours-per-week']

    # select page in the left side of webpage    
    select_page = st.sidebar.radio("Select Page:", ["Introduction", "Data analysis", "Model Training", "Fairness analysis", "User input prediction"])

    if select_page == "Introduction":
        # st.markdown("Our project utilizes income datasets sourced from various Census surveys and programs. With this data, our aim is to uncover patterns within salary information, recognizing the paramount importance individuals place on salary in their career trajectories. We seek to identify the common factors influencing salary while scrutinizing the presence of biases within the job market. We are attentive to potential biases introduced during data collection processes and vigilant against biases emerging during data analysis, whether stemming from human factors or algorithmic/model biases. Our project not only provides users with opportunities to interact with the data and glean insights but also endeavors to identify and address potential biases throughout the entire process.")
        
        st.subheader("Source of Our Datasets")
        url = "https://archive.ics.uci.edu/dataset/2/adult"
        st.markdown("Our project utilizes income datasets sourced from various Census surveys and programs, which can be found at [UC Irvine Machine Learning Repository](%s)" %url) # "check out this [link](%s)" % url
        st.markdown("With this data, our aim is to uncover patterns within salary information, recognizing the paramount importance individuals place on salary in their career trajectories. We seek to identify the common factors influencing salary while scrutinizing the presence of biases within the job market. We are attentive to potential biases introduced during data collection processes and vigilant against biases emerging during data analysis, whether stemming from human factors or algorithmic/model biases. Our project not only provides users with opportunities to interact with the data and glean insights but also endeavors to identify and address potential biases throughout the entire process.")
        
        st.subheader("Datasets")
        st.markdown("To begin, here are a couple of example data for your reference:")
        st.dataframe(X.head())

        st.markdown("Here are the statistics about the datasets:")
        st.dataframe(X.describe().drop(columns=['fnlwgt']))

        st.markdown("You can access the Data analysis page, Model training page, Fairness analysis page, or the User input prediction page via the selection bar located on the left side.")

    elif select_page == "Data analysis":
        # plot_feature_distribution(X)
        # plot_two_feature_distribution(X)
        # _ = plot_race_and_income(df)
        st.subheader("Distribution of features in the dataset")
        st.markdown("Please select the feature you would like to visualize the distribution of")
        plot_feature_distribution_combined(df)

    elif select_page == "Model Training":    
        st.subheader("Types of Models for Dataset Training")
        st.markdown("Please select the machine learning model you wish to train with the datasets")
        st.markdown("Please note that we have balanced the data distribution by upsampling the data where income='>50k' to match the number of data points where income='<50k'.")
        st.markdown("The following are our selected features for training and testing the model:")
        st.markdown("Categorical features includes: 'workclass', 'education', 'martial-status', 'race', 'sex', and 'native-country'")
        st.markdown("Numerical features includes: 'age', 'education-num', and 'hours-per-week'")
        ##### selectbox #####
        model_types = ['Logistic Regression', 'Random Forest']
        model_select = st.selectbox('Model Types', model_types)

        # Plot distribution of selected feature
        if model_select:
            st.markdown("Building the model with the data sets...")
            st.markdown("Here are the results:")
            _, _, _ = build_model(df.drop(columns=['income']), df['income'], features_cat, features_num, model_select, print_report=True)
            st.markdown("Completed!")

    elif select_page == "Fairness analysis":
        st.subheader("Fairness Analysis")
        st.markdown("Please select the feature to visualize its distribution with respect to income")
        st.markdown("The following figure displays the percentage of each category with an income >50k")
        plot_feature_vs_income(df)

    elif select_page == "User input prediction":
        st.subheader("User Input Prediction")
        st.markdown("Please select the machine learning model you wish to train with the datasets")
        st.markdown("Please input the user data to be used as input for predicting with the models on the left side of the page.")
        st.markdown("In this section, we'll construct three models based on the model type you've chosen. Each model will have a different feature set: one using all features, one using all features except sex, and one using all features except race. This comparison will help us assess whether the model's predictions differ, indicating potential bias towards sex or race.")
        ##### selectbox #####
        model_types = ['Logistic Regression', 'Random Forest']
        model_select = st.selectbox('Model Types', model_types)

        if model_select:
            X_user = get_user_inp(original_X)

            if st.button('Predict with original model'):
                # build the original model
                scaler, enc, model = build_model(df.drop(columns=['income']), df['income'], features_cat, features_num, model_select)
                X_user_preprocess = preprocessing_data(X_user, enc, scaler, features_cat, features_num)
                y_user = model.predict(X_user_preprocess)[0] 
                st.success(f"Income: {y_user}")
            
            if st.button('Predict with model without considering sex'):
                # features_cat_without_sex = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'native-country']
                features_cat_without_sex = ['workclass', 'education', 'martial-status', 'race', 'native-country']
                scaler_without_sex, enc_without_sex, model_without_sex = build_model(df.drop(columns=['income']), df['income'], features_cat_without_sex, features_num, model_select)
                X_user_preprocess_without_sex = preprocessing_data(X_user, enc_without_sex, scaler_without_sex, features_cat_without_sex, features_num)
                y_user_without_sex = model_without_sex.predict(X_user_preprocess_without_sex)[0]
                st.success(f"Income: {y_user_without_sex}")
            
            if st.button('Predict with model without considering race and native-country'):
                # features_cat_without_race = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'sex', 'native-country']
                features_cat_without_race = ['workclass', 'education', 'martial-status', 'sex']
                scaler_without_race, enc_without_race, model_without_race = build_model(df.drop(columns=['income']), df['income'], features_cat_without_race, features_num, model_select)
                X_user_preprocess_without_race = preprocessing_data(X_user, enc_without_race, scaler_without_race, features_cat_without_race, features_num)
                y_user_without_race = model_without_race.predict(X_user_preprocess_without_race)[0]
                st.success(f"Income: {y_user_without_race}")
        

if __name__ == '__main__':
    main()
