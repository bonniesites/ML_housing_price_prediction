import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math
import os
import time
import json


# Code modified from https://medium.com/international-school-of-ai-data-science/make-your-streamlit-web-app-look-better-14355c2db871


st.set_page_config(
    page_title="ML Housing Prediction App",
    page_icon=":cityscape:"
)

# hide_default_format = """
#        <style>
#        #MainMenu {visibility: hidden; }
#        footer {visibility: hidden;}
#        </style>
#        """
st.markdown(hide_default_format, unsafe_allow_html=True)

# End snippet 


st.set_option('deprecation.showPyplotGlobalUse', False)

session_vars = ('target_chosen','chart_chosen', 'done_charts', 'set_features', 'fit_model', 'imp_chart', 'predict', 'set_predict')

for sess in session_vars:    
    if sess not in st.session_state:
         
        st.session_state.target_chosen = False
    
if 'chart_chosen' not in st.session_state:
    st.session_state.chart_chosen = False
    
if 'done_charts' not in st.session_state:
    st.session_state.done_charts = False
    
if 'set_features' not in st.session_state:
    st.session_state.set_features = False
    
if 'fit_model' not in st.session_state:
    st.session_state.fit_model = False
    
if 'imp_chart' not in st.session_state:
    st.session_state.imp_chart = False
    
if 'predict' not in st.session_state:
    st.session_state.predict = False
    
if 'set_predict' not in st.session_state:
    st.session_state.set_predict = False


st.title('ML Housing Prediction App')
mtarget = st.empty()
mfeatures = st.empty()
mheader = st.empty()
mainbox = st.empty()
st.sidebar.title('Welcome!')  
sheader = st.sidebar.empty()  
sheader2 = st.sidebar.empty()
smeasure1 = st.sidebar.empty()
smeasure2 = st.sidebar.empty()
smeasure3 = st.sidebar.empty()
sidebox = st.sidebar.empty()
chart_check = st.sidebar.empty()
data_file = None
XGBmodel = XGBRegressor()     


@st.cache_data  # for storage in database/frame 
def get_data(file):    
    return pd.read_csv(file)


# Show X feature chart
def build_chart(df, X, y):
    plt.scatter(df[X], df[y])
    plt.xlabel(X.capitalize())
    plt.ylabel(y.capitalize())
    plt.title('{} vs. {}'.format(X.capitalize(), y.capitalize()))
    

def save_model(model):
    filename = 'model.pickle'
    if os.path.exists(filename):
        os.remove(filename)
        #st.header(f"File '{filename}' deleted successfully.")
    #else:
        #st.header(f"File '{filename}' does not exist.") 
    # Save the model as a pickle file
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return True


#@st.cache_resource # for ML models or db connections 
def load_model():
    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def plot_feature_importance(_model, features):
    importance = _model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    
 
# Get model fit measures
@st.cache_resource(ttl=60)
def get_model_fit(data, target, features, _model):
    y = data[target]
    X = data[features]
    X = pd.get_dummies(X, drop_first=True)
    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)    
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    _model.fit(X_train_scaled, y_train)
    save_model(_model)
    # R2 score
    r2 = _model.score(X, y)
    predictions = _model.predict(X_test_scaled)
    # RMSE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    # MSE
    mse = mean_squared_error(y_test, predictions, squared=True)  
    return r2, rmse, mse


# Get predictions based on user input
@st.cache_resource
def get_prediction(user_input, features):
    st.session_state.set_predict = True
    loaded_model = load_model()
    input_data = pd.DataFrame([user_input], columns=features)
    prediction = loaded_model.predict(input_data)[0]
    return prediction


def set_target():
    st.session_state.target_chosen = True

def set_chart():
    st.session_state.chart_chosen = True
    
def done_charts():
    st.session_state.done_charts = True
    
def set_features():
    st.session_state.set_features = True
    
def fit_model():
    st.session_state.fit_model = True
    
def imp_chart():
    st.session_state.imp_chart = True
    
def predict():
    st.session_state.predict = True
    
def set_predict():
    st.session_state.set_predict = True


# Requirement 1: Allow the users to upload the data.csv file into the app.
sheader.header('Start by uploading the data file:')
with sidebox:
    if data_file is None:    
        #data_file = sidebox.file_uploader('', type=['csv'], label_visibility='hidden')
        data_file = 'data.csv'

if data_file is not None:
    data = get_data(data_file)
    data.drop(columns = ['id', 'date'], axis=1, inplace=True)
    with mainbox:
        mheader.header('Preview data')
        mainbox.write(data.head(5))
    
    # Requirement 2: Require the user to select the target variable.
    sheader.empty()
    sheader.header('Select your target:')
    target_form = sidebox.form('form_target')
    target = target_form.radio("Choose one", options=data.columns, horizontal=True, index=len(data.columns)-1)
    target_chosen = target_form.form_submit_button('Save Target', on_click=set_target)
    if not target:
        st.sidebar.error("'Target is required', icon='🔥'")
        target = target_form.radio("Choose one", options=data.columns, horizontal=True, index=len(data.columns)-1)
    # Drop target from features
    features = data.drop(target, axis=1)

    # Requirement 4: Provide a plot that allows the users to pick an x and plot it against their target (you can have it only work for numeric values).
    if st.session_state.target_chosen:
        mtarget.subheader('Your saved target is {}.'.format(target))
        dollar = ''
        if target == 'price':
            dollar = '$'     
        plot_features = features.select_dtypes(include='number').columns.tolist()
        chart_check.empty()        
        done_charts = chart_check.checkbox('Done with charts', on_change=done_charts)         
        if not st.session_state.done_charts:
            sidebox.empty() 
            sheader.header('Select your X feature for chart:')                   
            plot_form = sidebox.form('plot')
            x_axis = plot_form.radio('Choose one', options=plot_features, horizontal=True)
            chart_chosen = plot_form.form_submit_button('Show Chart', on_click=set_chart)
            if st.session_state.chart_chosen:
                build_chart(data, x_axis, target)
                mainbox.empty()
                mheader.header('Show Chart')
                mainbox.pyplot()
        else:
            # Requirement 3: Allow the users to pick the features they want to use in their ML model.
            sidebox.empty()
            mainbox.empty()
            mheader.empty()
            chart_check.empty()
            sheader.header('Select your feature(s):')
            feature_form = sidebox.form('form_feature')
            feature_list = features.columns.tolist()
            
            selected_features = feature_form.multiselect("Delete features you don't want", options=feature_list, default=feature_list)
            features_set = feature_form.form_submit_button('Save Features', on_click=set_features)
            
            # Requirement 5: Explain your ML model (pick something fun or easy) and provide them a`fit` button.
            if st.session_state.set_features:
                #selected_features_display = [s.replace("'", "") for s in selected_features]
                pretty_features = json.dumps(selected_features, indent=4)
                pretty_list = pretty_features[1:-1]
                mfeatures.subheader('Your saved features are: {}'.format(pretty_list))
                chart_check.empty()
                mainbox.empty()
                mheader.empty()
                sidebox.empty()
                sheader.markdown('### This app uses the [XGBoost Regressor model](https://www.kaggle.com/code/carlmcbrideellis/an-introduction-to-xgboost-regression)')
                model_fitted = sidebox.form('model_fit')
                model_fitted.header('Fit and save the model?')
                model_fit =  model_fitted.form_submit_button('Save Model', on_click=fit_model)
                if st.session_state.fit_model:
                    mheader.write('')
                    mtarget.write('')
                    sheader.header('Saving model...')
                    mainbox.write('')
                    fit_measure = get_model_fit(data, target, selected_features, XGBmodel)
                    sheader.header('Model has been trained and saved! ✅')
                    mainbox.write('')
                
                    # Requirement 6: Report a feature importance plot and at least one measure of model fit.
                    #show_imp = sidebox.form('imp_form')
                    #imp_box = sidebox.checkbox('Show Feature Importance Chart', on_change=imp_chart)
                    #if st.session_state.imp_chart:
                    sidebox.empty()
                    mheader.empty()
                    mfeatures.empty()
                    mheader.header('Feature Importance Chart')
                    saved_model = load_model()
                    plot_imp = plot_feature_importance(saved_model, selected_features)
                    mainbox.pyplot()
                    sidebox.empty()
                    sheader2.empty()
                    smeasure1.empty()
                    smeasure2.empty()
                    smeasure3.empty()
                    sheader2.header('Fit Measurements:')
                    smeasure1.header(f'R2:   {fit_measure[0]:.4f}')
                    smeasure2.header(f'RMSE: {fit_measure[1]:.4f}')
                    smeasure3.header(f'MSE:  {fit_measure[2]:.4f}')

                    # Requirement 7: Get predictions based on user input
                    predict_box = sidebox.checkbox('Get a new Prediction?', on_change=predict)
                    if st.session_state.predict:
                        sheader.header('Enter your data to predict from:')
                        sidebox.empty()
                        sheader2.empty()
                        mtarget.empty()
                        mheader.header('Your prediction result will show here...')
                        mainbox.empty()
                        predict_form = sidebox.form('predict_form')            
                        smeasure1.empty()
                        smeasure2.empty()
                        smeasure3.empty()
                        user_input = {}
                        count = 0
                        for feature in selected_features:
                            if 'zipcode' in feature:
                                init = 90210
                                step = 1
                            elif 'lot' in feature:
                                init = 1000
                                step = 500
                            elif 'sqft' in feature:
                                init = 500
                                step = 100
                            elif 'yr' in feature:
                                init = 1900
                                step = 10
                            elif 'lat' in feature or 'long' in feature:
                                min = -90
                                step = 10
                            else:
                                min = 0
                                max = 10000000
                                init = 1
                                step = 1
                            user_input[feature] = predict_form.number_input(f'Enter {feature}', key=feature, value=init, step=step, min_value=min, max_value=max)                                   
                        predict1 = predict_form.form_submit_button('Predict', on_click=set_predict)
                        if st.session_state.set_predict:
                            result = get_prediction(user_input, selected_features)
                            prediction = round(result, 2)
                            mheader.header('Prediction: {}{}'.format(dollar, prediction))

                            # Requirement 8: Allow them to download their pickle model file
                            if mainbox.download_button(label='Download Model File', data='application/octet-stream', file_name='model.pickle'):
                                with open('model.pickle', 'rb') as file:
                                    model_file = file.read()
                                
