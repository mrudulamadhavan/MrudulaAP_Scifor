#Importing Libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import image


#import model
st.markdown("<h1 style='text-align: center; color: red;'>Know your Fitness Level</h1>", unsafe_allow_html=True)



#resources path
FILE_DIR1 = os.path.dirname(os.path.abspath("C://Users//Mrudula Madhavan//Desktop//scifor//MainProject//pages//Predictor.py"))
# FILE_DIR1 = os.path.dirname(os.path.abspath("MainProject//pages//Predictor.py"))
FILE_DIR = os.path.join(FILE_DIR1,os.pardir)
dir_of_interest = os.path.join(FILE_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "predict.png")
img = image.imread(IMAGE_PATH)
st.image(img) 

DATA_PATH = os.path.join(dir_of_interest, "data")

#Load data
DATA_PATH1=os.path.join(DATA_PATH, "fitnesstracker_dataset.csv")
df=pd.read_csv(DATA_PATH1)
df1 = df.copy

xgb = pickle.load(open('C://Users//Mrudula Madhavan//Desktop//scifor//MainProject//xgb_model.pkl','rb'))

# Function to calculate BMI category
def get_bmi_category(bmi):
    if bmi < 18.5:
        
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        
        return 'Normal Weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def prediction(dayname,agegroup,gender,weight,height,sleeptime,sedentarytime,activetime,totaldistance):
    dayname_map = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}
    agegroup_map = {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6}
    gender_map = {'Male': 0, 'Female': 1}
    
    dayname_encoded = dayname_map.get(dayname, 0)
    agegroup_encoded = agegroup_map.get(agegroup, 0)
    gender_encoded = gender_map.get(gender, 0)
    
    # Prepare the input features for prediction
    features = [[dayname_encoded, agegroup_encoded, gender_encoded, weight, height, sleeptime, 
                 sedentarytime, activetime, totaldistance]]
    
    prediction = xgb.predict(features)
    return prediction[0]  # Return the predicted value



st.write('<h5 span style="color:brown"><b>Enter the details below :</b></h5></span>', unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)
with col1:
    dayname = st.selectbox("Activity Day", ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
with col2:  
    agegroup = st.selectbox("Age Group", ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
with col3:  
    gender = st.selectbox("Gender", ['Male','Female'])


col3, col4 = st.columns(2)
with col3:  
    weight = abs(float(st.text_input("Weight (kg)  :", value='0.0')))
with col4:  
    height = abs(float(st.text_input("Height (m) :  ", value='0.0')))


if st.button('**Calculate BMI**'):
    if weight == 0 or height == 0:
        st.error("Please enter your Weight and Height")
        if st.button('Continue'):
            st.experimental_rerun()

    else:
        bmi = weight / (height ** 2)
        bmi = round(bmi, 1)
        bmi_category = get_bmi_category(bmi)
        st.write("BMI : <span style='font-size: 20px; color: red'><b>{}</b></span>".format(bmi), unsafe_allow_html=True)
        st.write("BMI Category : <span style='font-size: 20px; color: green'><b>{}</b></span>".format(bmi_category), unsafe_allow_html=True)

st.write("------------------------------------------------------------------------------------------")





left_column, middle_column, right_column = st.columns(3)
with left_column:
    sleeptime = abs(int(st.text_input("Total Minutes Asleep :  ", value='0'))) 
with middle_column:
    sedentarytime = abs(int(st.text_input('Sedentary Minutes :  ', value='0')))
with right_column:
    activetime = abs(int(st.text_input("Total Active Minutes :  ", value='0')))

totaldistance = abs(float(st.text_input("Total Distance Covered (km) : ", value='0.0')))



total_steps = ''

#Create dataframe using all these values
sample = pd.DataFrame({
    "ActivityDayName": [dayname],
    "AgeGroup": [agegroup],
    "Gender": [gender],
    "WeightKg": [float(weight)],
    "Heightm": [float(height)],
    "TotalMinutesAsleep": [int(sleeptime)],
    "SedentaryMinutes": [int(sedentarytime)],
    "TotalActiveMinutes": [int(activetime)],
    "TotalDistancekm": [float(totaldistance)]  
})




# Map Activity Day to numerical values
day_map = {'Sunday': 1.0, 'Monday': 2.0, 'Tuesday': 3.0, 'Wednesday': 4.0, 'Thursday': 5.0, 'Friday': 6.0, 'Saturday': 7.0}
sample['ActivityDayName'] = sample['ActivityDayName'].map(day_map)
df['ActivityDayName'] = df['ActivityDayName'].map(day_map)

# Map Age Group to numerical values
age_map = {'18-24': 0.0, '25-34': 1.0, '35-44': 2.0, '45-54': 3.0, '55-64': 4.0, '65+': 5.0}
sample['AgeGroup'] = sample['AgeGroup'].map(age_map)
df['AgeGroup'] = df['AgeGroup'].map(age_map)

# Map Gender to numerical values
gender_map = {'Male': 0.0, 'Female': 1.0}
sample['Gender'] = sample['Gender'].map(gender_map)  
df['Gender'] = df['Gender'].map(gender_map)






#from sklearn.preprocessing import MinMaxScaler

# Get the feature names from the DataFrame
#feature_names = ['TotalDistancekm','TotalActiveMinutes','TotalMinutesAsleep','SedentaryMinutes']

# Fit the scaler to the entire DataFrame
#scaler = MinMaxScaler().fit(df[feature_names])
# Transform the DataFrame using the same feature names
#df[feature_names] = scaler.transform(df[feature_names])


df.drop(['CaloriesBurnt'],inplace = True,axis = 1)

#Split data into X and y
X=df.drop(['TotalSteps'], axis=1).values
y=df['TotalSteps'].values


# Splitting data into 75:25 ratio
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

# Fit the scaler on the sample DataFrame
scaler = StandardScaler()
scaler.fit(sample)
# Transform the sample DataFrame using the fitted scaler
sample = scaler.transform(sample)

#Train the model
xgb=XGBRegressor(learning_rate=0.15, n_estimators=50, max_leaves=0, random_state=42)
xgb.fit(X,y)



# Prediction
# Prediction
if st.button('**Predict Steps Counts**'):
    if any([weight == 0, height == 0]):
        st.error("Please enter all details to continue.")
        if st.button('Continue'):
            st.experimental_rerun()
    elif totaldistance == 0:
        price = prediction(dayname, agegroup, gender, weight, height, sleeptime, sedentarytime, activetime, totaldistance)
        step_cnt = abs(max(0,int(price))) # Ensure step count is non-negative and integer
        st.subheader(":blue[The Predicted Value for Step Counts :] :green[{}]".format(step_cnt))
        if st.button('Continue'):
            st.experimental_rerun()

    else:
        price = prediction(dayname, agegroup, gender, weight, height, sleeptime, sedentarytime, activetime, totaldistance)
        step_cnt = abs(int(price)) # Ensure step count is non-negative and integer
        st.subheader(":blue[The Predicted Value for Step Counts :] :green[{}]".format(step_cnt))
        if st.button('Continue'):
            st.experimental_rerun()


    



