import streamlit as st
from matplotlib import image
import pandas as pd
import plotly.express as px
import os
import numpy as np

#resources path
FILE_DIR1 = os.path.dirname(os.path.abspath("C://Users//Mrudula Madhavan//Desktop//scifor//MainProject//pages//Recommendations.py"))
FILE_DIR = os.path.join(FILE_DIR1,os.pardir)
dir_of_interest = os.path.join(FILE_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "exercise.png")


#import model
st.markdown("<h1 style='text-align: center; color: blue;'>Exercise Recommendations</h1>", unsafe_allow_html=True)

img = image.imread(IMAGE_PATH)
st.image(img) 




# Function to provide detailed exercise recommendations
def suggest_exercises_with_details(goal):
    exercises = {
        'Maintain tone': [
            {
                'name': 'Push-ups',
                'benefits': 'Strengthens chest, shoulders, and triceps.',
                'description': 'Push-ups are a classic bodyweight exercise that targets multiple muscle groups in the upper body. They help maintain muscle tone and strength in the chest, shoulders, and triceps.',
                'how to': 'Lie facedown and place hands on the floor, slightly wider than shoulders. Push up to lift shoulders, torso, and legs until arms are fully extended. Slowly lower your body until chest almost touches the floor, then repeat.',
                'repetitions': '10-15'
            },
            {
                'name': 'Plank',
                'benefits': 'Strengthens core muscles.',
                'description': 'Planks are an isometric exercise that primarily targets the core muscles, including the abdominals, obliques, and lower back. They help improve core stability and posture.',
                'how to':'Start in plank position, with elbows and toes on the floor, core engaged, and torso elevated.',
                'duration': 'Hold for 30-60 seconds'
            }
        ],
        'Build muscle': [
            {
                'name': 'Squats',
                'benefits': 'Strengthens quadriceps, hamstrings, and glutes.',
                'description': 'Squats are a compound lower-body exercise that targets the quadriceps, hamstrings, and glutes. They are effective for building muscle mass and strength in the lower body.',
                'how to':'Stand with feet slightly wider than shoulder width. Extend arms straight with palms facing down. Inhale and push hips back slightly as you bend your knees. Look straight ahead and keep chin up, shoulders upright, and back straight. Squat as low as you comfortably can, aiming to have your hips sink below your knees. Engage your core to push upward explosively from your heels.',
                'repetitions': '3–5 sets of 8–12 reps'
            },
            {
                'name': 'Deadlifts',
                'benefits': 'Builds lower back, hamstrings, and glutes.',
                'description': 'Deadlifts are a compound exercise that targets the posterior chain, including the lower back, hamstrings, and glutes. They are excellent for building overall strength and muscle mass in the lower body and back.',
                'how to':'Bend down, grab an object on the floor, and lift it off the floor.',
                'repetitions': '6-10'
            }
        ],
        'Lose fat': [
            {
                'name': 'Running',
                'benefits': 'Burns calories and improves cardiovascular health.',
                'description': 'Running is a high-intensity cardiovascular exercise that burns a significant number of calories and helps improve cardiovascular health. It is effective for burning fat and losing weight when combined with a balanced diet.',
                'duration': '20-30 minutes'
            },
            {
                'name': 'Jumping Jacks',
                'benefits': 'Burns calories and improves overall fitness.',
                'description': 'Jumping jacks are a full-body exercise that elevates the heart rate and burns calories. They are effective for improving cardiovascular fitness and can be performed as part of a high-intensity interval training (HIIT) workout.',
                'repetitions': '30-60 seconds'
            }
        ]
    }
    return exercises.get(goal, 'No exercises available for this goal.')


goal = st.selectbox('**Select your goal**', ['Maintain tone', 'Build muscle', 'Lose fat'])

if st.button('Suggest Exercises'):
    exercises = suggest_exercises_with_details(goal)
    
    for exercise in exercises:
        st.subheader(exercise['name'])
        st.write('<span style="color:black"><b>Benefits :</b></span>', exercise['benefits'], unsafe_allow_html=True)
        st.write('<span style="color:black"><b>Description :</b></span>', exercise['description'], unsafe_allow_html=True)
        if 'how to' in exercise:
            st.write('<span style="color:black"><b>How to :</b></span>', exercise['how to'], unsafe_allow_html=True)
        if 'repetitions' in exercise:
            st.write('<span style="color:black"><b>Repetitions :</b></span>', exercise['repetitions'], unsafe_allow_html=True)
        if 'duration' in exercise:
            st.write('<span style="color:black"><b>Duration :</b></span>', exercise['duration'], unsafe_allow_html=True)
        st.write('----------------------')