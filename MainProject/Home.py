
import streamlit as st
from PIL import Image
import os

# Title of the home page
st.markdown("<h1 style='text-align: center; color: red;'>Fitness Buddy</h1>", unsafe_allow_html=True)

# Adding Image
FILE_DIR = os.path.dirname(os.path.abspath("C://Users//Mrudula Madhavan//Desktop//scifor//MainProject//Home.py"))
dir_of_interest = os.path.join(FILE_DIR, "resources", "images")
IMAGE_PATH = os.path.join(dir_of_interest, "homepage.png")

img = Image.open(IMAGE_PATH)
st.image(img)

# Using subheader
st.write('By: :green[Mrudula A P]')
