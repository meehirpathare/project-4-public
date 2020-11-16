import streamlit as st
from PIL import Image
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

image = Image.open('header.png')

st.image(image, use_column_width=True)
title = st.text_input("", " ")
st.write('The current movie title is', title)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("college_confidential.css")
