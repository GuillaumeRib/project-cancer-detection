from importlib.resources import path
import streamlit as st
import os
from PIL import Image
import numpy as np
from project_cancer_detection.interface.main_local import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from tempfile import NamedTemporaryFile



# the page icon
st.set_page_config("Deep Learning for Cancer Diagnosis", "🔬")

bg_img = '''
    <style>
    section.css-k1vhr4.egzxvld3 {
    background-image: url("https://i.imgur.com/3BOSVg9.png");
    background-size: cover;
    }
    </style>
    '''

# cool image link: https://i.imgur.com/3BOSVg9.png

# the whole page: css-18ni7ap e8zbici2
# upper part is white: css-1wrcr25.egzxvld4

st.markdown(bg_img, unsafe_allow_html=True)


header_bg_img = '''
    <style>
    header.css-18ni7ap.e8zbici2 {
    background-color: #f0f2f6;
    background-size: cover;
    }
    </style>
    '''

# the whole page: css-18ni7ap e8zbici2
# upper part is white: css-1wrcr25.egzxvld4

st.markdown(header_bg_img, unsafe_allow_html=True)


middle_bg_img = '''
    <style>
    div.block-container.css-12oz5g7.egzxvld2 {
    background-color: white;
    background-size: cover;
    color: dark-grey;
    }
    </style>
    '''
# small-middle-part: css-1n76uvr e1tzin5v0
# bigger-middle-part: block-container css-12oz5g7 egzxvld2

st.markdown(middle_bg_img, unsafe_allow_html=True)

button_bg_img = '''
    <style>
    button.css-1cpxqw2.edgvbvh9 {
    background-color: white;
    background-size: cover;
    border-color: red;
    color: red;
    }
    </style>
    '''
# row-widget stButton

st.markdown(button_bg_img, unsafe_allow_html=True)


# header
col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('https://images.emojiterra.com/twitter/v14.0/512px/1f9ec.png', width=60)
with col2:
    st.markdown("<h2 style='text-align: center; color: #131032;'>Deep Learning for Cancer Diagnosis</h2>", unsafe_allow_html=True)

# upload image
st.markdown("<h6 style='text-align: center; color: grey;'>Insert image for metastatic tissue prediction</h6>", unsafe_allow_html=True)
uploaded= st.file_uploader("", type=['tif'])

# link to sample files from googledrive - 0 and 1
drive = '[Download sample scan images for testing](https://drive.google.com/drive/folders/1S7Tbb1WZ6xciym6exqT37zbY9aZ1KXXV)'
st.markdown(drive, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model_cache():
    model = load_model(model_version=84)
    return model

test_generator = 0
c1, c2, c3= st.columns(3)
if uploaded:

    with NamedTemporaryFile("wb", suffix=".tif") as f:
        print("Hello, we are inside the uploaded function")
        f.write(uploaded.getvalue())
        path_to_temp = f.name # f.name is the path of the temporary file
        print(path_to_temp)
        im = Image.open(f.name)
        with c1:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.markdown("<h5 style='color: #131032;'>Image input</5>", unsafe_allow_html=True)
        c1.image(im, width=200)
        #c1.subheader('Image was loaded for model prediction')
        st.write("")
        with c2:
            if st.button("Predict on Model"):
                im_np = np.array(im)
                im_np = np.expand_dims(im_np, 0)

                # initialize model
                print("Hello")
                model = load_model_cache()
                print("Bye !")

                # preprocessing with MobileNetV2
                im_prep = preprocess_input(im_np)

                # prediction on model
                model_preds = model.predict(im_prep)

                with c3:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.markdown("<h5 style='color: #131032;'>Model prediction</5>", unsafe_allow_html=True)
                model_preds = model_preds.tolist()
                print(model_preds)
                c3.write("🧪 " + str(round(model_preds[0][0], 4)*100) + "%" + " probability of containing metastatic tissue")
