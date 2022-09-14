import streamlit as st
from PIL import Image
import numpy as np
from project_cancer_detection.interface.main_local import load_model
from tempfile import NamedTemporaryFile



path_to_test = '/home/naz/code/GuillaumeRib/project-cancer-detection/raw_data/test_small'

# the page icon
st.set_page_config("Deep Learning for Cancer Diagnosis", "🔬")

# bg_img = '''
#     <style>
#     section.css-k1vhr4.egzxvld3 {
#     background-image: url("https://i.imgur.com/3BOSVg9.png");
#     background-size: cover;
#     }
#     </style>
#     '''

# # cool image link: https://i.imgur.com/3BOSVg9.png

# # the whole page: css-18ni7ap e8zbici2
# # upper part is white: css-1wrcr25.egzxvld4

# st.markdown(bg_img, unsafe_allow_html=True)


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
    st.header('Deep Learning for Cancer Diagnosis')

# upload image
st.markdown("<h6 style='text-align: center; color: grey;'>Insert image for tumor detection</h6>", unsafe_allow_html=True)
uploaded= st.file_uploader("", type=['tif'])

@st.cache(allow_output_mutation=True)
def load_model_cache():
    model = load_model(model_version=53)
    return model

test_generator = 0
c1, c2, c3= st.columns(3)
if uploaded:
    with NamedTemporaryFile("wb", suffix=".tif") as f:
        print("Hello, we are inside the uploaded function")
        f.write(uploaded.getvalue())
        path_to_temp = f.name # f.name is the path of the temporary file
        im = Image.open(f.name)
        with c1:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.markdown("<h5 style=color: grey;'>Image input</5>", unsafe_allow_html=True)
        c1.image(im, width=200)
        #c1.subheader('Image was loaded for model prediction')
        st.write("")
        with c2:
            if st.button("Predict on Model"):
                im_np = np.array(im)
                im_np = np.expand_dims(im_np, 0)
                #print(f'Shape of numpu.array ---> {im_np.shape}')
                # initialize model
                model = load_model_cache()

                # prediction on model
                model_preds = model.predict(im_np)

                #model_pred_classes = np.argmax(model_preds, axis=1)
                with c3:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.markdown("<h5 style=color: grey;'>Model prediction</5>", unsafe_allow_html=True)
                model_preds = model_preds.tolist()
                c3.write("🧪 " + str(round(model_preds[0][0], 2)) + "%" + " of tumor detection")
