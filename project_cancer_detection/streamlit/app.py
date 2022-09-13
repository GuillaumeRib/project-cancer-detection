import streamlit as st
from streamlit_image_comparison import image_comparison
import cv2
from PIL import Image

# Helping Pathologists in Cancer detection thanks to Deep Learning
st.set_page_config("Deep Learning for Cancer Diagnosis", "🔬")
st.write("")


st.image(
    "https://images.emojiterra.com/twitter/v14.0/512px/1f9ec.png",
    width=120,
)
st.header("Deep Learning for Cancer Diagnosis")
st.write("")
st.markdown('**Description:**')
"Models are trained on a single GPU in a couple hours, and achieved competitive scores to identify metastatic tissue in histopathologic scans of lymph node sections (tumor detection and whole-slide image diagnosis). **[I want to know more](https://www.kaggle.com/competitions/histopathologic-cancer-detection)**"
st.write("")


st.markdown("### Examples of input images")
image_comparison(
    img1="https://www.kaggleusercontent.com/kf/9245458/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..40OugCNodb0vDIdou2v3WQ.9E0JmMsUL90KAkQKbem_QQPUgiTXPTqZs1kOSCAvSFyZIT8REwdw1-ofLlXbEa2C_3DqhcuuswaUdknnM5bNP23tLXMeoq3jTMQIFDnTGe_34uqI0owbTBM_638sEHxn1UwogGOtp8FEOOS-UP6vMj5rgcMPQkwBeG6ZV0ySVpK2SW2uEzms2SO5z-KX-l0pOCkeeYYgNqZHgHd2PGfP7X6m9oY5seCLDzPsX6TqEc4T8w6YuPM8naRmWQt5MD2B6QDbHM4QOC2efWFj-ZuvtViqrHz1GmR1ndB2nFC4Ajlolc72cZVh1_Vldc7Q1YoM0ioskNVNIT3rSQmXX7t7KJIV-l9H08pGWb245h4xeBIc6IshQEjL_LLnfRGsRWORgI7MPP7bZYz8PqZu9IVZwD6KY_HbklkACAXjuCjKx5zZiGWfJ7-x57bDMv7FYJDBDl6SGSoEqy0gee22OJoHAod80HBmdcvhv2nnGPsMO6Dq-Zsc9SwSMZ_GnNFejyLfK7R7t8vl9XcaAxB3XNVc_t4nHprzeuTmRufVTkzN7Lc54H1Mk7PnrxZlSYPWKrhf-lTYPB2uEIwAAjApfpXVtehKOVv3dTKMyeUS3DEi7l4BUfWY609N26ATeUF9e0keA8IY29ha7FuTrxTKiunFdrYZrtLN7ZANBi8F8VIQxYM.tVeoaVTcAwv9Bnv60vy1_Q/__results___files/__results___5_0.png",
    img2="https://www.kaggleusercontent.com/kf/9245458/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..40OugCNodb0vDIdou2v3WQ.9E0JmMsUL90KAkQKbem_QQPUgiTXPTqZs1kOSCAvSFyZIT8REwdw1-ofLlXbEa2C_3DqhcuuswaUdknnM5bNP23tLXMeoq3jTMQIFDnTGe_34uqI0owbTBM_638sEHxn1UwogGOtp8FEOOS-UP6vMj5rgcMPQkwBeG6ZV0ySVpK2SW2uEzms2SO5z-KX-l0pOCkeeYYgNqZHgHd2PGfP7X6m9oY5seCLDzPsX6TqEc4T8w6YuPM8naRmWQt5MD2B6QDbHM4QOC2efWFj-ZuvtViqrHz1GmR1ndB2nFC4Ajlolc72cZVh1_Vldc7Q1YoM0ioskNVNIT3rSQmXX7t7KJIV-l9H08pGWb245h4xeBIc6IshQEjL_LLnfRGsRWORgI7MPP7bZYz8PqZu9IVZwD6KY_HbklkACAXjuCjKx5zZiGWfJ7-x57bDMv7FYJDBDl6SGSoEqy0gee22OJoHAod80HBmdcvhv2nnGPsMO6Dq-Zsc9SwSMZ_GnNFejyLfK7R7t8vl9XcaAxB3XNVc_t4nHprzeuTmRufVTkzN7Lc54H1Mk7PnrxZlSYPWKrhf-lTYPB2uEIwAAjApfpXVtehKOVv3dTKMyeUS3DEi7l4BUfWY609N26ATeUF9e0keA8IY29ha7FuTrxTKiunFdrYZrtLN7ZANBi8F8VIQxYM.tVeoaVTcAwv9Bnv60vy1_Q/__results___files/__results___7_0.png",
    label1="Negative",
    label2="Positive",
)
st.write("_image source: [AIMDATA](https://www.kaggle.com/code/aimdata/cancer-detection-using-cnn/notebook)_")
st.markdown("")
st.write("")

st.markdown("**About models:**")
"Models are trained on a single GPU in a couple hours, and achieved competitive scores to identify metastatic tissue in histopathologic scans of lymph node sections (tumor detection and whole-slide image diagnosis). **[I need more explanation](https://www.kaggle.com/competitions/histopathologic-cancer-detection)**"
st.write("")

st.write("")
"Models are trained on a single GPU in a couple hours, and achieved competitive scores to identify metastatic tissue in histopathologic scans of lymph node sections (tumor detection and whole-slide image diagnosis)."
st.write("")


st.markdown("### Data Driven Future: cancer detection using DL & ML")
image_comparison(
    img1="https://ars.els-cdn.com/content/image/1-s2.0-S1876034120305633-gr1_lrg.jpg",
    img2="https://ars.els-cdn.com/content/image/1-s2.0-S1876034120305633-gr6_lrg.jpg",
    label1="ML application in cancer detection",
    label2="DCNN process for breast cancer detection",
)
st.write("_Source: [TanzilaSaba](https://www.sciencedirect.com/science/article/pii/S1876034120305633)_")
