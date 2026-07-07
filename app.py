import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="Gender Classification AI",
    page_icon="🧠",
    layout="wide"
)

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>

html, body, [class*="css"]{
    background:#0B0F19;
    color:white;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    text-align:center;
    font-size:58px;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#A0A0A0;
    font-size:22px;
    margin-bottom:40px;
}

.result-card{
    background:#151B26;
    padding:25px;
    border-radius:25px;
    text-align:center;
    box-shadow:0px 10px 35px rgba(0,0,0,.4);
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# Load Model
# ------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("male_female_classifier.keras")

model = load_model()

# ------------------------
# Header
# ------------------------

st.markdown(
    "<div class='title'>👤 Gender Classification AI</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='subtitle'>Deep Learning • TensorFlow • Streamlit</div>",
    unsafe_allow_html=True,
)

# ------------------------
# Upload
# ------------------------

uploaded = st.file_uploader(
    "Upload Face Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    col1,col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = image.resize((128,128))

    img = np.array(img)/255.0

    img = np.expand_dims(img,axis=0)

    with st.spinner("Analyzing image..."):

        prediction = model.predict(img,verbose=0)[0][0]

    if prediction>0.5:

        label="👨 Male"
        confidence=prediction*100

    else:

        label="👩 Female"
        confidence=(1-prediction)*100

    with col2:

        st.markdown(
        f"""
        <div class='result-card'>

        <h2>Prediction</h2>

        <h1>{label}</h1>

        <br>

        <h3>Confidence</h3>

        </div>

        """,
        unsafe_allow_html=True
        )

        st.progress(confidence/100)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.success("Prediction Complete")

st.markdown("---")

st.caption(
"Built with ❤️ using TensorFlow & Streamlit"
)
