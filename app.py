import streamlit as st
import joblib


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Research Paper Category Predictor",
    page_icon="📄",
    layout="centered"
)

# -------------------------------------------------
# Background Image + Styling (ONLY ONE IMAGE)
# -------------------------------------------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("data:image/jpeg;base64,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");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        .main {
            background-color: rgba(255, 255, 255, 0.80);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.35);
        }

        h1, h2, h3, p, label {
            color: #1e1e1e !important;
            font-weight: 600;
        }

        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 10px;
            border: 1px solid #cccccc;
        }

        .stButton button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.6em 1.2em;
        }

        .stButton button:hover {
            background-color: #1e40af;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------
# -------------------------------------------------
# Load Trained Model (PICKLE - CORRECT WAY)
# -------------------------------------------------
import streamlit as st
import joblib

model = joblib.load("_MLProject2_.joblib")



# -------------------------------------------------
# Title Section
# -------------------------------------------------
st.markdown(
    """
    <div class="main">
        <h1>📄 Research Paper Category Predictor</h1>
        <p>
        Paste the abstract of a research paper below and the system will predict
        the most relevant academic category.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Input Text Area
# -------------------------------------------------
abstract = st.text_area(
    "📝 Enter Research Paper Abstract",
    height=220,
    placeholder="Paste the research paper abstract here..."
)

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.button("🔍 Predict Category"):
    if abstract.strip() == "":
        st.warning("⚠️ Please enter an abstract.")
    else:
        prediction = model.predict([abstract])[0]

        st.success("✅ Prediction Successful!")
        st.markdown(
            f"""
            <div class="main">
                <h3>📌 Predicted Category</h3>
                <h2 style="color:#2563eb;">{prediction}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#1e1e1e;">
        Built using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
