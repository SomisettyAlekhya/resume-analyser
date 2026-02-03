import streamlit as st
import joblib
import docx
import PyPDF2
import re

# ------------------------
# Load saved models
# ------------------------
svc_model = joblib.load("svm_resume_classifier.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_1.joblib")

# ------------------------
# Text cleaning function
# ------------------------
def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------------
# Extract text from files
# ------------------------
def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
        return text
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        return None

# ------------------------
# Prediction function
# ------------------------
def predict_category(resume_text):
    cleaned = clean_text(resume_text)
    vectorized = tfidf.transform([cleaned])
    predicted = svc_model.predict(vectorized)
    category = label_encoder.inverse_transform(predicted)
    return category[0]

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Resume Analyzer", page_icon="🧩", layout="wide")

# ------------------------
# Theme toggle button
# ------------------------
theme_choice = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

# Colors based on theme
if theme_choice == "Light":
    bg_color = "#f0f4f8"
    text_color = "#000000"
    header_bg = "#4B77BE"
    resume_bg = "#ffffff"
    resume_border = "#4B77BE"
    pred_box_bg = "#FFFAE6"
    pred_text_color = "#FF6B35"
else:  # Dark theme
    bg_color = "#1e1e1e"
    text_color = "#FFFFFF"
    header_bg = "#256D85"
    resume_bg = "#2e2e2e"
    resume_border = "#3caea3"
    pred_box_bg = "#444444"
    pred_text_color = "#FFA500"

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Helvetica', sans-serif;
    }}
    .main-header {{
        background-color: {header_bg};
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }}
    .resume-box {{
        background-color: {resume_bg};
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid {resume_border};
        box-shadow: 1px 1px 6px rgba(0,0,0,0.1);
        color: {text_color};
    }}
    .prediction-box {{
        background-color: {pred_box_bg};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: {pred_text_color};
        box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
    }}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧩 Resume Analyzer & Categorizer</div>', unsafe_allow_html=True)
st.write("\n")

# File uploader
uploaded_file = st.file_uploader("Upload Resume (PDF, Word, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)
    
    if resume_text:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Resume Preview")
            st.markdown(f'<div class="resume-box">{resume_text[:1000]}...</div>', unsafe_allow_html=True)
            with st.expander("Expand to view full resume"):
                st.markdown(f'<div class="resume-box">{resume_text}</div>', unsafe_allow_html=True)
        
        with col2:
            category = predict_category(resume_text)
            st.subheader("Predicted Category")
            st.markdown(f'<div class="prediction-box">{category}</div>', unsafe_allow_html=True)
            
            category_descriptions = {
                "Data Science": "Focused on analytics, ML, AI, and data-driven insights.",
                "Testing": "QA and software testing professional.",
                "Development": "Software developer or engineer, building applications.",
                "Management": "Project management and leadership roles.",
                "Design": "UI/UX and creative design roles.",
                "Cloud": "Cloud engineering, AWS, Azure, or DevOps."
            }
            desc = category_descriptions.get(category, "No description available.")
            st.info(desc)
            
        st.success("Resume processed successfully!")
    else:
        st.error("This file type is not supported yet.")
else:
    st.info("Upload a resume to see its predicted category.")
