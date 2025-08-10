"""
Resume Category Prediction Streamlit App

This script creates a web application using Streamlit to predict the category of a resume.
Users can upload a resume file (PDF, DOCX, or TXT), and the app will use a pre-trained
machine learning model to classify it into a predefined job category.
"""

import streamlit as st
import pickle
import re
import docx
import PyPDF2

# --- 1. LOAD THE TRAINED MODELS ---
# Load the pre-trained machine learning model (classifier).
# This model is responsible for the actual classification task.
svc_model = pickle.load(open('model.pkl', 'rb'))

# Load the saved TF-IDF Vectorizer.
# This transforms the raw resume text into numerical features that the model can understand.
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Load the saved Label Encoder.
# This is used to convert the model's numerical predictions back into human-readable category names.
le = pickle.load(open('encoder.pkl', 'rb'))


# --- 2. HELPER FUNCTIONS ---

def cleanResume(resume_text):
    """
    Cleans the input resume text by removing URLs, mentions, hashtags, special characters,
    and extra whitespace.

    Args:
        resume_text (str): The raw text extracted from the resume.

    Returns:
        str: The cleaned resume text.
    """
    cleanText = re.sub('http\S+\s', ' ', resume_text)  # remove URLs
    cleanText = re.sub('RT|cc', ' ', cleanText)  # remove RT and cc
    cleanText = re.sub('#\S+\s', ' ', cleanText)  # remove hashtags
    cleanText = re.sub('@\S+', '  ', cleanText)  # remove mentions
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)  # remove punctuations
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # remove non-ASCII characters
    cleanText = re.sub('\s+', ' ', cleanText)  # remove extra whitespace
    return cleanText


def extract_text_from_pdf(file):
    """
    Extracts text from an uploaded PDF file.

    Args:
        file: The uploaded PDF file object from Streamlit.

    Returns:
        str: The extracted text content.
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    """
    Extracts text from an uploaded DOCX file.

    Args:
        file: The uploaded DOCX file object from Streamlit.

    Returns:
        str: The extracted text content.
    """
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


def extract_text_from_txt(file):
    """
    Extracts text from an uploaded TXT file, handling potential encoding issues.

    Args:
        file: The uploaded TXT file object from Streamlit.

    Returns:
        str: The extracted text content.
    """
    try:
        # Try decoding with 'utf-8' first
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # Fallback to 'latin-1' if 'utf-8' fails
        text = file.read().decode('latin-1')
    return text


def handle_file_upload(uploaded_file):
    """
    Determines the file type and calls the appropriate text extraction function.

    Args:
        uploaded_file: The file object uploaded by the user.

    Returns:
        str: The extracted text from the file.
    
    Raises:
        ValueError: If the uploaded file type is not supported.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")


def pred(input_resume):
    """
    Takes cleaned resume text, vectorizes it, and predicts the category.

    Args:
        input_resume (str): The text of the resume.

    Returns:
        str: The predicted category name.
    """
    # 1. Clean the input text
    cleaned_text = cleanResume(input_resume)

    # 2. Vectorize the text using the pre-trained TF-IDF vectorizer
    vectorized_text = tfidf.transform([cleaned_text])

    # 3. Convert the sparse matrix to a dense array for prediction
    vectorized_text = vectorized_text.toarray()

    # 4. Predict the category using the trained classifier
    predicted_category_id = svc_model.predict(vectorized_text)

    # 5. Convert the predicted category ID back to its original name
    predicted_category_name = le.inverse_transform(predicted_category_id)

    return predicted_category_name[0]


# --- 3. STREAMLIT APP LAYOUT ---

def main():
    """
    Main function to define the Streamlit application's layout and functionality.
    """
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="Resume Category Prediction",
        page_icon="📄",
        layout="wide"
    )

    # --- APP TITLE AND DESCRIPTION ---
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format to predict its job category.")

    # --- FILE UPLOAD WIDGET ---
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    # --- PROCESSING AND PREDICTION LOGIC ---
    if uploaded_file is not None:
        try:
            # Step 1: Extract text from the uploaded file
            resume_text = handle_file_upload(uploaded_file)
            st.success("Successfully extracted text from the resume.")

            # Step 2: Display the extracted text if the user chooses to
            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Step 3: Predict the category and display the result
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.info(f"The predicted category for this resume is: **{category}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
