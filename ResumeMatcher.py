import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a Word document
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to calculate similarity between job description and resume
def calculate_similarity(job_description, resume_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1]
    return similarity_score * 100

# Main function
def main():
    st.title("Resume Matcher")

    # File uploader for resume
    st.subheader("Upload your resume (in .docx format)")
    uploaded_file = st.file_uploader("Choose a file", type=['docx'])

    # Text area for job description
    st.subheader("Enter the job description")
    job_description = st.text_area("Paste the job description here")

    # Compare button
    if st.button("Compare"):
        if uploaded_file is not None and job_description:
            resume_text = extract_text_from_docx(uploaded_file)
            similarity_score = calculate_similarity(job_description, resume_text)
            st.subheader("Matching Score")
            st.write(f"The similarity between your resume and the job description is: {similarity_score:.2f}%")

# Display the Streamlit app in the notebook
main()
