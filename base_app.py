"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:
    https://docs.streamlit.io/en/latest/
"""

# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Vectorizer
vectorizer_path = os.path.join(current_dir, "tfidfvect.pkl")
news_vectorizer = open(vectorizer_path, "rb")
test_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analyzing news articles")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Home", "Information", "Prediction", "EDA", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Home" page
    if selection == "Home":
        st.info("Welcome to the News Classifier App")
        st.markdown("""
        ## Welcome to the News Classifier Application
        This application allows you to classify news articles into different categories using various machine learning models. Explore the different pages to learn more about the app, classify news articles, and view exploratory data analysis.
        """)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown("""
        ## News Classifier Application
        This application classifies news articles into different categories using various pre-trained machine learning models.

        ### Models Used
        - **Logistic Regression:** A linear model for binary classification.
        - **Random Forest:** An ensemble learning method for classification.
        - **Support Vector Machine:** A supervised learning model for classification.

        ### Dataset
        The models were trained on a dataset of labeled news articles. The dataset contains various categories of news, providing a diverse set of examples for training.

        ### Purpose
        The purpose of this application is to demonstrate the capabilities of machine learning models in classifying news articles. It serves as a tool for analyzing the content of news articles and understanding the distribution of topics.

        ### How to Use
        1. Navigate to the "Prediction" page.
        2. Enter the text of the news article you want to classify.
        3. Select the model you wish to use from the sidebar.
        4. Click the "Classify" button to see the predicted category.

        ### References
        - [Streamlit Documentation](https://docs.streamlit.io/en/latest/)
        - [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
        - [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
        """)

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter News Article", "Type Here")
        
        model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Support Vector Machine"])

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = test_cv.transform([news_text]).toarray()
            
            # Load and apply the chosen model
            if model_choice == "Logistic Regression":
                model_path = os.path.join(current_dir, "logistic_regression.pkl")
            elif model_choice == "Random Forest":
                model_path = os.path.join(current_dir, "rf_classifier_model.pkl")
            elif model_choice == "Support Vector Machine":
                model_path = os.path.join(current_dir, "svm_classifier_model.pkl")

            predictor = joblib.load(open(model_path, "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("News Article categorized as: {}".format(prediction[0]))

    # Building out the EDA page
    if selection == "EDA":
        st.info("Exploratory Data Analysis")
        st.markdown("### Category Distribution")
        st.image(os.path.join(current_dir, "Category EDA.png"))
        st.markdown("### Frequent Words in Articles")
        st.image(os.path.join(current_dir, "Frequent words.png"))

    # Building out the "About Us" page
    if selection == "About Us":
        st.info("About Us")
        st.markdown("""
        ## About Us

        ### Mission Statement
        Our mission is to leverage machine learning to automate the classification of news articles, providing insights into the distribution of news topics and aiding in content analysis.

        ### Team Members
        - **John Doe:** Data Scientist
        - **Jane Smith:** Machine Learning Engineer
        - **Alice Johnson:** Software Developer
        - **Bob Lee:** Project Manager

        ### Technologies Used
        - **Programming Languages:** Python
        - **Frameworks:** Streamlit, scikit-learn
        - **Libraries:** pandas, joblib, matplotlib

        ### Contact Information
        - **Email:** contact@exploreai.com
        - **Phone:** +123-456-7890
        - **Address:** 123 AI Street, Data City, ML State, 56789

        ### Future Plans
        We plan to expand the functionality of this application by incorporating more advanced models and additional features, such as real-time news analysis and integration with news APIs.

        Thank you for using our News Classifier application!
        """)

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
