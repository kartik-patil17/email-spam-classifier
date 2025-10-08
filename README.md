# email-spam-classifier

# Email/SMS Spam Classifier

This project is a machine learning-based application designed to classify text messages (SMS or emails) as either "Spam" or "Not Spam" (Ham). The model is built using a Multinomial Naive Bayes classifier and integrated into a simple, interactive web application using Streamlit.



## Table of Contents
- [Features](#features)
- [Methodology](#methodology)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)

---

## Features
- **Text Classification**: Classifies input text into Spam or Ham.
- **NLP Preprocessing**: Implements a standard NLP pipeline including lowercasing, tokenization, removal of special characters & stopwords, and stemming.
- **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical vectors.
- **Machine Learning Model**: Employs a Multinomial Naive Bayes (MNB) model, which is highly effective for text classification tasks.
- **Interactive UI**: A user-friendly web interface built with Streamlit to test the model with custom messages.
- **Data Visualization**: The training notebook includes various visualizations like pie charts, histograms, and word clouds for exploratory data analysis.

---

## Methodology
The project follows a structured machine learning workflow, detailed in the `sms-spam-detection.ipynb` notebook.

1.  **Data Cleaning and EDA**:
    - The dataset (`spam.csv`) is loaded, and unnecessary columns are removed.
    - Columns are renamed to `target` and `text`.
    - The target variable is label-encoded (spam=1, ham=0).
    - Duplicates are identified and removed.
    - Exploratory Data Analysis (EDA) is performed to understand the data distribution. This includes checking the class balance and analyzing message characteristics (length, word count, sentence count) for both classes.
    - Word clouds are generated to visualize the most frequent words in spam and ham messages.

2.  **Text Preprocessing**:
    - A custom function `transform_text` is created to process the raw text data. This involves:
        1.  Converting text to lowercase.
        2.  Tokenizing the text into words.
        3.  Removing non-alphanumeric characters.
        4.  Filtering out stopwords (common words like 'the', 'a', 'in') and punctuation.
        5.  Applying stemming using the Porter Stemmer to reduce words to their root form (e.g., 'loving' -> 'love').

3.  **Model Building**:
    - The preprocessed text data is converted into numerical feature vectors using `TfidfVectorizer` with a maximum of 3000 features.
    - The dataset is split into training (80%) and testing (20%) sets.
    - Three different Naive Bayes classifiers are trained and evaluated: GaussianNB, MultinomialNB, and BernoulliNB.
    - The models' performance is compared based on accuracy and precision. MultinomialNB was chosen for its high precision (1.0) and accuracy (97%).

4.  **Model Export**:
    - The trained `TfidfVectorizer` and the `MultinomialNB` model are serialized and saved as pickle files (`vectorizer.pkl` and `model.pkl`) for use in the web application.

---

## Technology Stack
- **Language**: Python 3
- **Libraries**:
    - **ML/NLP**: Scikit-learn, NLTK
    - **Data Manipulation**: Pandas, NumPy
    - **Data Visualization**: Matplotlib, Seaborn, WordCloud
    - **Web Framework**: Streamlit
- **Development Environment**: Jupyter Notebook

---

## Project Structure


---

## Setup and Installation
Follow these steps to set up and run the project on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content:
    ```
    pandas
    numpy
    nltk
    scikit-learn
    matplotlib
    seaborn
    wordcloud
    streamlit
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset**:
    - Download the "SMS Spam Collection Dataset" from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
    - Unzip the file and place `spam.csv` in the root directory of the project.

---

## How to Run
1.  **Open the Jupyter Notebook (Optional)**:
    To explore the data analysis and model training process, run the Jupyter Notebook:
    ```bash
    jupyter notebook sms-spam-detection.ipynb
    ```

2.  **Run the Streamlit Web App**:
    To start the interactive spam classifier application, run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    This will open a new tab in your web browser with the application running.

---

## Model Performance
The final model chosen was **Multinomial Naive Bayes** with TF-IDF features (`max_features=3000`). Its performance on the test set was:
- **Accuracy**: 97.1%
- **Precision**: 100%

Precision was prioritized in this task. A high precision score ensures that when the model predicts a message as "Spam," it is very likely to be correct. This is crucial to minimize the risk of legitimate messages (Ham) being incorrectly classified as spam (false positives).
