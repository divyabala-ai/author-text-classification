# Author Classification

## Overview
This project predicts the author of a given text using literary works from the [Project Gutenberg](https://www.gutenberg.org/) digital library.  
The workflow covers text preprocessing, multiple feature extraction techniques (Bag of Words, TF-IDF, Word2Vec), and training various supervised and deep learning models including LSTM and BERT.

## Project Workflow
1. **Data Exploration**  
   - Collected texts from at least 10 authors, each with multiple works.  
   - Analyzed text length, vocabulary size, word frequency distributions, and class balance.

2. **Text Preprocessing**  
   - Removed punctuation, lowercased text, and cleaned unwanted characters.  
   - Tokenized using NLTK and spaCy, removed stopwords, and applied lemmatization.

3. **Feature Extraction & Model Training**  
   - **Bag of Words**: Trained Logistic Regression, Decision Tree, Random Forest, KNN, and Gradient Boosting. Tuned models with GridSearchCV.  
   - **TF-IDF**: Repeated model training and tuning.  
   - **Word2Vec**: Trained embeddings and applied supervised models.  

4. **Deep Learning Models**  
   - Implemented LSTM for sequence modeling.  
   - Applied BERT using Hugging Face Transformers for contextualized embeddings and classification.

5. **Topic Modeling**  
   - Applied LSA, LDA, and NMF to identify topics in the text.  
   - Compared top 10 words per topic for interpretability.

6. **Model Comparison & Findings**  
   - BERT achieved the highest accuracy across all methods.  
   - Traditional supervised models performed better with TF-IDF than Bag of Words.  
   - Word2Vec embeddings improved performance for some models but required more computation.

## Technologies Used
- Python  
- NLTK, spaCy  
- Scikit-learn, Gensim  
- TensorFlow / Keras  
- Transformers (Hugging Face)  
- Matplotlib, Seaborn  

## Dataset
Text data collected from the [Project Gutenberg](https://www.gutenberg.org/) library,  
including works from at least 10 authors.
