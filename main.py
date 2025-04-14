import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import streamlit as st

# -------------------------------
# Helper function to load external CSS
# -------------------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load external CSS (ensure style.css is in the same directory)
local_css("style.css")

# -------------------------------
# Sidebar Navigation
# -------------------------------
# Sidebar radio for page navigation.
page = st.sidebar.radio("Navigation", ["Sentiment Analysis", "About"])

# -------------------------------
# "About" Page
# -------------------------------
if page == "About":
    st.title("About IMDB Movie Review Sentiment Analysis")
    st.write("""
        This app leverages a fine-tuned BERT model to analyze the sentiment of movie reviews. 
        It not only classifies the review as **Positive** or **Negative**, but also provides several key metrics:
        
        - **Overall Sentiment Score:** The confidence level for the predicted sentiment.
        - **Detailed Class Probabilities:** Individual probabilities for both positive and negative classes.
        - **Review Token Count:** The number of tokens found in the review.
        
        Use this application to quickly gauge audience sentiment on movie reviews!
    """)
    st.info("Navigate to the 'Sentiment Analysis' tab from the sidebar to get started.")

# -------------------------------
# Sentiment Analysis Page
# -------------------------------
if page == "Sentiment Analysis":
    st.title('IMDB Movie Review Sentiment Analysis')
    st.markdown("### Enter a movie review below to classify its sentiment.")

    # Sidebar: Sample reviews (visible only on this page)
    with st.sidebar:
        st.header("Sample Reviews")
        sample_reviews = {
            "Positive": "I absolutely loved this movie. The plot was engaging and the characters were brilliant!",
            "Negative": "The movie was a complete waste of time. Poor acting and a predictable plot."
        }
        selected_sample = st.selectbox("Choose a sample review", list(sample_reviews.keys()))
        if st.button("Load Sample Review"):
            st.session_state['user_input'] = sample_reviews[selected_sample]

    # -------------------------------
    # Load Pre-trained BERT Model & Tokenizer
    # -------------------------------
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained("bert_imdb_final")  # Ensure this folder exists

    # -------------------------------
    # Helper Functions for Prediction
    # -------------------------------
    def preprocess_text(text, max_length=128):
        """
        Tokenizes and encodes the input text using the BERT tokenizer.
        """
        return tokenizer(text,
                         padding='max_length',
                         truncation=True,
                         max_length=max_length,
                         return_tensors='tf')

    def predict_sentiment(review):
        """
        Processes the review, gets the model's prediction, and returns:
            - the sentiment label,
            - overall confidence for the predicted class,
            - positive class probability,
            - negative class probability,
            - number of tokens in the review.
            
        Uses softmax to convert logits to probabilities.
        """
        # Tokenize the review for token count metric
        tokens = tokenizer.tokenize(review)
        token_count = len(tokens)
        
        # Preprocess and get model outputs
        inputs = preprocess_text(review)
        outputs = model(inputs)
        # Softmax for probability distribution
        probs = tf.nn.softmax(outputs.logits, axis=1)
        probs_np = probs.numpy()[0]
        
        # Index mapping assumed: 0 -> Negative, 1 -> Positive
        predicted_class = np.argmax(probs_np)
        sentiment = 'Positive' if predicted_class == 1 else 'Negative'
        confidence = probs_np[predicted_class]
        positive_prob = probs_np[1]
        negative_prob = probs_np[0]
        
        return sentiment, confidence, positive_prob, negative_prob, token_count

    # -------------------------------
    # Main App: User Input and Prediction Display
    # -------------------------------
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""
    
    user_input = st.text_area('Your Movie Review', st.session_state['user_input'], height=200)

    if st.button('Classify', key="classify_button", help="Click to classify the sentiment of the review"):
        if user_input.strip() == "":
            st.error("Please enter a movie review before classifying.")
        else:
            with st.spinner("Classifying..."):
                sentiment, conf, pos_prob, neg_prob, token_count = predict_sentiment(user_input)
            st.success("Classification Complete!")
            st.markdown(f"### Sentiment: **{sentiment}**")
            st.markdown(f"**Overall Confidence:** {conf:.4f}")
            st.markdown(f"**Review Token Count:** {token_count}")
    else:
        st.info("Please enter a movie review and click the **Classify** button to see the sentiment prediction.")
