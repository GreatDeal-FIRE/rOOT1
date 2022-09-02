# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image, ImageFilter  # Import classes from the library.
# preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
# modeling
from sklearn import svm
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# creating page sections
site_header = st.container()
business_context = st.container()
data_desc = st.container()
performance = st.container()
tweet_input = st.container()
model_results = st.container()
sentiment_analysis = st.container()
contact = st.container()

with site_header:
    st.title('HATE SPEECH DETECTION ON TWITTER')
    st.write("""
    Created by 💕 [Rieko Brian] https://github.com/GreatDeal-FIRE)

    HATE SPEECH DETECTION IN KENYA

    Computers possess have  cognitive abilities to understand the human language the way we do unless we train them to  .This project aimed at  aimed at training computer models to identify/classify hate speech on twitter(Location :**KENYA**)

    Therefore, this project aims to automate content moderation to identify hate speech using machine learning techniques in Kenya.

    Kenya is a multilingual country [with over 42 tribes];

    Kenyans tweet in English & Swahili [Others mix their tweets with a bit of Vernacular]. This also occurs when people are conversating. This is what is known as Codeswitching (Speaker alternates between two or more languages when either speaking or in this case **TWEETING**)

    **DEFINATIONS OF HATE SPEECH.**

    Hate speech is defined in a variety of ways in the literature, and there is **no universal definition**. Other organizations, such as international and domestic legislation, have attempted to define hate speech and even identified specific targets based on what are legally known as protected traits, such as race, ethnic origin, religion, or gender.

    Here are a few definitions of what hate **speech entails;**

    **[Twitter hateful conduct policy]**: “You may not promote violence against or directly attack or threaten other people on the basis of race, ethnicity, national origin, sexual orientation, gender, gender identity, religious affiliation, age, disability, or disease. We also do not allow accounts whose primary purpose is inciting harm towards others on the basis of these categories.” 

    **[Kenya National Cohesion and Integration Commission (NCIC) Act, 2008]**: “Content that promotes violence against or has the primary purpose of inciting hatred against individuals or groups based on certain attributes, such as race or ethnic origin, religion, disability, gender, age, veteran status, sexual orientation/gender identity.” 

    **[The European Court of Human Rights]**: “All forms of expression which spread, incite, promote or justify racial hatred, xenophobia, anti-Semitism or other forms of hatred based on intolerance, including intolerance expressed by aggressive nationalism and ethnocentrism, discrimination and hostility towards minorities, migrants and people of immigrant origin.”

    **Hate speech should not be conflicted with freedom of speech**

    There is freedom of speech for what to speak/air out their opinions, but these should not give one a room to be offensive or use hate speech.

    Check out the project repository [here](https://github.com/sidneykung/twitter_hate_speech_detection).
    
    """)

with business_context:
    st.header('The Problem of Content Moderation')
    st.write("""
    
    Content moderation on social media is a big issue in modern times despite the current existing technology . 
    The amount of content flowing on social media especially in is so much that human moderators cannot annotate all at a given period of time ⏳. 
    
    Human content moderation will be expensive as it will involve a lot of employees to work on a certain amount of data.
    
    
    """)
    
with data_desc:
    understanding, venn = st.columns(2)
    with understanding:
        st.text('')
        st.write("""
        The **data** for this project was sourced from kaggle [study](https://www.kaggle.com/datasets/edwardombui/balanced2) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.
        
        The `.csv` file has **50,174 rows** where **6.46% of the tweets were labeled as "Hate Speech".**

        Each tweet's label was voted on by a group of selected individuals and determined by majority rules.
        """)
    with venn:
        st.image(Image.open(r'C:\Users\Ricky\Desktop\4.2 FINAL SEMESTER\PROJECT II  Computer systems Project\rOOT\Preprocessing\visualizations\word_venn.png'), width = 400)


with performance:
    description, conf_matrix = st.columns(2)
    with description:
        st.header('Final Model Performance')
        st.write("""
        These scores are indicative of the two major roadblocks of the project:
        - The massive class imbalance of the dataset.
        - Too much Noise (lack of Swahili corpus for stop words).
        - The model's inability to identify what constitutes as hate speech.
        """)
    with conf_matrix:
        st.image(Image.open(r'C:\Users\Ricky\Desktop\4.2 FINAL SEMESTER\PROJECT II  Computer systems Project\rOOT\Preprocessing\visualizations\normalized_log_reg_countvec_matrix.png'), width = 400)

with tweet_input:
    st.header('Is Your Tweet Considered Hate Speech?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an ACCURATE representation.*""")
    # user input here
    user_text = st.text_input('Enter Tweet', max_chars=280) # setting input as user_text


with model_results:    
    st.subheader('Prediction:')
    if user_text:
    # processing user_text
        # removing punctuation
        user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        # tokenizing
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
        # taking root word
        lemmatizer = WordNetLemmatizer() 
        lemmatized_output = []
        for word in stopwords_removed:
            lemmatized_output.append(lemmatizer.lemmatize(word))

        # instantiating count vectorizor
        count = CountVectorizer(stop_words=stop_words)
        X_train = pickle.load(open(r'C:\Users\Ricky\Desktop\4.2 FINAL SEMESTER\PROJECT II  Computer systems Project\rOOT\Preprocessing\pickle\X_train_2.pkl', 'rb'))
        X_test = lemmatized_output
        X_train_count = count.fit_transform(X_train)
        X_test_count = count.transform(X_test)


        # loading in model
        final_model = pickle.load(open(r'C:\Users\Ricky\Desktop\4.2 FINAL SEMESTER\PROJECT II  Computer systems Project\rOOT\Preprocessing\pickle\final_log_reg_count_model.pkl', 'rb'))

        # apply model to make predictions
        prediction = final_model.predict(X_test_count[0])

        if prediction ==0:
            st.subheader('**Not Hate Speech**')
        else:
            st.subheader('**Hate Speech**')
        st.text('')



with sentiment_analysis:
    if user_text:
        st.header('Sentiment Analysis with VADER')
        
        # explaining VADER
        st.write("""
        
        *VADER is a lexicon designed for scoring social media.*""")
        # spacer
        st.text('')
    
        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer() 
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text) 
        if sentiment_dict['compound'] >= 0.05 : 
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05 : 
            category = ("**Negative 🚫**") 
        else : 
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            # printing category
            st.write("Your Tweet is rated as", category) 
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**") 
            st.write(sentiment_dict['neg']*100, "% Negative") 
            st.write(sentiment_dict['neu']*100, "% Neutral") 
            st.write(sentiment_dict['pos']*100, "% Positive") 
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph) 

with contact:
    st.markdown("---")
    st.header('For More Information')
    st.text('')
    st.write("""

    **Check out the project repository [here].**

    Contact .
    """)

    st.subheader("Let's Connect!")
    st.write("""
    
        
          
    """)