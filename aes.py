import streamlit as st
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import nltk
from textblob import TextBlob
import spacy
import re
from transformers import AutoTokenizer, AutoModel
import torch
import language_tool_python
import math
from PIL import Image
spacy.cli.download("en_core_web_sm")
# Load the model and scaler
df2 = pd.read_csv('essayembed.csv')
df2.drop(['TEXT'], axis=1, inplace=True)
df2['SCORE'] = df2['cEXT'] + df2['cNEU'] + df2['cAGR'] + df2['cCON'] + df2['cOPN']
df2['EMBEDDINGS'] = [np.fromstring(i.strip('[]'), sep=' ') for i in df2['EMBEDDINGS']]
df2['embedding_norm'] = df2['EMBEDDINGS'].apply(lambda x: np.linalg.norm(x))

Q1 = df2['embedding_norm'].quantile(0.15)
Q3 = df2['embedding_norm'].quantile(0.85)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df2 = df2[(df2['embedding_norm'] >= lower_bound) & (df2['embedding_norm'] <= upper_bound)]
df2 = df2.drop(columns=['embedding_norm'])

x = df2['EMBEDDINGS'].apply(pd.Series)
y = df2['SCORE']
ros = RandomOverSampler(random_state=42)
x, y = ros.fit_resample(x, y)

sc = StandardScaler()
x = sc.fit_transform(x)

xgb_regressor = XGBRegressor(n_estimators=500, learning_rate=0.2)
xgb_regressor.fit(x, y)

# Language model setup
at = AutoTokenizer.from_pretrained('bert-base-uncased')
am = AutoModel.from_pretrained('bert-base-uncased')


# Grammar check function
def grammar_score(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    critical_errors = [match for match in matches if match.ruleIssueType in ('misspelling', 'grammar')]
    error_count = len(critical_errors)
    wcount=len(text.split())
    score = max(0, 1-(error_count/(wcount*0.1)))
    return score,error_count

# Structure score function
def structure_score(text):
    blob = TextBlob(text)
    sentence_lengths = [len(sentence.split()) for sentence in blob.sentences]
    if len(sentence_lengths) < 6:
        return 0.2
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    if avg_sentence_length < 10:
        score = avg_sentence_length / 10
    elif avg_sentence_length > 20:
        score = max(0, 1 - (avg_sentence_length - 20) / 10)
    else:
        score = 1
    return round(score, 2)


# Flow score function
def flow_score(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    transitions = {"however", "therefore", "moreover", "thus", "although", "meanwhile"}
    count_transitions = sum(1 for token in doc if token.text.lower() in transitions)
    score = min(5, count_transitions)
    score = score / 5
    return score


# Word count function
def word_count(text, ideal_word_count=500, tolerance=100):
    word_count = len(text.split())
    deviation = abs(word_count - ideal_word_count)
    if deviation <= tolerance:
        return 1,word_count
    elif deviation <= tolerance * 2:
        return 0.8,word_count
    elif deviation <= tolerance * 3:
        return 0.6,word_count
    elif deviation <= tolerance * 4:
        return 0.4,word_count
    else:
        return 0.2,word_count


# Vocabulary score function
def vocabulary_score(text):
    words = text.split()
    unique_words = set(words)
    vocab_richness = len(unique_words) / len(words) if words else 0
    score = min(1, vocab_richness * 4)
    return score


# Text correction and embedding function
def embed(text):
    in1 = at(text, return_tensors="pt", truncation=True, padding=True)
    outputs = am(**in1)
    sem = torch.mean(outputs.last_hidden_state, dim=1)
    return sem.squeeze().detach().numpy()


# Text cleaning functions
def clean(text):
    clean = text.replace('', "'")
    clean = re.sub(r'[^\x00-\x7F]+', '', clean)
    clean = re.sub(r'\.\.+', '.', clean)
    clean = re.sub(r'\!+', '!', clean)
    clean = re.sub(r'\?+', '?', clean)
    return clean


def correction(text):
    blob = TextBlob(text)
    ct = blob.correct()
    return str(ct)


def correct(text):
    def correct2(word):
        return re.sub(r'(.)\1+', r'\1\1', word)

    words = text.split()
    cw = [correct2(word) for word in words]
    return ' '.join(cw)


# Evaluate total score
def evaluate_score(text):
    grammar,_=grammar_score(text)
    structure=structure_score(text)
    flow=flow_score(text)
    count,_=word_count(text)
    vocabulary=vocabulary_score(text)
    total=grammar+structure+flow+count+vocabulary
    return total


# Streamlit UI
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/clean-gray-paper.png');
        background-size: cover;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    .subtitle {
        font-size: 18px;
        font-style: italic;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<div class="title">🌟 ESSAY SCORER </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Evaluate your essay with advanced AI techniques</div>', unsafe_allow_html=True)

# File Upload or Text Area
essay_input = st.text_area(
    "Enter your essay below:",
    placeholder="Type or paste your essay here...",
    height=300,
)

# Button to evaluate
if st.button("Evaluate 📝"):
    with st.spinner("Analyzing your essay... Please wait!"):
        sc1 = evaluate_score(essay_input)
        essay = clean(essay_input)
        essay = correction(essay)
        essay = correct(essay)
        _,ec=grammar_score(essay)
        _,wc=word_count(essay)
        cc=ec/wc
        if cc>=0.5:
            score=max(1,math.floor(1/cc))
        if wc<=100:
            score=math.ceil((3*wc)/100)    
        else:
            embeddings=embed(essay)
            embeddings=sc.transform(embeddings.reshape(1,-1))
            sc2=xgb_regressor.predict(embeddings)[0]
            if sc2<0:
                sc2=0
            if sc2>5:
                sc2=5
            score=sc1+sc2
            score=math.ceil(sc1+sc2)
    st.success("✅ Analysis Complete!")
    st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>Your Essay Score: {score}/10</h2>", unsafe_allow_html=True)

    # Provide additional feedback
    if score > 7:
        st.balloons()
        st.markdown("<p style='text-align: center; color: #228B22;'>Great job! Your essay is well-structured and impactful.</p>", unsafe_allow_html=True)
    elif score > 4:
        st.markdown("<p style='text-align: center; color: #FFA500;'>Good effort! Consider improving grammar or structure.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; color: #FF4500;'>Needs improvement. Focus on grammar and vocabulary.</p>", unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2024 Essay Scorer | Powered by AI</div>', unsafe_allow_html=True)
