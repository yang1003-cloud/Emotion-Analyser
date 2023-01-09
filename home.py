import pandas as pd
import numpy as np
import re
from track_utils import add_prediction_details
from datetime import datetime
# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# preparing input to our model
import numpy as np
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import neattext as nt
from neattext.functions import clean_text
import joblib
from sklearn.feature_extraction import text 
import pandas as pd
import numpy as np

import ktrain
from ktrain import text

max_seq_len = 500

@st.cache(allow_output_mutation=True)
def load_model():
    model = ktrain.load_predictor('fypmodel')
    return model

model = load_model()

max_words = 5000

emotion_Names = ['joy', 'sadness', 'fear', 'anger', 'neutral']
emotions_emoji_dict = {"joy":"😄","sadness":"😔","fear":"😨","anger":"😡","neutral":"😎"}

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

@st.cache(allow_output_mutation=True)
def predict_emotions(docx):
      pred = model.predict(docx)
      return pred

@st.cache(allow_output_mutation=True)
def get_prediction_proba(docx):
    results = model.predict_proba(docx).tolist()
    print(results)
    return results
def cleantext(docx):
    docxFrame = nt.TextFrame(text=docx)
    docxFrame.remove_hashtags()
    docxFrame.remove_userhandles()
    docxFrame.remove_multiple_spaces()
    docxFrame.remove_urls()
    docxFrame.remove_emails()
    docxFrame.remove_numbers()
    docxFrame.remove_emojis()
    docxFrame.remove_puncts()
    docxFrame.remove_special_characters()
    docxFrame.remove_non_ascii()
    
    
    cleanDocx = docxFrame.text
    cleanDocx = decontracted(cleanDocx)
    return cleanDocx
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "cannot", phrase)
    phrase = re.sub(r"cant", " cannot", phrase)
    phrase = re.sub(r"shouldnt", "should not", phrase)
    phrase = re.sub(r"wouldnt", "would not", phrase)
    phrase = re.sub(r"willnt", " will not", phrase)
    phrase = re.sub(r"mightnt", " might not", phrase)
    phrase = re.sub(r"dont", " do not", phrase)
    phrase = re.sub(r"didnt", " did not", phrase)
    phrase = re.sub(r"doesnt", " does not", phrase)
    phrase = re.sub(r"wasnt", " was not", phrase)
    phrase = re.sub(r"isnt", " is not", phrase)
    phrase = re.sub(r"arent", " are not", phrase)
    phrase = re.sub(r"werent", " were not", phrase)

    # general
    
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
st.set_option('deprecation.showPyplotGlobalUse', False)
def app():
    st.markdown(f'<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)

    st.markdown("""
        <style>
        blockquote.twitter-tweet {
            display: inline-block;
            font-family: "Helvetica Neue", Roboto, "Segoe UI", Calibri, sans-serif;
            font-size: 12px;
            font-weight: bold;
            line-height: 16px;
            border-color: #eee #ddd #bbb;
            border-radius: 5px;
            border-style: solid;
            border-width: 1px;
            box-shadow: 0 1px 3px rgb(0 0 0 / 20%);
            margin: 10px 15%;
            padding: 8px 16px 16px 16px;
            max-width: 468px;
            transition: transform 500ms ease;
            
        }
        .twitter-tweet:hover,
        .twitter-tweet:focus-within {
            transform: scale(1.025);
        }
        </style>""",unsafe_allow_html=True)
    st.markdown('<h1 style="font-weight:20;font-size: 50px;font-family:Source Sans Pro, sans-serif;text-align:center;">Emolyser</h1>',unsafe_allow_html=True)
    st.markdown('<h1 style="font-weight:10;font-size: 35px;font-family:Source Sans Pro, sans-serif;text-align:center;">(Emotion Analysis from Text)</h1>',unsafe_allow_html=True)
    space(2)
    col_1, col_2, col_3 = st.columns([1,8,1])

    with col_1:
        st.write()
    
    with col_2:
        st.markdown("**Instructions:** Please type in some text for the emotion analysis")

        with st.form(key='emotion_form'):
            raw_text = st.text_area('Typing Area',"My boyfriend didn't turn up after promising that he was coming.")
            cleanDocx = cleantext(raw_text)
            submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        #st.balloons()  #display some balloons effect xD
        col1, col2, col3, col4 = st.columns([1,2,4,1])
        # col1,col2 = st.columns(2)

        # Apply Prediction Funtion Here
        prediction = predict_emotions(cleanDocx)
        probability = get_prediction_proba(cleanDocx)

        add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

        with col2:
            # st.success("Original Text")
            # st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction,emoji_icon))
            st.write("Score:{:.0%}".format(np.max(probability)))
        
        with col3:
            # st.success("Preprocessing Text")
            # st.write(cleanDocx)

            st.success("Emotion Score")
            #st.write(probability)
            proba_df = pd.DataFrame( np.array(probability).reshape(1,5),columns=model.get_classes())
            #st.write(proba_df.T)
            porba_df_clean = proba_df.T.reset_index()
            porba_df_clean.columns = ["emotions","probability"]

            # fig = alt.Chart(porba_df_clean,height=400).mark_bar().encode(x='emotions',y='probability', color='emotions')
            # st.altair_chart(fig, use_container_width=True)
            # ---------------------- Emotion Bar Chart ---------------------
            import plotly.express as px 
            bar_CC = px.bar(porba_df_clean, x='emotions', y='probability', color='emotions',color_discrete_sequence=px.colors.qualitative.T10)
            # https://plotly.com/python/discrete-color/

            bar_CC.update_xaxes() #tickangle=0
            bar_CC.update_layout() #margin_t=10,margin_b=150
            st.plotly_chart(bar_CC,use_container_width=True)
    else:
        with col_2: 
            st.write("*Analysis of text will appear here after you click the 'Analyze' button*")
