import streamlit as st
import pandas as pd
import numpy as np
import neattext as nt
import neattext.functions as nfx
from wordcloud import WordCloud
import plotly.express as px 
import matplotlib.pyplot as plt

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

# =============Function=============
def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

@st.cache(allow_output_mutation=True)
def load_data():
    data =  pd.read_excel('finaldataset.xlsx')
    return data

@st.cache(allow_output_mutation=True)
def load_corpus():
    data = pd.read_pickle("emotion_corpusV3.pkl")
    return data

@st.cache(allow_output_mutation=True)
def load_month_trend():
    data = pd.read_pickle("month_trend.pkl")
    return data
@st.cache(allow_output_mutation=True)
def load_cleancorpus():
    data = pd.read_pickle("dtm.pkl")
    return data

@st.cache(persist=True,suppress_st_warning=True)
def get_top_text_ngrams(corpus, ngrams=(1,1), nr=None):
    vec = CountVectorizer( ngram_range=ngrams).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:nr]


# Disable Some Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

def app():

    def title(text,size):
        st.markdown(f'<h3 style="font-weight:bolder;font-size:{size}px;text-align:center;">{text}</h3>',unsafe_allow_html=True)

    def header(text):
        st.markdown(f"<p style='color:white;'>{text}</p>",unsafe_allow_html=True)

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
            transform: scale(1.05);
        }
        </style>""",unsafe_allow_html=True)


    # loading the data
    df = load_data()
    corpus = load_corpus()
    month_trend = load_month_trend()
    clean_corpus = load_cleancorpus()
    st.title("EDA Analysis of UM Facebook Confession")
    space(1)
    st.markdown("""
    * Dataset Size: 9357 posts data
    * Timeline: JUNE 2019 - NOVEMBER 2022
    """)
    space(1)
    st.markdown("**IMPORTANT**: It might take some time for the results to load due to the large dataset that is needed to process. ")
    space(1)

    st.write("***")

    # -------------------- Emotion selection ------------------------
    space(1)
    st.subheader("Dataset")    
    with st.expander("Click to See Datasets 👇"):
        emotion_list = ['All', 'joy', 'sadness', 'fear', 'anger', 'neutral']
        select_emotion = st.selectbox('select emotion',emotion_list)
        # Filtering data
        if select_emotion == 'All':
            df_selected_tweet = df
        else:
            df_selected_tweet = df[(df.Emotion.isin([select_emotion]))]

        st.header('Display Tweets of Selected Emotion(s)')
        st.write('Data Dimension: '+str(df_selected_tweet.shape[0]) + ' rows and '+ str(df_selected_tweet.shape[1])+ ' columns.')
        st.dataframe(df_selected_tweet)    
    
    space(1)
    #st.write("***")
    # -------------------- Visualisation ------------------------
    st.subheader("Data Visualisation")
    with st.container():

        col_1, col_2, col_3, col_4 = st.columns([2,0.5,7,1])
        with col_1:
            space(3)
            choiceSelection = st.radio("Choose a visualization", ("Emotion Distribution","Emotion Word Cloud","Trendy Words","Trendy Words Based on Timeline")) 
        with col_3:
            space(2)
            if choiceSelection=="Emotion Distribution":
                title('Distribution of Emotions',30)
                # ---------------------- Emotion Bar Chart ---------------------
                emotion_count = df['Emotion'].value_counts().rename_axis('Emotions').reset_index(name='Counts')
                bar_CC = px.bar(emotion_count, x='Emotions', y='Counts', color='Emotions', color_discrete_sequence=px.colors.sequential.Plotly3)
                # bar_CC.update_xaxes(tickangle=0)
                bar_CC.update_layout(height=450) #margin_t=10,margin_b=150,
                st.plotly_chart(bar_CC,use_container_width=True)


            elif choiceSelection=="Emotion Word Cloud":
                #--------------------------WORD_CLOUD---------------------------
                title('Emotions WordCloud',30)

                unique_emotion = ['sadness', 'neutral', 'anger', 'fear', 'joy']
                sl = st.slider('Pick Number of Words',50,1000)
                
                def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
                    return("hsl(240,100%%, %d%%)" % np.random.randint(45,55))
                
                wc = WordCloud(background_color="white", color_func = grey_color_func, max_font_size=150, random_state=42,max_words=sl, collocations=False)

                plt.rcParams['figure.figsize'] = [30, 30]  #16,6 #40,40
                full_names = unique_emotion

                # Create subplots for each emotion
                for index, emotion in enumerate(corpus.emotion):
                    wc.generate(corpus.clean_message[emotion])
                    
                    plt.subplot(4, 2, index+1)  #3,4 #4,2
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    plt.title(full_names[index], fontsize = 40)
                    
                st.pyplot()
            elif choiceSelection=="Trendy Words":
                #-------------------------Module 1-----------------------------

                title('Most Popular One Word',30)
                # st.caption('removing all the stop words in the sense common words.')

                sl_2 = st.slider('Pick Number of Words',5,50,10, key="1")

                # Unigrams - Most Popular One Keyword
                top_text_bigrams = get_top_text_ngrams(corpus.clean_message, ngrams=(1,1), nr=sl_2)
                top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
                x, y = zip(*top_text_bigrams)
                bar_C1 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title='Most Popular One Word', text=y, color_continuous_scale=px.colors.sequential.Plotly3[::-1])
                bar_C1.update_traces(textposition="outside", cliponaxis=False)
                bar_C1.update_yaxes(dtick=1, automargin=True)

                st.plotly_chart(bar_C1,use_container_width=True)

                #-------------------------Module 2-----------------------------
                title('Most Popular Two Words',30)

                sl_3 = st.slider('Pick Number of Words',5,50,10, key="2")

                # Unigrams - Most Popular One Keyword
                top_text_bigrams = get_top_text_ngrams(corpus.clean_message, ngrams=(2,2), nr=sl_3)
                top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
                x, y = zip(*top_text_bigrams)
                bar_C2 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title='Most Popular Two Word', text=y, color_continuous_scale='Plotly3_r')
                bar_C2.update_traces(textposition="outside", cliponaxis=False)
                bar_C2.update_yaxes(dtick=1, automargin=True)

                st.plotly_chart(bar_C2,use_container_width=True)

                #-------------------------Module 3-----------------------------
                title('Most Popular Three Words',30)

                # header("range")
                sl_4 = st.slider('Pick Number of Words',5,50,10, key="3")

                # Unigrams - Most Popular One Keyword
                top_text_bigrams = get_top_text_ngrams(corpus.clean_message, ngrams=(3,3), nr=sl_4)
                top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
                x, y = zip(*top_text_bigrams)
                bar_C3 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title='Most Popular Three Word', text=y,color_continuous_scale='Plotly3_r')
                bar_C3.update_traces(textposition="outside", cliponaxis=False)
                bar_C3.update_yaxes(dtick=1, automargin=True)

                st.plotly_chart(bar_C3,use_container_width=True)
                
            else:
                month_trend_1=month_trend[0:50]
                # st.write(month_trend[0:50])
                # "Trendy Words Based on Timeline"  
                #----------------------Line Chart Keywords--------------------------
                title('Trendy Words Across Timeline',30)
                line_chart = px.line(month_trend_1, x='Month', y='Counts', color='Words',markers=True,color_discrete_sequence=px.colors.cyclical.HSV)
                line_chart.update_traces(mode="markers+lines", hovertemplate=None)
                line_chart.update_layout(height=430,hovermode="x unified") # plot_bgcolor='aliceblue'
                st.plotly_chart(line_chart,use_container_width=True)
                st.write("") 

        with col_4:
          st.write("")
          space(2)

        if choiceSelection=="Trendy Words Based on Timeline":
            
          with col_3:
              title('Trendy Words Based on Month',30)
            
          col__1, col__2, col__3, col__4 = st.columns([3,3,3,2])

          with col__2:
              monthChoice = st.radio("Select Month", ('June_2019','July_2019','August_2019','September_2019', 'October_2019', 'November 2019', 'December 2019',
               'January_2020','February_2020','March_2020','April_2020','May_2020','June_2020','July_2020','August_2020','September_2020', 'October_2020', 'November 2020', 'December 2020',
               'January_2021','February_2021','March_2021','April_2021','May_2021','June_2021','July_2021','August_2021','September_2021', 'October_2021', 'November 2021','Decmber_2021',
               'January_2022','February_2022','March_2022','April_2022','May_2022','June_2022','July_2022','August_2022','September_2022', 'October_2022', 'November 2022'))

          with col__3:
              sl_5 = st.slider("Pick Number of Words",5,50,10, key="4")
            #-------------------------Module 4-----------------------------
            
          col___1, col___2, col___3 = st.columns([2,7,1])

          with col___2:
              months_name = [
                'June_2019','July_2019','August_2019','September_2019', 'October_2019', 'November 2019', 'December 2019',
               'January_2020','February_2020','March_2020','April_2020','May_2020','June_2020','July_2020','August_2020','September_2020', 'October_2020', 'November 2020', 'December 2020',
               'January_2021','February_2021','March_2021','April_2021','May_2021','June_2021','July_2021','August_2021','September_2021', 'October_2021', 'November 2021','Decmber_2021',
               'January_2022','February_2022','March_2022','April_2022','May_2022','June_2022','July_2022','August_2022','September_2022', 'October_2022', 'November 2022'
               ]
              months_list = [
                  '2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12',
                  '2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
                  '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
                  '2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11'
                  ]
              months = {months_name[i]: months_list[i] for i in range(len(months_name))}
              df2 = df.copy()
              df2['clean_message'] = df2['clean_message'].apply(nfx.remove_stopwords)
              df2['datetime'] = pd.to_datetime(df2['datetime'])
              df2_date = df2.set_index('datetime')

              # title(f"Top {sl_5} Keywords For {monthChoice}",40,'black')
              # Unigrams - Most Popular One Keyword
              selected_month = months[monthChoice]
              top_text_bigrams = get_top_text_ngrams(df2_date.loc[selected_month].clean_message, ngrams=(1,1), nr=sl_5)
              top_text_bigrams = sorted(top_text_bigrams, key=lambda x:x[1], reverse=False)
              x, y = zip(*top_text_bigrams)
              bar_C4 = px.bar(x=y,y=x, color=y, labels={'x':'Number of words','y':'Words','color':'frequency'}, title=f'Top {sl_5} Keywords In {monthChoice}', text=y, color_continuous_scale='Plotly3_r')
              bar_C4.update_traces(textposition="outside", cliponaxis=False)
              # bar_C4.update_layout(title=f'Top KeywordWord In{monthChoice}')
              # bar_C4.update_layout(autosize=True)
              bar_C4.update_yaxes(dtick=1, automargin=True)

              st.plotly_chart(bar_C4,use_container_width=True)
   
