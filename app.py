import streamlit as st
import hydralit_components as hc
import utils as udisp
img = 'sen pic.jpg'
import home , dataVisualization
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


st.set_page_config(
    page_title="Emolyzer",
    page_icon= img,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "### Emotion Analyzer"
    }
)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://miro.medium.com/max/1400/1*Dahjzk4_GsaFH-kRXHfaiw.png");
background-size: 1000px 270px;
background-position: center top ;
background-repeat: no-repeat;
background-attachment: local;
}}

</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)
MENU = {
    "Home" : home,
    "Exploratory Data Analysis" : dataVisualization,
    
}
def main():
    
    # specify the primary menu definition
    menu_data = [
        {'icon': "far fa-chart-bar", 'label':"Exploratory Data Analysis"},#no tooltip message
        {'icon': "fas fa-desktop",'label':"Monitor"},
        {'icon': "far fa-copy", 'label':"Documentation"},
        {'icon': "fas fa-info-circle", 'label':"About"}, 
    ]

    #create_emotionclf_table()

    over_theme = {'txc_inactive': '#FFFFFF','menu_background':'#660066'}
    menu_id = hc.nav_bar(
        menu_definition=menu_data,
        override_theme=over_theme,
        home_name='Home',
        hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
        sticky_nav=True, #at the top or not
        sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
    )

    
    # st.sidebar.title("Navigate yourself...")
    # menu_selection = st.sidebar.radio("Menu", list(MENU.keys()))

    menu = MENU[menu_id]
    menu_selection = menu_id
    with st.spinner(f"Loading {menu_id} ..."):
        udisp.render_page(menu)
if __name__ == '__main__':
      main()
