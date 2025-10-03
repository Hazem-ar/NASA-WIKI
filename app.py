
import streamlit as st
import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import html
from dateutil.relativedelta import relativedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="NASA Explorer",
    page_icon="🚀", 
    layout="wide"
)

# --- UPDATE: Removed the YOUR_LOGO_URL variable ---

# --- Function to load custom CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# --- API Key Configuration ---
# (This section is unchanged)
try:
    NASA_API_KEY = st.secrets["NASA_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except FileNotFoundError:
    st.error("Secrets file not found. Please create a .streamlit/secrets.toml file.")
    st.stop()
except KeyError as e:
    st.error(f"API key '{e.args[0]}' not found in secrets. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

# --- Helper Functions ---
# (This section is unchanged)
def get_apod_data(api_key, date=None):
    url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}"
    if date:
        url += f"&date={date.strftime('%Y-%m-%d')}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None

# --- Page Definitions ---

def home_page():
    st.title("Welcome to the NASA Explorer! 🌌")
    # --- UPDATE: Removed the custom logo from the homepage ---
    st.markdown("---")

    # (Rest of the homepage is unchanged)
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h4>🎂 Your Birthday Discovery</h4>
            <p>See the cosmos on the day you were born. Select your birth date to view the Astronomy Picture of the Day.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Your Birthday", use_container_width=True):
            st.session_state.page = "Picture of the Day"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h4>⏪ A Decade Ago Today</h4>
            <p>Travel back in time! This button will show you the Astronomy Picture of the Day from exactly 10 years ago.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Look Back 10 Years", use_container_width=True):
            st.session_state.page = "Picture of the Day"
            st.session_state.target_date = datetime.today() - relativedelta(years=10)
            st.rerun()

    with col3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h4>🤖 Ask a Question</h4>
            <p>Have a burning question about space? Chat with our AI Astronaut, powered by Google Gemini, and get answers.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ask the AI Astronaut", use_container_width=True):
            st.session_state.page = "Ask an AI Astronaut"
            st.rerun()

def picture_of_the_day_page():
    # (This function is unchanged)
    st.title("🔭 Astronomy Picture of the Day")
    
    default_date = st.session_state.get('target_date', datetime.today() - timedelta(days=2))
    
    if 'target_date' in st.session_state:
        del st.session_state['target_date']

    min_date = datetime(2000, 1, 1)
    max_date = datetime.today() - timedelta(days=2)

    selected_date = st.date_input(
        "Select a date to view a picture", 
        value=default_date,
        min_value=min_date, 
        max_value=max_date
    )
    
    data = get_apod_data(NASA_API_KEY, selected_date)
    
    if data:
        st.markdown(f'## {data.get("title", "No Title Available")}')
        col1, col2 = st.columns(2)
        with col1:
            media_type = data.get("media_type")
            if media_type == "image":
                st.image(data.get("hdurl", data.get("url")), caption="Click to enlarge")
            elif media_type == "video":
                st.video(data.get("url"))
            else:
                st.warning(f"Unsupported media type: {media_type}")
        with col2:
            copyright_text = data.get('copyright')
            if copyright_text:
                copyright_text = html.escape(copyright_text.strip())
            else:
                copyright_text = "Public Domain"
            st.markdown(f"""
            <div class="card">
                <strong>Date:</strong> {data.get('date', 'N/A')} <br>
                <strong>Copyright:</strong> {copyright_text}
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Read the explanation", expanded=True):
                 st.write(data.get("explanation", "No explanation available."))
    else:
        st.error("Could not retrieve a picture for the selected date. Please try another one.")

def ai_astronaut_page():
    st.title("🤖 Ask an AI Astronaut")
    st.markdown("""
    <div class="card">
    Have questions about space, planets, or the universe? Our AI Astronaut, powered by Google Gemini, is here to help!
    </div>
    """, unsafe_allow_html=True)

    # --- UPDATE: Use the current model name ---
    model = genai.GenerativeModel('gemini-1.5-flash')

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
        
    for message in st.session_state.chat_session.history:
        avatar_icon = "🧑‍🚀" if message.role == "user" else "🤖"
        with st.chat_message(name=message.role, avatar=avatar_icon):
            st.markdown(message.parts[0].text)

    if prompt := st.chat_input("What would you like to ask?"):
        with st.chat_message(name="user", avatar="🧑‍🚀"):
            st.markdown(prompt)
        try:
            response = st.session_state.chat_session.send_message(prompt)
            with st.chat_message(name="model", avatar="🤖"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"An error occurred with the AI model: {e}")

# --- Main App Router ---
# (This section is unchanged)
if 'page' not in st.session_state:
    st.session_state.page = "Home"

page_options = {
    "Home": home_page,
    "Picture of the Day": picture_of_the_day_page,
    "Ask an AI Astronaut": ai_astronaut_page
}

# --- Sidebar Navigation ---
# --- UPDATE: Reverted to the NASA logo ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=100)
st.sidebar.title("Navigation")

def set_page():
    st.session_state.page = st.session_state.sidebar_selection

st.sidebar.radio(
    "Go to", 
    options=list(page_options.keys()), 
    key="sidebar_selection", 
    on_change=set_page
)

# Call the function for the currently selected page
page_options[st.session_state.page]()