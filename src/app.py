import streamlit as st
import requests
import html
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from PIL import Image
from pathlib import Path
import RAG  # your RAG module

# --- Page Configuration ---
st.set_page_config(
    page_title="BioSpace AI Astronaut",
    page_icon="🦠",
    layout="wide"
)

# --- Load CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Use RAG theme CSS
load_css("src/style.css")

# --- API Key Configuration ---
# try:
#     NASA_API_KEY = st.secrets["NASA_API_KEY"]
# except FileNotFoundError:
#     st.error("Secrets file not found. Please create a .streamlit/secrets.toml file.")
#     st.stop()
# except KeyError as e:
#     st.error(f"API key '{e.args[0]}' not found in secrets. Please add it to your .streamlit/secrets.toml file.")
#     st.stop()

# --- NASA APOD Helper ---
def get_apod_data(api_key, date=None):
    url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}"
    if date:
        url += f"&date={date.strftime('%Y-%m-%d')}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


# ===============================
# === PAGE 1: HOME PAGE
# ===============================
def home_page():
    st.title("Welcome to the NASA Explorer!")
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h4>🤖 Ask a Question</h4>
            <p>Have a burning question about space? Our RAG Astronaut will retrieve facts from real NASA data and explain them.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ask the RAG Astronaut", use_container_width=True):
            st.session_state.page = "RAG Astronaut"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h4>⏪ A Decade Ago Today</h4>
            <p>Travel back in time! View the Astronomy Picture of the Day from exactly 10 years ago.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Look Back 10 Years", use_container_width=True):
            st.session_state.page = "Picture of the Day"
            st.session_state.target_date = datetime.today() - relativedelta(years=10)
            st.rerun()


# ===============================
# === PAGE 2: PICTURE OF THE DAY
# ===============================
def picture_of_the_day_page():
    st.title("🔭 Astronomy Picture of the Day")

    default_date = st.session_state.get('target_date', datetime.today() - timedelta(days=2))
    if 'target_date' in st.session_state:
        del st.session_state['target_date']

    min_date = datetime(2000, 1, 1)
    max_date = datetime.today() - timedelta(days=2)

    selected_date = st.date_input("Select a date to view a picture",
                                  value=default_date, min_value=min_date, max_value=max_date)
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
            copyright_text = html.escape(data.get('copyright', 'Public Domain'))
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


# ===============================
# === PAGE 3: RAG ASTRONAUT PAGE
# ===============================
@st.cache_resource
def load_rag():
    return RAG.NasaRAG(documents_csv='Nasa_Data_scraped_data_eda.csv', directory='data')

def rag_astronaut_page():
    rag = load_rag()

    # --- Session State ---
    for key in ['last_query', 'last_result', 'last_images']:
        if key not in st.session_state:
            st.session_state[key] = None

    # --- Sidebar ---
    st.sidebar.title("⚙️ Controls",)
    st.sidebar.markdown("### 🔧 Parameters")

    top_k = st.sidebar.slider('Top-K Results', 1, 20, 20)
    threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5, 0.05)
    # max_new_tokens = st.sidebar.selectbox('Max New Tokens', [128, 256, 512, 1024])
    max_new_tokens = 1024
    alpha = st.sidebar.slider('Alpha (BM25 ↔ FAISS)', 0.0, 1.0, 0.5, 0.05)
    use_db = st.sidebar.checkbox('Enable Database Search',value=True)

    st.sidebar.markdown("### 🚀 Actions")
    action = st.sidebar.selectbox('Action', ['Ask Question', 'Delete Question', 'Reset Database', 'View Database'])
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 *Tune parameters on the left, ask your question in the main panel!*")

    # --- Title ---
    st.markdown("""
    <div class="title-container">
        <h1>BioSpace AI Astronaut</h1>
        <p>Explore the mysteries of life beyond Earth through intelligent retrieval and generation.</p>
        <p>Using Gemini-2.5-Flash.</p>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    query_text = st.text_input('🧠 Ask a Question:', key='q', placeholder='e.g. How do cells adapt to microgravity?')

    # --- Helper Function ---
    def get_answer(query):
        answer, images = rag.search_question(query, top_k=top_k,
                                             max_new_tokens=max_new_tokens,
                                             alpha=alpha, use_db=use_db)
        return answer, images

    # --- Main Button ---
    if st.button(f"{action}"):
        st.session_state.last_query = None
        st.session_state.last_result = None
        st.session_state.last_images = None

        match action:
            case 'Ask Question':
                with st.spinner('Generating answer...'):
                    result, images = get_answer(query_text)
                if result:
                    st.session_state.last_query = query_text
                    st.session_state.last_result = result
                    st.session_state.last_images = images
                else:
                    st.error('⚠️ Try increasing max_new_tokens or adjusting alpha.')

            case 'Delete Question':
                if rag.db.delete(embedded_question=rag.embed_query(query_text),
                                 deletion_type=rag.db.delete_one,
                                 collection='questions'):
                    st.success(f'🗑️ Deleted: {query_text}')
                else:
                    st.error(f'❌ Could not find: {query_text}')

            case 'Reset Database':
                if rag.db.delete(embedded_question=rag.embed_query(query_text),
                                 deletion_type=rag.db.delete_all,
                                 collection='questions'):
                    st.success('Database Reset Successfully!')

            case 'View Database':
                data = rag.db.get_all_questions()
                if data:
                    st.success('📚 Stored Questions:')
                    for i, d in enumerate(data):
                        st.markdown(f"**{i+1}.** {d['question']}")
                else:
                    st.warning('Database is empty.')

    # --- Display Results ---
    if st.session_state.last_result:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.success("✅ Answer Generated")

        st.markdown(f"""
        <div class="answer-box">{st.session_state.last_result}</div>
        """, unsafe_allow_html=True)

        images = st.session_state.last_images
        if images:
            st.markdown("### 🖼️ Related Images")
            for image in images:
                try:
                    st.image(Image.open(image), use_container_width=True)
                except:
                    st.warning(f"Could not display image: {image}")

        if st.button('💾 Save Answer in Database'):
            rag.db.delete(rag.embed_query(query_text), deletion_type=rag.db.delete_one)
            rag.db.insert_question(st.session_state['last_query'],
                                   st.session_state['last_result'],
                                   rag.embed_query(st.session_state['last_query']),
                                   st.session_state['last_images'])
            st.success('Answer Saved!')
            st.session_state.last_query = None
            st.session_state.last_result = None


# ===============================
# === ROUTER & NAVIGATION BAR
# ===============================
if 'page' not in st.session_state:
    st.session_state.page = "Home"

page_options = {
    "Home": home_page,
    "Picture of the Day": picture_of_the_day_page,
    "RAG Astronaut": rag_astronaut_page
}

# Top Navigation
logo_col, nav_col = st.columns([1, 4])
with logo_col:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=80)

with nav_col:
    page_names = list(page_options.keys())
    current_page_index = page_names.index(st.session_state.page)
    selected_page = st.radio(
        "Navigation", options=page_names,
        index=current_page_index,
        key="sidebar_selection", horizontal=True
    )
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

# Execute the selected page
page_options[st.session_state.page]()
