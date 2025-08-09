import streamlit as st
import pandas as pd
import pickle
import gzip
import os
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from io import BytesIO
import xlsxwriter
import html
import textwrap
from rapidfuzz import process, fuzz
import urllib.parse

# ---------------------------
# Helper: review persistence
# ---------------------------
def save_review(user_review):
    review_df = pd.DataFrame([user_review])
    review_df.to_csv('user_reviews.csv', mode='a', header=not os.path.exists('user_reviews.csv'), index=False)

def load_reviews():
    if os.path.exists('user_reviews.csv'):
        return pd.read_csv('user_reviews.csv')
    return pd.DataFrame(columns=['username', 'rating', 'review'])

def calculate_statistics(reviews):
    average_rating = reviews['rating'].mean() if not reviews.empty else 0
    total_reviews = len(reviews)
    total_visits = 0
    return total_visits, average_rating, total_reviews

# ---------------------------
# Load data (with helpful errors)
# ---------------------------
# Use cache so Streamlit doesn't reload heavy pickles every rerun unnecessarily
@st.cache_data
def load_manhwa_data():
    try:
        with gzip.open('manhwa_dict_with_cover.pkl.gz', 'rb') as f:
            manhwa_dict = pickle.load(f)
        manhwas_df = pd.DataFrame(manhwa_dict).fillna("")
        if 'title' not in manhwas_df.columns:
            raise ValueError("Kolom 'title' tidak ditemukan di dataset manhwa_dict_with_cover.pkl.gz")
        return manhwas_df
    except Exception as e:
        raise

@st.cache_data
def load_similarity():
    try:
        with gzip.open('similarity.pkl.gz', 'rb') as f:
            sim = pickle.load(f)
        return sim
    except Exception as e:
        raise

@st.cache_data
def load_tag_resources():
    try:
        with gzip.open('tag_vectorizer.pkl.gz', 'rb') as f:
            tv = pickle.load(f)
        with gzip.open('tag_vectors.pkl.gz', 'rb') as f:
            tvectors = pickle.load(f)
        return tv, tvectors
    except Exception:
        return None, None

# Attempt load and show friendly errors if missing
try:
    manhwas = load_manhwa_data()
    titles_list_all = manhwas['title'].astype(str).tolist()
except Exception as e:
    st.error(f"Error memuat dataset manhwa: {e}")
    st.stop()

try:
    similarity = load_similarity()
except Exception as e:
    st.error(f"Error memuat similarity matrix: {e}")
    st.stop()

tag_vectorizer, tag_vectors = load_tag_resources()
# tag_vectorizer/tag_vectors may be None — keyword mode will show error accordingly

# ---------------------------
# Utilities
# ---------------------------
translator = Translator()
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='id', dest='en')
        return translated.text
    except Exception:
        return text

# fuzzy helper using rapidfuzz
FUZZY_THRESHOLD = 60  # adjust if you want stricter/looser matching

def fuzzy_top_matches(query, choices, limit=5):
    """
    Return list of (title, score) tuples sorted desc, but only those >= FUZZY_THRESHOLD.
    """
    results = process.extract(query, choices, limit=limit, scorer=fuzz.token_sort_ratio)
    # results: list of tuples (match, score, index)
    filtered = [(match, score) for (match, score, idx) in results if score >= FUZZY_THRESHOLD]
    return filtered

def search_titles_combined(query, max_options=10, fuzzy_limit=5):
    """
    Combined search:
    - if substring (case-insensitive) matches exist -> return substring matches (up to max_options)
    - else -> return fuzzy top matches (title strings) up to fuzzy_limit
    """
    if not query or not query.strip():
        # By default return top subset of all titles to avoid massive dropdown (first 200)
        return titles_list_all[:200]

    # substring matches (case-insensitive)
    substring_matches = manhwas[manhwas['title'].str.contains(query, case=False, na=False)]['title'].tolist()
    if substring_matches:
        # limit length for performance
        return substring_matches[:max_options]
    # else fallback fuzzy
    fuzzy = fuzzy_top_matches(query, titles_list_all, limit=fuzzy_limit)
    if fuzzy:
        # return just titles, perhaps annotate score? For dropdown keep plain title
        return [t for t, s in fuzzy]
    # nothing
    return []

# Recommendation functions
def recommend_by_title_include_self(selected_title, top_n):
    """
    Return top_n recommendations including the item itself at highest similarity (index 0).
    """
    try:
        idx = manhwas[manhwas['title'] == selected_title].index[0]
    except IndexError:
        return []
    distances = similarity[idx]
    # sort descending; include index 0 (self) as first
    ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    top = [i for (i,score) in ranked][:top_n]  # take top_n indices including self (if self highest)
    results = []
    for i in top:
        row = manhwas.iloc[i]
        results.append({
            'title': row['title'],
            'cover_url': row.get('cover_url', None),
            'synopsis': row.get('synopsis', ''),
            'genres': row.get('genres', ''),
            'authors': row.get('authors', ''),
            'score': row.get('score', None)
        })
    return results

def recommend_by_keyword(user_input, top_n):
    if tag_vectorizer is None or tag_vectors is None:
        return []
    try:
        user_vec = tag_vectorizer.transform([user_input])
        scores = cosine_similarity(user_vec, tag_vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
    except Exception:
        return []
    results = []
    for i in top_indices:
        row = manhwas.iloc[i]
        results.append({
            'title': row['title'],
            'cover_url': row.get('cover_url', None),
            'synopsis': row.get('synopsis', ''),
            'genres': row.get('genres', ''),
            'authors': row.get('authors', ''),
            'score': row.get('score', None)
        })
    return results

# ---------------------------
# Streamlit UI initializations
# ---------------------------
st.set_page_config(page_title="Manhwa Recommender", layout="wide")
st.title('Manhwa Recommender System')

# Sidebar
page = st.sidebar.selectbox("Select Page:", ["Recommendation", "Review"])

# Session state defaults
if 'results' not in st.session_state:
    st.session_state.results = []
if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'top_n' not in st.session_state:
    st.session_state.top_n = 5
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = None

# ---------------------------
# Recommendation Page
# ---------------------------
if page == "Recommendation":
    st.subheader("Recommendation Page")

    mode = st.radio("Select Recommendation Mode:", ["By Title", "By Keyword"])

    # ---------- By Title ----------
    if mode == "By Title":
        st.markdown("Ketik sebagian judul di bawah — dropdown akan menampilkan *combined* matching: substring/full match (jika ada), atau fallback fuzzy-match (top 5) jika tidak ada substring match.")

        # input textbox for typing query; dropdown will be built from it
        title_query = st.text_input("Type the Manhwa Title:", value=st.session_state.get('query', ''))
        st.session_state.query = title_query

        # get dropdown options using combined logic
        dropdown_options = search_titles_combined(title_query, max_options=50, fuzzy_limit=5)

        if not dropdown_options:
            st.info("Tidak ada hasil pencarian untuk input ini.")
            selected_title = None
            # show disabled selectbox-like text
            st.selectbox("Choose a Manhwa Title:", options=["(No results)"])
        else:
            selected_title = st.selectbox("Choose a Manhwa Title:", options=dropdown_options, index=0)
            st.session_state.selected_title = selected_title

        # When user clicks Find Recommendations => immediately show default 5 recommendations (include self)
        if st.button("Find Recommendations", key='find_title'):
            if not selected_title:
                st.warning("Silakan pilih judul terlebih dahulu dari dropdown.")
            else:
                st.session_state.top_n = 5  # default
                st.session_state.results = recommend_by_title_include_self(selected_title, st.session_state.top_n)
                st.session_state.last_mode = "title"

        # If results present and last_mode title, show them and show buttons to change top_n
        if st.session_state.results and st.session_state.last_mode == "title":
            st.subheader("Recommendations:")
            for idx, item in enumerate(st.session_state.results):
                col1, col2 = st.columns([1, 3])
                with col1:
                    if item['cover_url']:
                        try:
                            st.image(item['cover_url'], width=120)
                        except Exception:
                            st.text("(cover load failed)")
                    else:
                        st.text("(no cover)")
                with col2:
                    st.markdown(f"### {item['title']}")
                    st.markdown(f"**Author:** {item.get('authors','')}")
                    st.markdown(f"**Genre:** {item.get('genres','')}")
                    if item.get('score') is not None:
                        st.markdown(f"**Score:** {item.get('score')}")
                    checkbox_key = f"show_synopsis_title_{idx}"
                    show_full = st.checkbox("📖 Show full synopsis", key=checkbox_key)
                    if show_full:
                        st.markdown(html.escape(item.get('synopsis','')))
                        webtoon_search_url = "https://www.webtoons.com/id/search?keyword=" + urllib.parse.quote(item['title'])
                        st.markdown(f'<a href="{webtoon_search_url}" target="_blank">🔍 Baca/ Cari di Webtoon</a>', unsafe_allow_html=True)
                    else:
                        short = textwrap.shorten(item.get('synopsis',''), width=200, placeholder="...")
                        st.markdown(html.escape(short))
                st.markdown("---")

            # Buttons to change number of recommendations on the fly
            st.write("Ubah jumlah rekomendasi:")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("5"):
                st.session_state.top_n = 5
                # recalc based on last selected title
                if st.session_state.selected_title:
                    st.session_state.results = recommend_by_title_include_self(st.session_state.selected_title, st.session_state.top_n)
            if c2.button("10"):
                st.session_state.top_n = 10
                if st.session_state.selected_title:
                    st.session_state.results = recommend_by_title_include_self(st.session_state.selected_title, st.session_state.top_n)
            if c3.button("15"):
                st.session_state.top_n = 15
                if st.session_state.selected_title:
                    st.session_state.results = recommend_by_title_include_self(st.session_state.selected_title, st.session_state.top_n)
            if c4.button("20"):
                st.session_state.top_n = 20
                if st.session_state.selected_title:
                    st.session_state.results = recommend_by_title_include_self(st.session_state.selected_title, st.session_state.top_n)

    # ---------- By Keyword ----------
    elif mode == "By Keyword":
        st.markdown("Masukkan kata kunci (genre, deskripsi, gaya cerita) untuk rekomendasi berdasarkan tag/content.")
        user_input = st.text_input("Enter free keywords (genre, story style, etc.):", value=st.session_state.get('query', ''))

        # store keyword in session state too
        st.session_state.query = user_input

        if st.button("Find Recommendations", key='find_keyword'):
            if not user_input or not user_input.strip():
                st.warning("Silakan masukkan kata kunci.")
            else:
                # default 5
                st.session_state.top_n = 5
                recs = recommend_by_keyword(user_input.strip(), st.session_state.top_n)
                if recs:
                    st.session_state.results = recs
                    st.session_state.last_mode = "keyword"
                else:
                    # if tag resources missing, give feedback
                    if tag_vectorizer is None or tag_vectors is None:
                        st.error("Keyword recommendation tidak tersedia (tag_vectorizer/tag_vectors tidak ditemukan).")
                    else:
                        st.info("Tidak ditemukan rekomendasi berdasarkan kata kunci tersebut.")

        # when results exist for keyword
        if st.session_state.results and st.session_state.last_mode == "keyword":
            st.subheader("Recommendations:")
            for idx, item in enumerate(st.session_state.results):
                col1, col2 = st.columns([1, 3])
                with col1:
                    if item['cover_url']:
                        try:
                            st.image(item['cover_url'], width=120)
                        except Exception:
                            st.text("(cover load failed)")
                    else:
                        st.text("(no cover)")
                with col2:
                    st.markdown(f"### {item['title']}")
                    st.markdown(f"**Author:** {item.get('authors','')}")
                    st.markdown(f"**Genre:** {item.get('genres','')}")
                    if item.get('score') is not None:
                        st.markdown(f"**Score:** {item.get('score')}")
                    checkbox_key = f"show_synopsis_kw_{idx}"
                    show_full = st.checkbox("📖 Show full synopsis", key=checkbox_key)
                    if show_full:
                        st.markdown(html.escape(item.get('synopsis','')))
                        webtoon_search_url = "https://www.webtoons.com/id/search?keyword=" + urllib.parse.quote(item['title'])
                        st.markdown(f'<a href="{webtoon_search_url}" target="_blank">🔍 Baca/ Cari di Webtoon</a>', unsafe_allow_html=True)
                    else:
                        short = textwrap.shorten(item.get('synopsis',''), width=200, placeholder="...")
                        st.markdown(html.escape(short))
                st.markdown("---")

            # Buttons to change number of recommendations on the fly (re-run keyword recalc)
            st.write("Ubah jumlah rekomendasi:")
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("5", key='k5'):
                st.session_state.top_n = 5
                st.session_state.results = recommend_by_keyword(st.session_state.query, st.session_state.top_n)
            if c2.button("10", key='k10'):
                st.session_state.top_n = 10
                st.session_state.results = recommend_by_keyword(st.session_state.query, st.session_state.top_n)
            if c3.button("15", key='k15'):
                st.session_state.top_n = 15
                st.session_state.results = recommend_by_keyword(st.session_state.query, st.session_state.top_n)
            if c4.button("20", key='k20'):
                st.session_state.top_n = 20
                st.session_state.results = recommend_by_keyword(st.session_state.query, st.session_state.top_n)

    # ---------- Common: Review form under recommendations ----------
    if (st.session_state.results and st.session_state.last_mode in ["title", "keyword"]):
        with st.expander("📬 Submit a Review for a Manhwa"):
            username = st.text_input("User Name:", key="rec_username")
            rating = st.slider("Rating (1-5):", 1, 5, key="rec_rating")
            review_text = st.text_area("Write a Review:", key="rec_review")
            if st.button("Submit Review", key="submit_rec"):
                user_review = {
                    'username': username,
                    'rating': rating,
                    'review': review_text
                }
                save_review(user_review)
                st.success("Review submitted successfully!")

# ---------------------------
# Review Page (unchanged core behavior)
# ---------------------------
elif page == "Review":
    st.subheader("User Review")

    username = st.text_input("User Name:")
    rating = st.slider("Rating (1-5):", 1, 5)
    review_text = st.text_area("Write a Review:")

    if st.button("Submit a Review"):
        user_review = {
            'username': username,
            'rating': rating,
            'review': review_text
        }
        save_review(user_review)
        st.success("Review submitted successfully!")

    reviews = load_reviews()
    total_visits, average_rating, total_reviews = calculate_statistics(reviews)
    st.write(f"Current Rating: {average_rating:.2f} from {total_reviews} reviews")

    st.subheader("Rating Distribution:")
    if not reviews.empty:
        rating_counts = reviews['rating'].value_counts().sort_index()
        for rating_val in range(1, 6):
            count = rating_counts.get(rating_val, 0)
            st.write(f"Rating {rating_val}: {count} User")
    else:
        for rating_val in range(1, 6):
            st.write(f"Rating {rating_val}: 0 User")

    st.subheader("Latest Review:")
    if not reviews.empty:
        latest_reviews = reviews.tail(5)
        for index, row in latest_reviews.iterrows():
            st.write(f"**{row['username']}** - Rating: {row['rating']}")
            st.write(row['review'])
            st.write("---")
    else:
        st.write("No reviews yet.")

    st.subheader("Download Review Data:")
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Ulasan')
        return output.getvalue()

    excel_data = convert_df_to_excel(reviews)
    st.download_button(
        label="📥 Download as Excel",
        data=excel_data,
        file_name='ulasan_pengguna.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.subheader("All Reviews:")
    st.dataframe(reviews)
