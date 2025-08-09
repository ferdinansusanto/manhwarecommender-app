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
try:
    with gzip.open('manhwa_dict_with_cover.pkl.gz', 'rb') as f:
        manhwa_dict = pickle.load(f)
    manhwas = pd.DataFrame(manhwa_dict)
    # ensure title column exists and clean
    if 'title' not in manhwas.columns:
        st.error("Kolom 'title' tidak ditemukan di dataset manhwa_dict_with_cover.pkl.gz")
        st.stop()
    titles_list_all = manhwas['title'].dropna().astype(str).tolist()
except Exception as e:
    st.error(f"Error memuat dataset manhwa: {e}")
    st.stop()

try:
    with gzip.open('similarity.pkl.gz', 'rb') as f:
        similarity = pickle.load(f)
except Exception as e:
    st.error(f"Error memuat similarity matrix: {e}")
    st.stop()

try:
    with gzip.open('tag_vectorizer.pkl.gz', 'rb') as f:
        tag_vectorizer = pickle.load(f)
    with gzip.open('tag_vectors.pkl.gz', 'rb') as f:
        tag_vectors = pickle.load(f)
except Exception as e:
    # Keyword mode akan error jika tidak ada tag_vectorizer/tag_vectors
    tag_vectorizer = None
    tag_vectors = None

# ---------------------------
# Utility functions
# ---------------------------
translator = Translator()
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='id', dest='en')
        return translated.text
    except Exception:
        return text

# fuzzy helper using rapidfuzz
FUZZY_THRESHOLD = 70  # ubah jika perlu (0-100)
def find_closest_titles(query, choices, limit=5):
    # returns list of matched title strings with score >= threshold
    results = process.extract(query, choices, limit=limit, scorer=fuzz.token_sort_ratio)
    return [match for match, score, _ in results if score >= FUZZY_THRESHOLD]

def recommend_by_title(selected_title, top_n):
    # assume selected_title exists in dataset
    try:
        idx = manhwas[manhwas['title'] == selected_title].index[0]
    except IndexError:
        st.error("Selected title not found in dataset.")
        return []
    distances = similarity[idx]
    manhwa_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1: top_n + 1]
    results = []
    for i in manhwa_list:
        row = manhwas.iloc[i[0]]
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
        st.error("Keyword recommendation tidak tersedia karena vectorizer/vectors tidak ditemukan.")
        return []
    try:
        user_vec = tag_vectorizer.transform([user_input])
        scores = cosine_similarity(user_vec, tag_vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
    except Exception as e:
        st.error(f"Error saat menghitung rekomendasi keyword: {e}")
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
# Streamlit UI
# ---------------------------
st.title('Manhwa Recommender System')

# Sidebar
page = st.sidebar.selectbox("Select Page:", ["Recommendation", "Review"])

# Initialize session state keys
if 'results' not in st.session_state:
    st.session_state.results = []
if 'awaiting_num_recs' not in st.session_state:
    st.session_state.awaiting_num_recs = False
if 'awaiting_num_recs_keyword' not in st.session_state:
    st.session_state.awaiting_num_recs_keyword = False
if 'selected_title_for_recommendation' not in st.session_state:
    st.session_state.selected_title_for_recommendation = None
if 'keyword_for_recommendation' not in st.session_state:
    st.session_state.keyword_for_recommendation = None

# --- Recommendation Page ---
if page == "Recommendation":
    st.subheader("Recommendation Page")
    mode = st.radio("Select Recommendation Mode:", ["By Title", "By Keyword"])

    # ---------- By Title ----------
    if mode == "By Title":
        st.markdown("Ketik sebagian judul di bawah — dropdown akan menampilkan judul yang berisi teks yang kamu ketik. Jika tidak ada substring match, sistem akan menampilkan hasil fuzzy-match (typo correction).")

        title_input = st.text_input("Type the Manhwa Title:")

        # prepare dropdown options
        dropdown_options = titles_list_all  # default all titles

        if title_input and title_input.strip():
            # First try substring (case-insensitive) matches to mimic previous behavior
            substring_matches = manhwas[manhwas['title'].str.contains(title_input, case=False, na=False)]['title'].tolist()
            if substring_matches:
                dropdown_options = substring_matches
            else:
                # fallback to fuzzy matching
                fuzzy_matches = find_closest_titles(title_input, titles_list_all, limit=10)
                if fuzzy_matches:
                    dropdown_options = fuzzy_matches
                    st.info("Tidak ada substring match — menampilkan suggestion terdekat (typo correction).")
                else:
                    dropdown_options = []
                    st.warning("Tidak ditemukan judul yang cocok atau mirip di dataset.")

        if dropdown_options:
            # keep stable ordering; convert to list
            # show selectbox with the options (user must pick one)
            selected_title = st.selectbox("Choose a Manhwa Title:", dropdown_options)
        else:
            selected_title = None

        # First button: "Find Recommendations" -> triggers showing number selection
        if st.button("Find Recommendations", key='find_by_title'):
            if not selected_title:
                st.warning("Silakan pilih judul terlebih dahulu dari dropdown.")
            else:
                st.session_state.awaiting_num_recs = True
                st.session_state.selected_title_for_recommendation = selected_title

        # If awaiting number of recommendations, show selectbox and final trigger
        if st.session_state.awaiting_num_recs:
            num_recommendations = st.selectbox("Jumlah rekomendasi:", [5, 10, 15, 20], index=0, key='num_recs_select_title')
            if st.button("Show Recommendations", key='show_recs_title'):
                st.session_state.results = recommend_by_title(st.session_state.selected_title_for_recommendation, num_recommendations)
                # reset awaiting flag so user can do a new search next time
                st.session_state.awaiting_num_recs = False

    # ---------- By Keyword ----------
    elif mode == "By Keyword":
        st.markdown("Masukkan kata kunci (genre, deskripsi, gaya cerita) untuk mencari rekomendasi berdasarkan tag/content.")
        user_input = st.text_input("Enter free keywords (genre, story style, etc.):")

        if st.button("Find Recommendations", key='find_by_keyword'):
            if not user_input or not user_input.strip():
                st.warning("Silakan masukkan kata kunci.")
            else:
                st.session_state.awaiting_num_recs_keyword = True
                st.session_state.keyword_for_recommendation = user_input.strip()

        if st.session_state.awaiting_num_recs_keyword:
            num_recommendations = st.selectbox("Jumlah rekomendasi:", [5, 10, 15, 20], index=0, key='num_recs_select_keyword')
            if st.button("Show Recommendations", key='show_recs_keyword'):
                st.session_state.results = recommend_by_keyword(st.session_state.keyword_for_recommendation, num_recommendations)
                st.session_state.awaiting_num_recs_keyword = False

    # ---------- Show results ----------
    results = st.session_state.results
    if results:
        st.subheader("Recommendations:")
        for idx, item in enumerate(results):
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
                checkbox_key = f"show_synopsis_{idx}"
                show_full = st.checkbox("📖 Show full synopsis", key=checkbox_key)
                if show_full:
                    st.markdown(html.escape(item.get('synopsis','')))
                    # Link to Webtoon search (encoded) — open in new tab
                    webtoon_search_url = "https://www.webtoons.com/id/search?keyword=" + urllib.parse.quote(item['title'])
                    st.markdown(f'<a href="{webtoon_search_url}" target="_blank">🔍 Baca/ Cari di Webtoon</a>', unsafe_allow_html=True)
                else:
                    short = textwrap.shorten(item.get('synopsis',''), width=200, placeholder="...")
                    st.markdown(html.escape(short))
            st.markdown("---")

        # review form (same as before)
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

# --- Review Page ---
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
