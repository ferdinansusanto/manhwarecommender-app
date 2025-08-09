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
# Config / Constants
# ---------------------------
FUZZY_LIMIT = 5
SUBSTRING_MAX_OPTIONS = 50  # cap jumlah hasil substring agar dropdown tidak terlalu panjang
FUZZY_THRESHOLD = 0  # we will show top FUZZY_LIMIT even if score low when no substring match

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
    if 'title' not in manhwas.columns:
        st.error("Kolom 'title' tidak ditemukan di dataset manhwa_dict_with_cover.pkl.gz")
        st.stop()
    # normalize titles as strings
    manhwas['title'] = manhwas['title'].astype(str)
    titles_list_all = manhwas['title'].dropna().astype(str).tolist()
except Exception as e:
    st.error(f"Error memuat dataset manhwa: {e}")
    st.stop()

try:
    with gzip.open('similarity.pkl.gz', 'rb') as f:
        similarity = pickle.load(f)
    # basic consistency check
    if len(similarity) != len(manhwas):
        st.warning("Peringatan: panjang similarity matrix tidak sama dengan jumlah manhwa. Periksa indexing jika rekomendasi berperilaku aneh.")
except Exception as e:
    st.error(f"Error memuat similarity matrix: {e}")
    st.stop()

try:
    with gzip.open('tag_vectorizer.pkl.gz', 'rb') as f:
        tag_vectorizer = pickle.load(f)
    with gzip.open('tag_vectors.pkl.gz', 'rb') as f:
        tag_vectors = pickle.load(f)
except Exception as e:
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

# fuzzy helper using rapidfuzz: returns list of tuples (display_label, real_title)
def find_fuzzy_labels(query, choices, limit=FUZZY_LIMIT):
    # results: (choice, score, idx)
    results = process.extract(query, choices, limit=limit, scorer=fuzz.token_sort_ratio)
    out = []
    for choice, score, _ in results:
        label = f"{choice} — {int(score)}%"
        out.append((label, choice))
    return out

def recommend_by_title(selected_title, top_n):
    # find index of selected_title
    try:
        idx = manhwas[manhwas['title'] == selected_title].index[0]
    except IndexError:
        st.error("Selected title not found in dataset.")
        return []
    distances = similarity[idx]
    # get top indices sorted descending (include the title itself)
    top_indices = distances.argsort()[::-1][:top_n]
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
if 'selected_title_for_recommendation' not in st.session_state:
    st.session_state.selected_title_for_recommendation = None
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = None

# --- Recommendation Page ---
if page == "Recommendation":
    st.subheader("Recommendation Page")
    mode = st.radio("Select Recommendation Mode:", ["By Title", "By Keyword"])

    # ---------- By Title ----------
    if mode == "By Title":
        st.markdown("Ketik sebagian judul di bawah — dropdown akan menampilkan *combined matching*: substring/full matches (jika ada) atau top fuzzy suggestions (jika tidak ada substring match).")

        # text input for typing query
        title_input = st.text_input("Type the Manhwa Title:", key="title_input")

        # prepare dropdown options (combined behaviour)
        dropdown_display_options = []
        display_to_real = {}  # map displayed label back to real title

        if title_input and title_input.strip():
            q = title_input.strip()
            # substring (case-insensitive) matches
            substring_matches = manhwas[manhwas['title'].str.contains(q, case=False, na=False)]['title'].tolist()
            if substring_matches:
                # limit length for UI
                substring_matches = substring_matches[:SUBSTRING_MAX_OPTIONS]
                dropdown_display_options = substring_matches
                for t in substring_matches:
                    display_to_real[t] = t
                st.info(f"Menampilkan {len(substring_matches)} hasil substring match.")
            else:
                # fallback to fuzzy matching
                fuzzy = find_fuzzy_labels(q, titles_list_all, limit=FUZZY_LIMIT)
                if fuzzy:
                    dropdown_display_options = [label for label, real in fuzzy]
                    for label, real in fuzzy:
                        display_to_real[label] = real
                    st.info("Tidak ada substring match — menampilkan suggestion terdekat (typo correction).")
                else:
                    dropdown_display_options = []
                    st.warning("Tidak ditemukan judul yang cocok atau mirip di dataset.")
        else:
            # when input empty, we still may want to show some popular options or all titles (capped)
            dropdown_display_options = titles_list_all[:SUBSTRING_MAX_OPTIONS]
            for t in dropdown_display_options:
                display_to_real[t] = t

        if dropdown_display_options:
            selected_display = st.selectbox("Choose a Manhwa Title:", dropdown_display_options, key="selectbox_title")
            # map back to real title
            selected_title = display_to_real.get(selected_display, selected_display)
        else:
            selected_display = None
            selected_title = None

        # Find Recommendations button: immediately show default 5 recommendations including the selected title
        if st.button("Find Recommendations", key='find_by_title'):
            if not selected_title:
                st.warning("Silakan pilih judul terlebih dahulu dari dropdown.")
            else:
                st.session_state.selected_title_for_recommendation = selected_title
                st.session_state.results = recommend_by_title(selected_title, 5)

        # Quick-size buttons: only show if we already have a selected title stored
        if st.session_state.selected_title_for_recommendation:
            if st.session_state.selected_title_for_recommendation != selected_title and selected_title:
                # If user picked a new title but hasn't pressed Find, update the selected stored value so grow buttons act on latest choice
                st.session_state.selected_title_for_recommendation = selected_title

            # Show quick-change buttons horizontally
            st.markdown("**Quick change jumlah rekomendasi**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("5", key="btn_5"):
                    st.session_state.results = recommend_by_title(st.session_state.selected_title_for_recommendation, 5)
            with c2:
                if st.button("10", key="btn_10"):
                    st.session_state.results = recommend_by_title(st.session_state.selected_title_for_recommendation, 10)
            with c3:
                if st.button("15", key="btn_15"):
                    st.session_state.results = recommend_by_title(st.session_state.selected_title_for_recommendation, 15)
            with c4:
                if st.button("20", key="btn_20"):
                    st.session_state.results = recommend_by_title(st.session_state.selected_title_for_recommendation, 20)

    # ---------- By Keyword ----------
    elif mode == "By Keyword":
        st.markdown("Masukkan kata kunci (genre, deskripsi, gaya cerita) untuk mencari rekomendasi berdasarkan tag/content.")
        user_input = st.text_input("Enter free keywords (genre, story style, etc.):", key="keyword_input")

        if st.button("Find Recommendations", key='find_by_keyword'):
            if not user_input or not user_input.strip():
                st.warning("Silakan masukkan kata kunci.")
            else:
                st.session_state.selected_title_for_recommendation = None
                st.session_state.results = recommend_by_keyword(user_input.strip(), 5)

        # quick-size buttons for keyword mode too (applies if results exist)
        if st.session_state.results:
            st.markdown("**Quick change jumlah rekomendasi**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("5", key="kbtn_5"):
                    # need to re-run keyword recommendation with 5
                    if user_input and user_input.strip():
                        st.session_state.results = recommend_by_keyword(user_input.strip(), 5)
            with c2:
                if st.button("10", key="kbtn_10"):
                    if user_input and user_input.strip():
                        st.session_state.results = recommend_by_keyword(user_input.strip(), 10)
            with c3:
                if st.button("15", key="kbtn_15"):
                    if user_input and user_input.strip():
                        st.session_state.results = recommend_by_keyword(user_input.strip(), 15)
            with c4:
                if st.button("20", key="kbtn_20"):
                    if user_input and user_input.strip():
                        st.session_state.results = recommend_by_keyword(user_input.strip(), 20)

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
                    st.markdown(f'<a href="{webtoon_search_url}" target="_blank" rel="noopener">🔍 Baca/ Cari di Webtoon</a>', unsafe_allow_html=True)
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
