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
from rapidfuzz import process
import urllib.parse

# Fungsi untuk menyimpan ulasan
def save_review(user_review):
    review_df = pd.DataFrame([user_review])
    review_df.to_csv('user_reviews.csv', mode='a', header=not os.path.exists('user_reviews.csv'), index=False)

# Fungsi untuk memuat ulasan
def load_reviews():
    if os.path.exists('user_reviews.csv'):
        return pd.read_csv('user_reviews.csv')
    return pd.DataFrame(columns=['username', 'rating', 'review'])

# Fungsi untuk menghitung statistik
def calculate_statistics(reviews):
    average_rating = reviews['rating'].mean() if not reviews.empty else 0
    total_reviews = len(reviews)
    total_visits = 0
    return total_visits, average_rating, total_reviews

# Load data
with gzip.open('manhwa_dict_with_cover.pkl.gz', 'rb') as f:
    manhwa_dict = pickle.load(f)
manhwas = pd.DataFrame(manhwa_dict)

with gzip.open('similarity.pkl.gz', 'rb') as f:
    similarity = pickle.load(f)

with gzip.open('tag_vectorizer.pkl.gz', 'rb') as f:
    tag_vectorizer = pickle.load(f)

with gzip.open('tag_vectors.pkl.gz', 'rb') as f:
    tag_vectors = pickle.load(f)

# UI setup
st.title('Manhwa Recommender System')

# Sidebar
page = st.sidebar.selectbox("Select Page:", ["Recommendation", "Review"])

translator = Translator()

def translate_to_english(text):
    translated = translator.translate(text, src='id', dest='en')
    return translated.text

# Fuzzy match function
def find_closest_titles(query, choices, limit=5):
    results = process.extract(query, choices, limit=limit, scorer=process.fuzz.token_sort_ratio)
    return [match for match, score, _ in results if score >= 70]

# Recommendation functions
def recommend_by_title(selected_title, top_n):
    idx = manhwas[manhwas['title'] == selected_title].index[0]
    distances = similarity[idx]
    manhwa_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
    results = []
    for i in manhwa_list:
        row = manhwas.iloc[i[0]]
        results.append({
            'title': row['title'],
            'cover_url': row['cover_url'],
            'synopsis': row['synopsis'],
            'genres': row['genres'],
            'authors': row['authors'],
            'score': row['score']
        })
    return results

def recommend_by_keyword(user_input, top_n):
    user_vec = tag_vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, tag_vectors).flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    results = []
    for i in top_indices:
        row = manhwas.iloc[i]
        results.append({
            'title': row['title'],
            'cover_url': row['cover_url'],
            'synopsis': row['synopsis'],
            'genres': row['genres'],
            'authors': row['authors'],
            'score': row['score']
        })
    return results

# Halaman Recommendation
if page == "Recommendation":
    st.subheader("Recommendation Page")

    mode = st.radio("Select Recommendation Mode:", ["By Title", "By Keyword"])

    if 'results' not in st.session_state:
        st.session_state.results = []
        st.session_state.selected_title_final = None

    num_recommendations = 5  # default

    # Input
    if mode == "By Title":
        title_input = st.text_input("Type the Manhwa Title:")
        if st.button("Find Recommendations"):
            matches = find_closest_titles(title_input, manhwas['title'].values, limit=5)
            if matches:
                st.session_state.selected_title_final = matches[0]  # ambil yang paling mirip
                st.success(f"Closest match: {st.session_state.selected_title_final}")
                num_recommendations = st.selectbox("Jumlah rekomendasi:", [5, 10, 15, 20], index=0)
                st.session_state.results = recommend_by_title(st.session_state.selected_title_final, num_recommendations)
            else:
                st.warning("No similar titles found.")

    elif mode == "By Keyword":
        user_input = st.text_input("Enter free keywords (genre, story style, etc.):")
        if st.button("Find Recommendations"):
            num_recommendations = st.selectbox("Jumlah rekomendasi:", [5, 10, 15, 20], index=0)
            st.session_state.results = recommend_by_keyword(user_input, num_recommendations)

    results = st.session_state.results
    if results:
        st.subheader("Recommendations:")

        for idx, item in enumerate(results):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(item['cover_url'], width=120)

            with col2:
                st.markdown(f"### {item['title']}")
                st.markdown(f"**Author:** {item['authors']}")
                st.markdown(f"**Genre:** {item['genres']}")
                st.markdown(f"**Score:** {item['score']}")

                checkbox_key = f"show_synopsis_{idx}"
                show_full = st.checkbox("📖 Show full synopsis", key=checkbox_key)

                if show_full:
                    st.markdown(html.escape(item['synopsis']))
                    # Tambah link langsung ke Webtoon
                    webtoon_search_url = "https://www.webtoons.com/id/search?keyword=" + urllib.parse.quote(item['title'])
                    st.markdown(f"[🔍 Cari di Webtoon]({webtoon_search_url})", unsafe_allow_html=True)
                else:
                    short = textwrap.shorten(item['synopsis'], width=200, placeholder="...")
                    st.markdown(html.escape(short))

            st.markdown("---")

        # Formulir review
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

# Halaman Review
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
    rating_counts = reviews['rating'].value_counts().sort_index()
    for rating_val in range(1, 6):
        count = rating_counts.get(rating_val, 0)
        st.write(f"Rating {rating_val}: {count} User")

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
