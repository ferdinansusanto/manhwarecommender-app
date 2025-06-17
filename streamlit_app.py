import streamlit as st
import pandas as pd
import pickle
import gzip
import os
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from io import BytesIO
import xlsxwriter

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
    total_visits = 0  # Ganti dengan logika untuk menghitung kunjungan
    average_rating = reviews['rating'].mean() if not reviews.empty else 0
    total_reviews = len(reviews)
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

# Setup UI
st.title('Manhwa Recommender System')

# Sidebar untuk navigasi
page = st.sidebar.selectbox("Select Page:", ["Recommendation", "Review"])

if page == "Recommendation":
    st.subheader("Recommendation Page")

    mode = st.radio("Select Recommendation Mode:", ["By Title", "By Keyword"])

    # Fitur Terjemahan
    translator = Translator()

    def translate_to_english(text):
        translated = translator.translate(text, src='id', dest='en')
        return translated.text

    # Fungsi Rekomendasi Berdasarkan Judul
    def recommend_by_title(selected_title):
        idx = manhwas[manhwas['title'] == selected_title].index[0]
        distances = similarity[idx]
        manhwa_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        results = []
        for i in manhwa_list:
            row = manhwas.iloc[i[0]]
            results.append((row['title'], row['cover_url']))
        return results

    # Fungsi Rekomendasi Berdasarkan Keyword
    def recommend_by_keyword(user_input):
        user_vec = tag_vectorizer.transform([user_input])
        scores = cosine_similarity(user_vec, tag_vectors).flatten()
        top_indices = scores.argsort()[::-1][:5]
        
        results = []
        for i in top_indices:
            row = manhwas.iloc[i]
            results.append((row['title'], row['cover_url']))
        return results

    # Mode: Judul
    if mode == "By Title":
        selected_title = st.selectbox("Choose a Manhwa Title:", manhwas['title'].values)
        if st.button("Find Recommendations"):
            results = recommend_by_title(selected_title)
            st.subheader("Recommendations:")
            for title, img_url in results:
                st.image(img_url, width=120)
                st.markdown(f"**{title}**")

    # Mode: Keyword
    elif mode == "By Keyword":
        user_input = st.text_input("Enter free keywords (genre, story style, etc.):")
        if st.button("Find Recommendations"):
            results = recommend_by_keyword(user_input)
            st.subheader("Recommendations:")
            for title, img_url in results:
                st.image(img_url, width=120)
                st.markdown(f"**{title}**")

elif page == "Review":
    st.subheader("User Review")

    # Kumpulkan Ulasan dari Pengguna
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

    # Muat dan tampilkan statistik
    reviews = load_reviews()
    total_visits, average_rating, total_reviews = calculate_statistics(reviews)
    # st.write(f"Jumlah Kunjungan: {total_visits}")
    st.write(f"Current Rating: {average_rating:.2f} from {total_reviews} reviews")

    # Distribusi rating
    st.subheader("Rating Distribution:")
    rating_counts = reviews['rating'].value_counts().sort_index()
    for rating_val in range(1, 6):
        count = rating_counts.get(rating_val, 0)
        st.write(f"Rating {rating_val}: {count} User")

    # Tampilkan beberapa ulasan terbaru
    st.subheader("Latest Review:")
    if not reviews.empty:
        latest_reviews = reviews.tail(5)
        for index, row in latest_reviews.iterrows():
            st.write(f"**{row['username']}** - Rating: {row['rating']}")
            st.write(row['review'])
            st.write("---")
    else:
        st.write("No reviews yet.")

    # Unduh data sebagai Excel
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
    
    # Semua ulasan
    st.subheader("All Reviews:")
    st.dataframe(reviews)


