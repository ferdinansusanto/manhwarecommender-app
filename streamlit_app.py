import streamlit as st
import pandas as pd
import pickle
import gzip
import os
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# Fungsi bintang rating kustom
def star_rating(max_stars=5, default_value=3):
    stars = st.radio("Rating:", range(1, max_stars + 1), index=default_value - 1, format_func=lambda x: "★" * x)
    return stars

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
st.title('📚 Manhwa Recommender System')

# Sidebar untuk navigasi
page = st.sidebar.selectbox("Pilih Halaman:", ["Rekomendasi", "Ulasan"])

if page == "Rekomendasi":
    st.subheader("Halaman Rekomendasi")
    mode = st.radio("Pilih mode rekomendasi:", ["Berdasarkan Judul", "Berdasarkan Keyword Bebas"])
    translator = Translator()

    def translate_to_english(text):
        translated = translator.translate(text, src='id', dest='en')
        return translated.text

    def recommend_by_title(selected_title):
        idx = manhwas[manhwas['title'] == selected_title].index[0]
        distances = similarity[idx]
        manhwa_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        results = []
        for i in manhwa_list:
            row = manhwas.iloc[i[0]]
            results.append((row['title'], row['cover_url']))
        return results

    def recommend_by_keyword(user_input):
        user_vec = tag_vectorizer.transform([user_input])
        scores = cosine_similarity(user_vec, tag_vectors).flatten()
        top_indices = scores.argsort()[::-1][:5]
        results = []
        for i in top_indices:
            row = manhwas.iloc[i]
            results.append((row['title'], row['cover_url']))
        return results

    if mode == "Berdasarkan Judul":
        selected_title = st.selectbox("Pilih judul manhwa:", manhwas['title'].values)
        if st.button("Rekomendasikan"):
            results = recommend_by_title(selected_title)
            st.subheader("Rekomendasi:")
            for title, img_url in results:
                st.image(img_url, width=120)
                st.markdown(f"**{title}**")

    elif mode == "Berdasarkan Keyword Bebas":
        user_input = st.text_input("Masukkan keyword bebas (genre, gaya cerita, dll):")
        if st.button("Cari Rekomendasi"):
            results = recommend_by_keyword(user_input)
            st.subheader("Rekomendasi:")
            for title, img_url in results:
                st.image(img_url, width=120)
                st.markdown(f"**{title}**")

elif page == "Ulasan":
    st.subheader("Ulasan Pengguna")
    username = st.text_input("Nama Pengguna:")
    rating = star_rating()
    review_text = st.text_area("Tulis Ulasan:")

    if st.button("Kirim Ulasan"):
        user_review = {'username': username, 'rating': rating, 'review': review_text}
        save_review(user_review)
        st.success("Ulasan berhasil dikirim!")

    reviews = load_reviews()
    total_visits, average_rating, total_reviews = calculate_statistics(reviews)
    st.write(f"Jumlah Kunjungan: {total_visits}")
    st.write(f"Rating Saat Ini: {average_rating:.2f} dari {total_reviews} ulasan")
    st.subheader("Ulasan Terbaru:")
    if not reviews.empty:
        latest_reviews = reviews.tail(5)
        for index, row in latest_reviews.iterrows():
            st.write(f"**{row['username']}** - Rating: {'★' * row['rating']}")
            st.write(row['review'])
            st.write("---")
    else:
        st.write("Belum ada ulasan.")
