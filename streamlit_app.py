import streamlit as st
import pickle
import pandas as pd
import gzip
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

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

mode = st.radio("Pilih mode rekomendasi:", ["Berdasarkan Judul", "Berdasarkan Keyword Bebas"])

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
    # Translate user input to English if necessary
    if is_indonesian(user_input):
        user_input = translate_to_english(user_input)
    
    user_vec = tag_vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, tag_vectors).flatten()
    top_indices = scores.argsort()[::-1][:5]
    
    results = []
    for i in top_indices:
        row = manhwas.iloc[i]
        results.append((row['title'], row['cover_url']))
    return results

# Mode: Judul
if mode == "Berdasarkan Judul":
    selected_title = st.selectbox("Pilih judul manhwa:", manhwas['title'].values)
    if st.button("Rekomendasikan"):
        results = recommend_by_title(selected_title)
        st.subheader("Rekomendasi:")
        for title, img_url in results:
            st.image(img_url, width=120)
            st.markdown(f"**{title}**")

# Mode: Keyword
elif mode == "Berdasarkan Keyword Bebas":
    user_input = st.text_input("Masukkan keyword bebas (genre, gaya cerita, dll):")
    if st.button("Cari Rekomendasi"):
        results = recommend_by_keyword(user_input)
        st.subheader("Rekomendasi:")
        for title, img_url in results:
            st.image(img_url, width=120)
            st.markdown(f"**{title}**")
