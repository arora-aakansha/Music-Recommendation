
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Switch data from kaggle - main code in kaggle (https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset/input)
#join tables - in Kaggle
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    'song_title': ['Song A', 'Song B', 'Song C', 'Song A', 'Song D', 'Song B', 'Song E', 'Song C', 'Song F', 'Song A'],
    'genre': ['Pop', 'Rock', 'Jazz', 'Pop', 'Classical', 'Rock', 'Hip-Hop', 'Jazz', 'Blues', 'Pop']
}
df = pd.DataFrame(data)

# Preprocess
song_metadata = df.groupby('song_title')['genre'].first().reset_index()

# Create song profiles using TF-IDF vectorizer
tfidf = TfidfVectorizer()
song_profiles = tfidf.fit_transform(song_metadata['genre'])

#similarity matrix
similarity_matrix = cosine_similarity(song_profiles)

# mapping
song_index = pd.Series(song_metadata.index, index=song_metadata['song_title'])

def recommend_songs(song_name, num_recommendations=5):
    if song_name not in song_index:
        return "Song not found. Please try another song."

    idx = song_index[song_name]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]
    recommended_songs = song_metadata.iloc[recommended_indices]['song_title'].tolist()

    return recommended_songs


print("Recommendations for 'Song A':", recommend_songs('Song A'))
