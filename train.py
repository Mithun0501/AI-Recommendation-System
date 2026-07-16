import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Keep only needed columns
movies = movies[['title', 'cast', 'crew']]

# Fill missing values
movies = movies.fillna('')

# Create tags
movies['tags'] = movies['cast'] + " " + movies['crew']

# Convert text to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Save files
joblib.dump(movies, "movie_list.pkl")
joblib.dump(similarity, "similarity.pkl")

print("Recommendation Model Created Successfully!")