import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Läs in data
movies = pd.read_csv('Data/ml-latest/movies.csv')
tags = pd.read_csv('Data/ml-latest/tags.csv')

# 2. Gör om genres till listor
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# 3. Filtrera bort filmer utan genres
movies = movies[movies['genres'].apply(lambda x: '(no genres listed)' not in x)]

# 4. Slå ihop tags per film
tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna()))

# 5. Merge movies + tags
movies = movies.merge(tags_grouped, on='movieId', how='left')

# 6. Fyll NaN (filmer utan tags)
movies['tag'] = movies['tag'].fillna('')

# 7. Gör genres till text
movies['genres_text'] = movies['genres'].apply(lambda x: ' '.join(x))

# 8. Kombinera genres + tags
movies['combined'] = movies['genres_text'] + ' ' + movies['tag']

# 9. TF-IDF istället för one-hot
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])


# 10. Rekommendationsfunktion (oförändrad logik)
def recommend(title):
    title_lower = title.lower()

    # Försök hitta filmer som börjar med titeln
    starts_with = movies[movies['title'].str.lower().str.startswith(title_lower)]

    if len(starts_with) > 0:
        matches = starts_with.copy()
    else:
        matches = movies[movies['title'].str.contains(title, case=False, na=False)].copy()

    if len(matches) == 0:
        print("\n❌ Movie not found!")
        return

    print("\n🎬 Did you mean one of these movies?\n")

    options = matches['title'].head(5).tolist()

    for i, movie in enumerate(options, 1):
        print(f"{i}: {movie}")

    max_index = len(options)
    choice = input(f"\nChoose a movie (1-{max_index}, or press Enter to Exit): ").strip()

    if choice == "":
        return

    try:
        choice = int(choice)

        if choice < 1 or choice > max_index:
            print("\n❌ Invalid choice!")
            return

        selected_title = options[choice - 1]

    except:
        print("\n❌ Invalid input!")
        return

    # Hitta index för vald film
    idx = movies[movies['title'] == selected_title].index[0]

    # 🔥 Här är enda viktiga ändringen
    similarity = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix)

    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_5 = sim_scores[1:6]

    print(f"\n🎯 Top 5 similar movies to '{selected_title}':\n")

    for rank, (i, score) in enumerate(top_5, 1):
        print(f"{rank}. {movies.iloc[i]['title']}")


# 11. Kör programmet
if __name__ == "__main__":
    movie = input("Enter movie title: ")
    recommend(movie)