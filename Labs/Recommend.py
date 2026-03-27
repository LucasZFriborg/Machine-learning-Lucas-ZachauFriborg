import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv('Data/ml-latest/movies.csv')
tags = pd.read_csv('Data/ml-latest/tags.csv')

# splitting genres string into a list (e.g "Action|Drama" -> ["Action", "Drama"])
movies['genres'] = movies['genres'].str.split('|')

# function to filter out movies with valid genres
def has_genres(genre_list):
    return '(no genres listed)' not in genre_list

# keeping only movies that have at least one genre
movies = movies[movies['genres'].apply(has_genres)]

# removing rows where tag is missing (NaN)
tags = tags.dropna(subset=['tag'])

# grouping tags by movieId and concatenate them into a single string
# (e.g ["dark", "mafia"] -> "dark mafia")
tags_grouped = tags.groupby('movieId')['tag'].apply(' '.join)

# merging movies with grouped tags (left join keeps all movies)
movies = movies.merge(tags_grouped, on='movieId', how='left')

# replacing missing tags with empty strings
movies['tag'] = movies['tag'].fillna('')

# converting genres list into a space-separated string
movies['genres_text'] = movies['genres'].apply(lambda genre_list: ' '.join(genre_list))

# combining genres and tags into one text feature
# this will be used as input for TF-IDF
movies['combined'] = movies['genres_text'] + ' ' + movies['tag']

# initializing TF-IDF vectorizer (removes common English stop words)
tfidf = TfidfVectorizer(stop_words='english')

# transforming text data into TF-IDF feature matrix
tfidf_matrix = tfidf.fit_transform(movies['combined'])

def recommend(title):
    # converting input to lowercase for case-insensitive matching
    title_lower = title.lower()

    # first try to find titles that start with the input (more precise match)
    starts_with_matches = movies[movies['title'].str.lower().str.startswith(title_lower)]

    if len(starts_with_matches) > 0:
        matches = starts_with_matches.copy()
    else:
        # if no startswith matches, fall back to broader search (contains)
        contains_matches = movies[movies['title'].str.contains(title, case=False, na=False)]
        matches = contains_matches.copy()

    # if no matches found, exit
    if len(matches) == 0:
        print("\nMovie not found!")
        return

    print("\nDid you mean?\n")

    # showing up to 5 possible matches
    options = matches['title'].head(5).tolist()

    for i, movie_name in enumerate(options, 1):
        print(f"{i}: {movie_name}")

    max_index = len(options)
    choice = input(f"\nChoose a movie (1-{max_index}, or press Enter to Exit): ").strip()

    # allowing user to exit without making a choice
    if choice == "":
        return

    try:
        choice = int(choice)

        # validating user input
        if choice < 1 or choice > max_index:
            print("\nInvalid choice!")
            return

        selected_title = options[choice - 1]

    except:
        # handle non-integer input
        print("\nInvalid input!")
        return

    # getting index of selected movie
    selected_movie_index = movies[movies['title'] == selected_title].index[0]

    # computing cosine similarity between selected movie and all others
    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_movie_index:selected_movie_index + 1],
        tfidf_matrix
    )

    # pairing each movie with its similarity score
    similarity_list = list(enumerate(similarity_scores[0]))
    
    # sorting movies by similarity (highest first)
    similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)

    # selecting top 5 similar movies (skip the first one = itself)
    top_5 = similarity_list[1:6]

    print(f"\nTop 5 similar movies to '{selected_title}':\n")

    # printing recommendations
    for rank, (movie_index, score) in enumerate(top_5, 1):
        movie_title = movies.iloc[movie_index]['title']
        print(f"{rank}. {movie_title}")

if __name__ == "__main__":
    user_input = input("Enter movie title: ")
    recommend(user_input)