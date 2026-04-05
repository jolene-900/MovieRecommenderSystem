import streamlit as st
import pandas as pd

movies = pd.read_csv("movies_metadata.csv", low_memory=False)
ratings = pd.read_csv("ratings_small.csv")
links = pd.read_csv("links_small.csv")

print(movies.shape)

import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Select useful columns
movies = movies[['id', 'title', 'overview', 'genres', 'release_date', 'vote_average', 'vote_count']]

# Drop rows with missing important values
movies = movies.dropna(subset=['id', 'title', 'overview'])

# Convert movie id to numeric
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)

# Extract genres
def extract_genres(genre_str):
    try:
        genre_list = ast.literal_eval(genre_str)
        return " ".join([g['name'] for g in genre_list])
    except:
        return ""

movies['genres_clean'] = movies['genres'].apply(extract_genres)

# Clean links
links = links[['movieId', 'tmdbId']]
links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
links = links.dropna(subset=['tmdbId'])
links['tmdbId'] = links['tmdbId'].astype(int)

# Merge datasets
movies_merged = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')

movies_merged = movies_merged[
    ['movieId', 'title', 'overview', 'genres_clean', 'release_date', 'vote_average', 'vote_count']
]

# Remove duplicates
movies_merged = movies_merged.drop_duplicates(subset='movieId')
movies_merged = movies_merged.drop_duplicates(subset='title').reset_index(drop=True)

print("Merged dataset shape:", movies_merged.shape)
movies_merged.head()

# Combine features
movies_merged['combined_features'] = (
    movies_merged['overview'].fillna('') + " " +
    (movies_merged['genres_clean'].fillna('') + " ") * 3
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_merged['combined_features'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Title index mapping
indices = pd.Series(movies_merged.index, index=movies_merged['title']).drop_duplicates()

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Cosine similarity matrix shape:", cosine_sim.shape)

# Merge ratings with movie info
ratings_movies = pd.merge(ratings, movies_merged, on='movieId', how='inner')

# User-item matrix
user_movie_matrix = ratings_movies.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Item-item similarity
movie_similarity = cosine_similarity(user_movie_matrix_filled.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix_filled.columns,
    columns=user_movie_matrix_filled.columns
)

print("Ratings + Movies shape:", ratings_movies.shape)
print("User-movie matrix shape:", user_movie_matrix.shape)
print("Movie similarity matrix shape:", movie_similarity_df.shape)

def recommend_content(title, movies_df, cosine_sim, indices, top_n=10):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 30]

    recommendations = pd.DataFrame({
        'title': [movies_df.iloc[i[0]]['title'] for i in sim_scores],
        'model_score': [i[1] for i in sim_scores]
    })

    recommendations = pd.merge(
        recommendations,
        movies_df[['title', 'genres_clean', 'release_date', 'vote_average', 'vote_count']],
        on='title',
        how='left'
    )

    recommendations = recommendations[recommendations['vote_count'] > 50]
    recommendations = recommendations.drop_duplicates(subset='title')

    return recommendations.head(top_n)

def recommend_collaborative(movie_title, movies_df, movie_similarity_df, top_n=10):
    if movie_title not in movie_similarity_df.columns:
        return pd.DataFrame()

    similar_scores = movie_similarity_df[movie_title].sort_values(ascending=False)
    similar_scores = similar_scores.iloc[1: top_n + 30]

    recommendations = pd.DataFrame({
        'title': similar_scores.index,
        'model_score': similar_scores.values
    })

    recommendations = pd.merge(
        recommendations,
        movies_df[['title', 'genres_clean', 'release_date', 'vote_average', 'vote_count']],
        on='title',
        how='left'
    )

    recommendations = recommendations[recommendations['vote_count'] > 50]
    recommendations = recommendations.drop_duplicates(subset='title')

    return recommendations.head(top_n)

def hybrid_recommend(movie_title, movies_df, cosine_sim, indices, movie_similarity_df, top_n=10, alpha=0.5):
    if movie_title not in indices:
        return pd.DataFrame()

    if movie_title not in movie_similarity_df.columns:
        return pd.DataFrame()

    # Content part
    idx = indices[movie_title]
    sim_scores_content = list(enumerate(cosine_sim[idx]))
    sim_scores_content = sorted(sim_scores_content, key=lambda x: x[1], reverse=True)
    sim_scores_content = sim_scores_content[1:100]

    content_df = pd.DataFrame({
        'title': [movies_df.iloc[i[0]]['title'] for i in sim_scores_content],
        'content_score': [i[1] for i in sim_scores_content]
    })

    # Collaborative part
    sim_scores_collab = movie_similarity_df[movie_title].sort_values(ascending=False)
    sim_scores_collab = sim_scores_collab.iloc[1:100]

    collab_df = pd.DataFrame({
        'title': sim_scores_collab.index,
        'collab_score': sim_scores_collab.values
    })

    # Merge both
    hybrid_df = pd.merge(content_df, collab_df, on='title', how='outer')
    hybrid_df['content_score'] = hybrid_df['content_score'].fillna(0)
    hybrid_df['collab_score'] = hybrid_df['collab_score'].fillna(0)

    # Normalize
    if hybrid_df['content_score'].max() != 0:
        hybrid_df['content_score'] /= hybrid_df['content_score'].max()

    if hybrid_df['collab_score'].max() != 0:
        hybrid_df['collab_score'] /= hybrid_df['collab_score'].max()

    # Final score
    hybrid_df['model_score'] = alpha * hybrid_df['content_score'] + (1 - alpha) * hybrid_df['collab_score']

    # Merge movie info
    hybrid_df = pd.merge(
        hybrid_df,
        movies_df[['title', 'genres_clean', 'release_date', 'vote_average', 'vote_count']],
        on='title',
        how='left'
    )

    hybrid_df = hybrid_df[hybrid_df['vote_count'] > 50]
    hybrid_df = hybrid_df.drop_duplicates(subset='title')
    hybrid_df = hybrid_df.sort_values(by='model_score', ascending=False)

    return hybrid_df.head(top_n)

def get_explore_mode(movie_title, top_n=5):
    """
    Explore Mode:
    shows less obvious but still relevant recommendations
    from the hybrid recommender.
    """
    results = hybrid_recommend(
        movie_title,
        movies_merged,
        cosine_sim,
        indices,
        movie_similarity_df,
        top_n=20,
        alpha=0.5
    )

    if results.empty:
        return results

    explore_results = results.iloc[5:5 + top_n].copy()

    if explore_results.empty:
        explore_results = results.head(top_n).copy()

    explore_results['explanation'] = (
        "Recommended to help you explore something different "
        "instead of only the most obvious choices."
    )

    return explore_results


def get_new_user_recommendations(personality=None, mood=None, top_n=10):
    """
    For new users who do not enter a movie title.
    Uses mood + personality + high-rated movies.
    """
    results = movies_merged.copy()

    if personality:
        results = apply_personality_filter(results, personality)

    if mood:
        results = apply_mood_filter(results, mood)

    results = results.sort_values(
        by=['vote_average', 'vote_count'],
        ascending=[False, False]
    )

    results = results[results['vote_count'] > 100]
    results = results.drop_duplicates(subset='title')
    results = results.head(top_n).copy()

    results['model_score'] = results['vote_average']
    results['explanation'] = (
        "Recommended for new users based on selected mood, "
        "personality, and overall movie quality."
    )

    return results[
        ['title', 'genres_clean', 'release_date', 'vote_average',
         'vote_count', 'model_score', 'explanation']
    ]


# Map user mood to corresponding movie genres
def mood_to_genres(mood):
    mood_map = {
        "Happy": ["Comedy", "Family", "Animation"],
        "Sad": ["Drama"],
        "Romantic": ["Romance"],
        "Excited": ["Action", "Adventure", "Thriller", "Science Fiction"],
        "Curious": ["Mystery", "Documentary", "History"],
        "Scared": ["Horror", "Thriller"],
        "Relaxed": ["Music", "Family", "Comedy"]
    }
    # Return an empty list if mood is not found
    return mood_map.get(mood, [])


# Filter movies based on user personality
def apply_personality_filter(df, personality):
    personality_map = {
        "Adventurer": ["Adventure", "Action"],
        "Romantic": ["Romance", "Drama"],
        "Thinker": ["Science Fiction", "Documentary", "Mystery"],
        "Fun Lover": ["Comedy", "Animation", "Family"],
        "Dreamer": ["Fantasy", "Science Fiction", "Animation"],
        "Bold Explorer": ["Action", "Thriller", "Adventure"]
    }
    
    genres = personality_map.get(personality, [])
    if not genres or df.empty:
        return df

    # Use regex to match movies containing the mapped genres
    pattern = "|".join(genres)
    filtered = df[df['genres_clean'].str.contains(pattern, case=False, na=False)].copy()

    if filtered.empty:
        return df
    return filtered


# Search for movie titles (called from main menu)
def search_movie_titles():
    print("\n" + "=" * 60)
    print("🔍 SEARCH MOVIE TITLES")
    print("=" * 60)
    user_input = input("Enter a keyword to search for movies: ").strip()
    
    # Get matches using the existing search function
    matches = find_movie_matches(user_input, movies_merged, max_results=15)
    
    if not matches:
        print(f"No movies found matching '{user_input}'.")
    else:
        print(f"\nFound {len(matches)} matching movies:")
        for i, title in enumerate(matches, start=1):
            print(f"{i}. {title}")

def apply_mood_filter(df, mood):
    genres = mood_to_genres(mood)
    if not genres or df.empty:
        return df

    pattern = "|".join(genres)
    filtered = df[df['genres_clean'].str.contains(pattern, case=False, na=False)].copy()

    if filtered.empty:
        return df
    return filtered

def get_hidden_gems(df):
    if df.empty:
        return df

    gems = df[(df['vote_average'] >= 6.5) & (df['vote_count'] < 500)].copy()

    if gems.empty:
        return df
    return gems

def explain_recommendation(row, mode):
    genres = row['genres_clean'] if pd.notna(row['genres_clean']) else "similar genres"

    if mode == "Content-Based":
        return f"Recommended because it has similar storyline and genres: {genres}."
    elif mode == "Collaborative":
        return "Recommended because users who liked the selected movie also liked this movie."
    elif mode == "Hybrid":
        return f"Recommended using both content similarity and user rating behaviour, with related genres: {genres}."
    else:
        return "Recommended based on system analysis."

def get_recommendations(
    movie_title,
    mode="Hybrid",
    top_n=10,
    mood=None,
    hidden_gems=False,
    personality=None,
    alpha=0.5
):
    if mode == "Content-Based":
        results = recommend_content(movie_title, movies_merged, cosine_sim, indices, top_n=top_n)
    elif mode == "Collaborative":
        results = recommend_collaborative(movie_title, movies_merged, movie_similarity_df, top_n=top_n)
    else:
        results = hybrid_recommend(
            movie_title,
            movies_merged,
            cosine_sim,
            indices,
            movie_similarity_df,
            top_n=top_n,
            alpha=alpha
        )

    if results.empty:
        print("Movie not found or no recommendations available.")
        return results

    if mood:
        results = apply_mood_filter(results, mood)

    if personality:
        results = apply_personality_filter(results, personality)

    if hidden_gems:
        results = get_hidden_gems(results)

    results = results.head(top_n).copy()
    results['explanation'] = results.apply(lambda row: explain_recommendation(row, mode), axis=1)

    return results

content_result = get_recommendations(
    movie_title="Toy Story",
    mode="Content-Based",
    top_n=10
)

content_result

collab_result = get_recommendations(
    movie_title="Toy Story",
    mode="Collaborative",
    top_n=10
)

collab_result

hybrid_result = get_recommendations(
    movie_title="Toy Story",
    mode="Hybrid",
    top_n=10,
    alpha=0.5
)

hybrid_result

mood_result = get_recommendations(
    movie_title="Toy Story",
    mode="Hybrid",
    top_n=10,
    mood="Happy"
)

mood_result

hidden_result = get_recommendations(
    movie_title="Toy Story",
    mode="Hybrid",
    top_n=10,
    hidden_gems=True
)

hidden_result

def show_recommendations(df):
    if df.empty:
        print("\nNo recommendations found.\n")
        return

    print("\n" + "=" * 80)
    print("🎥 RECOMMENDED MOVIES")
    print("=" * 80)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\n{i}. {row['title']}")
        print("-" * 80)
        print(f"Genres       : {row['genres_clean']}")
        print(f"Release Date : {row['release_date']}")
        print(f"Rating       : {row['vote_average']}")
        print(f"Votes        : {row['vote_count']}")
        if 'model_score' in row:
            try:
                print(f"Score        : {row['model_score']:.4f}")
            except:
                print(f"Score        : {row['model_score']}")
        print(f"Why          : {row['explanation']}")
        print("=" * 80)

result = get_recommendations(
    movie_title="Toy Story",
    mode="Hybrid",
    top_n=5,
    mood="Happy"
)

show_recommendations(result)

def view_system_features():
    print("\n" + "=" * 75)
    print("✨ SYSTEM FEATURES")
    print("=" * 75)
    print("1. Content-Based Recommendation")
    print("   Recommends movies with similar storyline and genres.")
    print()
    print("2. Collaborative Filtering")
    print("   Recommends movies liked by users with similar interests.")
    print()
    print("3. Hybrid Recommendation")
    print("   Combines content-based and collaborative filtering results.")
    print()
    print("4. Explore Mode")
    print("   Suggests less obvious but still relevant movies.")
    print()
    print("5. New User Recommendation")
    print("   Provides recommendations even without a selected movie.")
    print()
    print("6. Mood-Based Filter")
    print("   Suggests movies based on the selected mood.")
    print()
    print("7. Personality-Based Filter")
    print("   Suggests movies that match the selected personality type.")
    print()
    print("8. Hidden Gems")
    print("   Highlights less popular but highly rated movies.")
    print()
    print("9. Explainable Recommendation")
    print("   Shows the reason why each movie is recommended.")
    print()
    print("10. Method Comparison")
    print("    Allows comparison of content-based, collaborative, and hybrid results.")
    print("-" * 75)


def find_movie_matches(user_input, movies_df, max_results=10):
    """
    Find movie titles by case-insensitive partial matching.
    """
    if not user_input:
        return []

    user_input = user_input.strip().lower()
    all_titles = movies_df['title'].dropna().unique()

    matches = [title for title in all_titles if user_input in title.lower()]

    return matches[:max_results]


def safe_input_choice(prompt, valid_choices):
    """
    Keep asking until user enters a valid choice.
    """
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid input. Please enter one of: {', '.join(valid_choices)}")


def resolve_movie_title(user_input, movies_df):
    """
    Resolve movie title using smarter rules:
    1. If input is too short, show suggestions only
    2. Else try exact case-insensitive match
    3. If no exact match, show partial match suggestions
    """
    if not user_input.strip():
        print("Movie title cannot be empty.")
        return None

    user_input_clean = user_input.strip().lower()
    all_titles = movies_df['title'].dropna().unique()

    # Rule 1: very short input -> do not exact match
    if len(user_input_clean) <= 2:
        matches = find_movie_matches(user_input, movies_df, max_results=10)

        if len(matches) == 0:
            print("No related movie titles found.")
            return None

        print("\nYour input is too short, so here are some related movie titles:")
        for i, title in enumerate(matches, start=1):
            print(f"{i}. {title}")
        print("0. Cancel")

        while True:
            choice = input("Select a movie number: ").strip()

            if choice.isdigit():
                choice = int(choice)

                if choice == 0:
                    return None

                if 1 <= choice <= len(matches):
                    return matches[choice - 1]

            print("Invalid selection. Please enter a valid number.")

    # Rule 2: exact case-insensitive match
    for title in all_titles:
        if title.lower() == user_input_clean:
            return title

    # Rule 3: partial match suggestions
    matches = find_movie_matches(user_input, movies_df, max_results=10)

    if len(matches) == 0:
        print("No related movie titles found.")
        return None

    print("\nMovie not found exactly. Did you mean one of these?")
    for i, title in enumerate(matches, start=1):
        print(f"{i}. {title}")
    print("0. Cancel")

    while True:
        choice = input("Select a movie number: ").strip()

        if choice.isdigit():
            choice = int(choice)

            if choice == 0:
                return None

            if 1 <= choice <= len(matches):
                return matches[choice - 1]

        print("Invalid selection. Please enter a valid number.")

def safe_numeric_choice(prompt, min_value, max_value, default=None):
    """
    Ask user for a number with validation.
    """
    while True:
        value = input(prompt).strip()

        if value == "" and default is not None:
            return default

        if value.isdigit():
            value = int(value)
            if min_value <= value <= max_value:
                return value

        print(f"Invalid input. Please enter a number from {min_value} to {max_value}.")


def choose_mood_by_number():
    mood_options = {
        1: None,
        2: "Happy",
        3: "Sad",
        4: "Romantic",
        5: "Excited",
        6: "Curious",
        7: "Scared",
        8: "Relaxed"
    }

    print("\nChoose your mood:")
    print("1. None")
    print("2. Happy")
    print("3. Sad")
    print("4. Romantic")
    print("5. Excited")
    print("6. Curious")
    print("7. Scared")
    print("8. Relaxed")

    choice = safe_numeric_choice("Enter your choice (1-8): ", 1, 8)
    return mood_options[choice]


def choose_personality_by_number():
    personality_options = {
        1: None,
        2: "Adventurer",
        3: "Romantic",
        4: "Thinker",
        5: "Fun Lover",
        6: "Dreamer",
        7: "Bold Explorer"
    }

    print("\nChoose your personality type:")
    print("1. None")
    print("2. Adventurer")
    print("3. Romantic")
    print("4. Thinker")
    print("5. Fun Lover")
    print("6. Dreamer")
    print("7. Bold Explorer")

    choice = safe_numeric_choice("Enter your choice (1-7): ", 1, 7)
    return personality_options[choice]


def print_result_summary(mode, movie_title, count):
    print("\n" + "=" * 80)
    print("📌 RECOMMENDATION SUMMARY")
    print("=" * 80)
    print(f"Recommendation Type : {mode}")
    if movie_title:
        print(f"Selected Movie      : {movie_title}")
    print(f"Total Results       : {count}")
    print("=" * 80)


def compare_all_methods(movie_title, top_n=5):
    print("\n" + "=" * 75)
    print(f"COMPARISON OF RECOMMENDATION METHODS FOR: {movie_title}")
    print("=" * 75)

    print("\n1. CONTENT-BASED RESULTS")
    content = get_recommendations(
        movie_title=movie_title,
        mode="Content-Based",
        top_n=top_n
    )
    print_result_summary("Content-Based", movie_title, len(content))
    show_recommendations(content)

    print("\n2. COLLABORATIVE FILTERING RESULTS")
    collab = get_recommendations(
        movie_title=movie_title,
        mode="Collaborative",
        top_n=top_n
    )
    print_result_summary("Collaborative", movie_title, len(collab))
    show_recommendations(collab)

    print("\n3. HYBRID RESULTS")
    hybrid = get_recommendations(
        movie_title=movie_title,
        mode="Hybrid",
        top_n=top_n
    )
    print_result_summary("Hybrid", movie_title, len(hybrid))
    show_recommendations(hybrid)

def get_recommendations(
    movie_title=None,
    mode="Hybrid",
    top_n=10,
    mood=None,
    hidden_gems=False,
    personality=None,
    alpha=0.5
):
    if mode == "New User":
        results = get_new_user_recommendations(
            personality=personality,
            mood=mood,
            top_n=top_n
        )

    elif mode == "Explore":
        results = get_explore_mode(movie_title, top_n=top_n)

    elif mode == "Content-Based":
        results = recommend_content(movie_title, movies_merged, cosine_sim, indices, top_n=top_n)

    elif mode == "Collaborative":
        results = recommend_collaborative(movie_title, movies_merged, movie_similarity_df, top_n=top_n)

    else:
        results = hybrid_recommend(
            movie_title,
            movies_merged,
            cosine_sim,
            indices,
            movie_similarity_df,
            top_n=top_n,
            alpha=alpha
        )

    if results.empty:
        print("Movie not found or no recommendations available.")
        return results

    if mood and mode not in ["New User"]:
        results = apply_mood_filter(results, mood)

    if personality and mode not in ["New User"]:
        results = apply_personality_filter(results, personality)

    if hidden_gems:
        results = get_hidden_gems(results)

    results = results.head(top_n).copy()

    if 'explanation' not in results.columns:
        results['explanation'] = results.apply(
            lambda row: explain_recommendation(row, mode if mode in ["Content-Based", "Collaborative", "Hybrid"] else "Hybrid"),
            axis=1
        )

    return results

def show_recommendations(df):
    if df.empty:
        print("No recommendations found.")
        return

    print("\n🎥 Here are your recommended movies:\n")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"{i}. {row['title']}")
        print(f"   Genre(s): {row['genres_clean']}")
        print(f"   Release Date: {row['release_date']}")
        print(f"   Average Rating: {row['vote_average']}")
        print(f"   Number of Votes: {row['vote_count']}")
        if 'model_score' in row:
            try:
                print(f"   Recommendation Score: {row['model_score']:.4f}")
            except:
                print(f"   Recommendation Score: {row['model_score']}")
        print(f"   Why this movie? {row['explanation']}")
        print("-" * 75)

def recommendation_page():
    print("\n" + "=" * 60)
    print("🎬 START RECOMMENDATION")
    print("=" * 60)

    print("\nChoose a recommendation type:")
    print("1. Content-Based")
    print("2. Collaborative Filtering")
    print("3. Hybrid")
    print("4. Explore Mode")
    print("5. New User Recommendation")

    mode_choice = safe_input_choice("Enter your choice (1-5): ", ["1", "2", "3", "4", "5"])

    mode_map = {
        "1": "Content-Based",
        "2": "Collaborative",
        "3": "Hybrid",
        "4": "Explore",
        "5": "New User"
    }
    mode = mode_map[mode_choice]

    movie_title = None
    if mode != "New User":
        print("\nEnter a movie title.")
        print("You can type the full title or part of the title.")
        user_movie_input = input("Movie title: ").strip()
        movie_title = resolve_movie_title(user_movie_input, movies_merged)

        if movie_title is None:
            print("\nNo valid movie selected.")
            return

    print("\nOptional filters")
    mood = choose_mood_by_number()
    personality = choose_personality_by_number()

    hidden_input = input("\nShow hidden gems only? (yes/no): ").strip().lower()
    hidden_gems = True if hidden_input == "yes" else False

    top_n = safe_numeric_choice(
        "\nHow many recommendations would you like? (1-10): ",
        1, 10, default=5
    )

    alpha = 0.5
    if mode == "Hybrid":
        alpha_input = input(
            "Enter hybrid alpha (0.0 to 1.0, default 0.5): "
        ).strip()
        try:
            alpha = float(alpha_input) if alpha_input != "" else 0.5
            if alpha < 0 or alpha > 1:
                alpha = 0.5
        except:
            alpha = 0.5

    print("\nGenerating recommendations...\n")

    results = get_recommendations(
        movie_title=movie_title,
        mode=mode,
        top_n=top_n,
        mood=mood,
        hidden_gems=hidden_gems,
        personality=personality,
        alpha=alpha
    )

    print_result_summary(mode, movie_title, len(results))
    show_recommendations(results)

    while True:
        print("\nNext action:")
        print("1. Try another recommendation")
        print("2. Compare all methods")
        print("3. Return to main menu")
        print("4. Exit")

        next_choice = safe_input_choice("Choose an option (1/2/3/4): ", ["1", "2", "3", "4"])

        if next_choice == "1":
            recommendation_page()
            return
        elif next_choice == "2":
            if movie_title:
                compare_all_methods(movie_title, top_n=5)
            else:
                print("Comparison is only available when a movie title is selected.")
        elif next_choice == "3":
            return
        else:
            print("\nThank you for using the system. Goodbye! 👋")
            raise SystemExit

def main_menu():
    while True:
        print("\n" + "=" * 60)
        print("🎬 MOVIE RECOMMENDATION SYSTEM")
        print("=" * 60)
        print("1. Start Recommendation")
        print("2. Search Movie Titles")
        print("3. View Features")
        print("4. Exit")
        print("-" * 60)

        choice = safe_input_choice("Choose an option (1/2/3/4): ", ["1", "2", "3", "4"])

        if choice == "1":
            recommendation_page()
        elif choice == "2":
            search_movie_titles()
        elif choice == "3":
            view_system_features()
        else:
            print("\nThank you for using the system. Goodbye! 👋")
            break

main_menu()
