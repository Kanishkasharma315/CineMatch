"""
recommender.py
--------------
CineMatch Recommendation Engine
Implements Content-Based Filtering using TF-IDF Vectorization and Cosine Similarity.
Provides search, filtering, and data aggregation utilities.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """
    Core Recommendation Engine for CineMatch.
    Demonstrates Python OOP, Pandas data manipulation, and Scikit-Learn TF-IDF.
    """

    def __init__(self, dataset_path: str = "data/movies.csv"):
        """
        Initialize the recommender with the path to the movie dataset.
        
        Parameters:
            dataset_path (str): Relative or absolute path to movies.csv
        """
        self.dataset_path = dataset_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.is_ready: bool = False
        self.error_message: str = ""

        # Auto-load dataset upon initialization
        self.load_dataset()

    def load_dataset(self) -> bool:
        """
        Load dataset from CSV file and compute TF-IDF similarity matrix.
        Includes robust error and exception handling.
        
        Returns:
            bool: True if loaded and matrix built successfully, False otherwise.
        """
        try:
            if not os.path.exists(self.dataset_path):
                self.error_message = (
                    f"Dataset not found at '{self.dataset_path}'. "
                    f"Please ensure movies.csv is located in the data directory."
                )
                self.is_ready = False
                return False

            # Read CSV with pandas
            df = pd.read_csv(self.dataset_path)

            # Validate minimum required columns
            required_cols = {"movie_id", "title", "genres", "release_year", "rating", "overview"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                self.error_message = f"Dataset is missing required columns: {missing_cols}"
                self.is_ready = False
                return False

            # Data Cleaning and Preprocessing
            # 1. Fill missing strings
            df["title"] = df["title"].fillna("Unknown Title").astype(str).str.strip()
            df["genres"] = df["genres"].fillna("General").astype(str).str.strip()
            df["director"] = df.get("director", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
            df["overview"] = df["overview"].fillna("").astype(str).str.strip()
            df["poster_url"] = df.get("poster_url", pd.Series([""] * len(df))).fillna("").astype(str)

            # 2. Convert numerical fields safely
            df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000).astype(int)
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(7.0).astype(float)
            df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").fillna(0).astype(int)

            # Reset index cleanly
            df = df.reset_index(drop=True)
            self.df = df

            # Build Recommendation Features (TF-IDF + Cosine Similarity)
            self._build_similarity_matrix()
            self.is_ready = True
            self.error_message = ""
            return True

        except Exception as e:
            self.error_message = f"Error loading dataset: {str(e)}"
            self.is_ready = False
            return False

    def _build_similarity_matrix(self):
        """
        Constructs the content soup for each movie and computes TF-IDF Cosine Similarity.
        Content soup combines: Genres (weighted 2x) + Director + Overview keywords.
        """
        # Create a combined 'metadata soup' string for each film
        def create_soup(row):
            # Repeating genres gives them higher weight in the TF-IDF representation
            genre_tokens = " ".join([g.strip() for g in row["genres"].replace(",", " ").split()])
            director_token = row["director"].replace(" ", "_")  # treat director name as single entity
            return f"{genre_tokens} {genre_tokens} {director_token} {row['overview']}".lower()

        self.df["content_soup"] = self.df.apply(create_soup, axis=1)

        # Scikit-learn TF-IDF Vectorizer
        # Removes standard English stop words ('the', 'is', 'and', etc.)
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)  # unigrams and bigrams
        )

        self.tfidf_matrix = tfidf.fit_transform(self.df["content_soup"])

        # Compute pairwise Cosine Similarity (Range: 0.0 to 1.0)
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_all_movies(self) -> pd.DataFrame:
        """Return the complete movie DataFrame."""
        return self.df.copy() if self.is_ready else pd.DataFrame()

    def get_movie_by_id(self, movie_id: int):
        """Retrieve single movie details as a dict by its unique movie_id."""
        if not self.is_ready:
            return None
        matches = self.df[self.df["movie_id"] == movie_id]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None

    def get_movie_by_title(self, title: str):
        """Retrieve single movie details by case-insensitive title."""
        if not self.is_ready:
            return None
        clean_title = title.strip().lower()
        matches = self.df[self.df["title"].str.lower() == clean_title]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None

    def get_all_genres(self) -> list:
        """
        Extract all unique genres from the dataset, cleaned and alphabetized.
        
        Returns:
            list: Sorted list of distinct genre names.
        """
        if not self.is_ready or self.df.empty:
            return []

        genre_set = set()
        for genre_string in self.df["genres"].dropna():
            # Support both comma-separated and slash/pipe separated
            items = [g.strip() for g in genre_string.replace("/", ",").replace("|", ",").split(",")]
            for item in items:
                if item:
                    genre_set.add(item)
        return sorted(list(genre_set))

    def recommend_similar_movies(self, title: str, top_n: int = 6) -> pd.DataFrame:
        """
        Generate content-based recommendations based on Cosine Similarity.
        
        Parameters:
            title (str): Title of the reference movie.
            top_n (int): Number of similar movies to recommend.
            
        Returns:
            pd.DataFrame: Recommended movies with similarity score.
        """
        if not self.is_ready or self.df.empty:
            return pd.DataFrame()

        # Find exact or substring match for the title
        clean_title = title.strip().lower()
        exact_matches = self.df[self.df["title"].str.lower() == clean_title]

        if not exact_matches.empty:
            target_idx = exact_matches.index[0]
        else:
            # Fallback to partial substring search
            partial_matches = self.df[self.df["title"].str.lower().str.contains(clean_title, na=False)]
            if not partial_matches.empty:
                target_idx = partial_matches.index[0]
            else:
                return pd.DataFrame()

        # Get pairwise similarity scores for this movie
        sim_scores = list(enumerate(self.cosine_sim[target_idx]))

        # Sort movies based on similarity score in descending order
        # sim_scores format: [(index, score), (index, score), ...]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the reference movie itself (index 0 which has score = 1.0)
        sim_scores = [item for item in sim_scores if item[0] != target_idx]

        # Take top_n items
        top_items = sim_scores[:top_n]
        top_indices = [item[0] for item in top_items]
        top_scores = [round(float(item[1]) * 100, 1) for item in top_items]

        # Extract DataFrame rows
        recommendations = self.df.iloc[top_indices].copy()
        recommendations["similarity_score"] = top_scores
        return recommendations

    def filter_movies(
        self,
        genres: list = None,
        min_rating: float = 0.0,
        sort_by: str = "rating",
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Filter movies by multiple genres, minimum rating, and sort.
        
        Parameters:
            genres (list): List of genre strings to match.
            min_rating (float): Minimum rating threshold (0.0 to 10.0).
            sort_by (str): Column to sort by ('rating', 'release_year', 'title').
            ascending (bool): Sort direction.
            
        Returns:
            pd.DataFrame: Filtered and sorted movies.
        """
        if not self.is_ready or self.df.empty:
            return pd.DataFrame()

        filtered_df = self.df.copy()

        # 1. Filter by Rating
        filtered_df = filtered_df[filtered_df["rating"] >= min_rating]

        # 2. Filter by Genres (movie must contain at least one of the selected genres)
        if genres and len(genres) > 0:
            def matches_genre(movie_genres_str):
                movie_genres = [g.strip().lower() for g in movie_genres_str.split(",")]
                return any(g.lower() in movie_genres for g in genres)

            filtered_df = filtered_df[filtered_df["genres"].apply(matches_genre)]

        # 3. Sort
        if sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        return filtered_df

    def search_movies(self, query: str) -> pd.DataFrame:
        """
        Search movies by query in title, director, or overview.
        
        Parameters:
            query (str): Keyword to search.
            
        Returns:
            pd.DataFrame: Matching movie records.
        """
        if not self.is_ready or not query:
            return pd.DataFrame()

        q = query.strip().lower()
        mask = (
            self.df["title"].str.lower().str.contains(q, na=False) |
            self.df["director"].str.lower().str.contains(q, na=False) |
            self.df["genres"].str.lower().str.contains(q, na=False) |
            self.df["overview"].str.lower().str.contains(q, na=False)
        )
        return self.df[mask].copy()

    def get_top_rated(self, top_n: int = 8) -> pd.DataFrame:
        """Return the highest rated movies in the catalog."""
        if not self.is_ready or self.df.empty:
            return pd.DataFrame()
        return self.df.sort_values(by=["rating", "release_year"], ascending=[False, False]).head(top_n).copy()

    def get_stats(self) -> dict:
        """
        Return high-level summary statistics of the catalog.
        
        Returns:
            dict: Summary metrics.
        """
        if not self.is_ready or self.df.empty:
            return {"total_movies": 0, "total_genres": 0, "avg_rating": 0.0, "year_span": "N/A"}

        return {
            "total_movies": len(self.df),
            "total_genres": len(self.get_all_genres()),
            "avg_rating": round(float(self.df["rating"].mean()), 2),
            "year_span": f"{int(self.df['release_year'].min())} - {int(self.df['release_year'].max())}"
        }

    def get_genre_counts(self) -> pd.DataFrame:
        """
        Count number of movies per genre for analytics and charts.
        
        Returns:
            pd.DataFrame: Columns ['Genre', 'Count'] sorted descending.
        """
        if not self.is_ready or self.df.empty:
            return pd.DataFrame(columns=["Genre", "Count"])

        counts = {}
        for genre_str in self.df["genres"]:
            for g in [item.strip() for item in genre_str.split(",")]:
                if g:
                    counts[g] = counts.get(g, 0) + 1

        chart_df = pd.DataFrame(list(counts.items()), columns=["Genre", "Count"])
        return chart_df.sort_values(by="Count", ascending=False).reset_index(drop=True)
