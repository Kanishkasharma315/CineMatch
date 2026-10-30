"""
test_recommender.py
-------------------
Unit tests for CineMatch Recommendation Engine.
Demonstrates Python test automation and code verification for academic submission.
"""

import unittest
import pandas as pd
from recommender import MovieRecommender


class TestMovieRecommender(unittest.TestCase):
    """Test suite for MovieRecommender class."""

    @classmethod
    def setUpClass(cls):
        """Instantiate recommender once for all test cases."""
        cls.recommender = MovieRecommender(dataset_path="data/movies.csv")

    def test_recommender_initialization(self):
        """Test that dataset loads properly and engine is marked ready."""
        self.assertTrue(self.recommender.is_ready, "Recommender should be initialized and ready.")
        self.assertGreater(len(self.recommender.df), 30, "Dataset should contain at least 30 movies.")

    def test_get_all_genres(self):
        """Test extraction of genres."""
        genres = self.recommender.get_all_genres()
        self.assertIsInstance(genres, list)
        self.assertIn("Action", genres)
        self.assertIn("Drama", genres)
        self.assertIn("Sci-Fi", genres)

    def test_recommend_similar_movies_valid_movie(self):
        """Test that recommendation returns top-n similar movies with similarity scores."""
        recs = self.recommender.recommend_similar_movies("Inception", top_n=3)
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 3, "Should return requested top_n recommendations.")
        self.assertIn("similarity_score", recs.columns)
        # Inception itself must NOT be in the recommendations
        self.assertNotIn("Inception", recs["title"].tolist(), "Selected movie must not recommend itself.")

    def test_recommend_similar_movies_nonexistent(self):
        """Test recommendation behavior when movie does not exist."""
        recs = self.recommender.recommend_similar_movies("NonExistentMovieTitle999", top_n=3)
        self.assertTrue(recs.empty, "Should return empty DataFrame for non-existent movie.")

    def test_search_movies_found(self):
        """Test search query for known movie."""
        results = self.recommender.search_movies("Matrix")
        self.assertFalse(results.empty)
        self.assertTrue(any("Matrix" in t for t in results["title"].tolist()))

    def test_search_movies_not_found(self):
        """Test search query for nonexistent string."""
        results = self.recommender.search_movies("ZzzzXxxxUnknown")
        self.assertTrue(results.empty)

    def test_filter_by_genres_and_rating(self):
        """Test genre and rating filter constraints."""
        filtered = self.recommender.filter_movies(genres=["Animation"], min_rating=8.0)
        self.assertFalse(filtered.empty)
        for _, row in filtered.iterrows():
            self.assertIn("Animation", row["genres"])
            self.assertGreaterEqual(row["rating"], 8.0)

    def test_missing_dataset_handling(self):
        """Test graceful error handling when CSV path is invalid."""
        bad_rec = MovieRecommender(dataset_path="data/does_not_exist.csv")
        self.assertFalse(bad_rec.is_ready)
        self.assertIn("Dataset not found", bad_rec.error_message)


if __name__ == "__main__":
    unittest.main()
