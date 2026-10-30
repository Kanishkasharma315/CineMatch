"""
CineMatch - Movie Recommendation System
Main Streamlit Application File.
"""

import os
import re
import html
import base64
import streamlit as st
import pandas as pd

from recommender import MovieRecommender

# Optional Plotly import with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="CineMatch – Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# 2. HELPER FUNCTION
# ==============================================================================

def clean_html(html_str: str) -> str:
    """Clean HTML whitespace for Streamlit rendering."""
    return "\n".join(
        line.strip()
        for line in html_str.splitlines()
        if line.strip()
    )


# ==============================================================================
# 3. LOAD CUSTOM CSS
# ==============================================================================

def load_custom_css():
    """Load CSS from assets/style.css."""

    css_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "style.css"
    )

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #0E1117;
                    color: #FFFFFF;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


load_custom_css()


# ==============================================================================
# 4. INITIALIZE SESSION STATE# 🎬 CineMatch – Movie Recommendation System Using Python

A complete, modern, and visually captivating **Content-Based Movie Recommendation System** built with **Python**, **Streamlit**, **Pandas**, and **Scikit-Learn**. Developed as a college mini project for the *Python Programming* subject.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Project Directory Structure](#-project-directory-structure)
4. [Technology Stack](#-technology-stack)
5. [How the Recommendation Algorithm Works](#-how-the-recommendation-algorithm-works)
6. [Windows Installation and Execution Guide](#-windows-installation-and-execution-guide)
7. [Testing and Verification](#-testing-and-verification)
8. [Troubleshooting Common Issues](#-troubleshooting-common-issues)
9. [Viva Voce Revision Questions & Answers](#-viva-voce-revision-questions--answers)
10. [Future Enhancements](#-future-enhancements)

---

## 🌟 Project Overview

**CineMatch** addresses the widespread challenge of "choice paralysis" in digital streaming entertainment. With an ever-growing catalog of international films, viewers often spend more time searching for content than watching it. 

CineMatch combines **Natural Language Processing (NLP)** and **Vector Geometry** to evaluate the thematic similarity between movies based on their genres, storyline overviews, and directorial vision. Wrapped inside a responsive, dark Netflix/HBO-inspired cinematic interface, the application delivers instant, high-confidence movie recommendations without needing expensive third-party APIs or external subscription services.

---

## ✨ Key Features

- **🎬 Dark Cinematic Streaming Aesthetic**: Built with custom CSS featuring Netflix Red (`#E50914`) and Cinema Gold (`#FFC837`) accents, glassmorphic cards, and hover elevation effects.
- **⚡ Content-Based AI Recommendations**: Calculates cosine similarity across combined metadata vectors (TF-IDF) to recommend thematically aligned movies with similarity percentage scores.
- **🔍 Advanced Movie Discovery & Search**: Case-insensitive search across titles, directors, and plot keywords with real-time feedback.
- **🏷️ Multi-Genre Filtering**: Filter films across Action, Sci-Fi, Drama, Thriller, Animation, Comedy, Crime, Fantasy, and more.
- **⭐ Minimum Rating & Multi-Criteria Sorting**: Sort by IMDb rating, release year (newest/oldest), or alphabetical title.
- **❤️ Interactive Watchlist (Favorites)**: Add and remove films to a personalized favorites collection persisted seamlessly across sessions via Streamlit Session State.
- **📊 Interactive Data Analytics**: Plotly-powered interactive visualizations showcasing genre distribution, IMDb rating vs. release year scatter plot, and top-ranked classics.
- **🛡️ Bulletproof Offline Fallback**: Includes an embedded SVG poster generator so the application never breaks or displays broken image icons even when offline.
- **🎓 Integrated Viva Guide**: Built-in tab containing academic project documentation, algorithmic formulas, and 15+ viva exam questions.

---

## 📂 Project Directory Structure

```text
CineMatch/
├── app.py                  # Main Streamlit web application & UI controllers
├── recommender.py          # OOP Recommendation Engine (TF-IDF & Cosine Similarity)
├── data/
│   └── movies.csv          # Curated dataset containing 36 iconic movies
├── assets/
│   └── style.css           # Custom dark cinematic streaming stylesheet
├── requirements.txt        # Python dependency specifications
└── README.md               # Complete project documentation & viva guide
```

### Purpose of Each File:
* **`app.py`**: Handles page routing, sidebar navigation, user interaction states, metric rendering, and grid layouts.
* **`recommender.py`**: Encapsulates data ingestion, metadata cleaning, tokenization, TF-IDF matrix generation, and similarity matrix ranking into a clean `MovieRecommender` class.
* **`data/movies.csv`**: Local dataset with 36 curated movies covering fields: `movie_id`, `title`, `genres`, `release_year`, `rating`, `director`, `overview`, `poster_url`.
* **`assets/style.css`**: Injected CSS providing a streaming platform UI look-and-feel.
* **`requirements.txt`**: Standard list of required Python packages for one-click installation via `pip`.
* **`README.md`**: Academic documentation, installation manual, and viva preparation resource.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming logic, data structures, and OOP |
| **Frontend UI** | Streamlit | Responsive web UI, reactive inputs, session state |
| **Data Processing** | Pandas & NumPy | Tabular data manipulation, cleaning, vector maths |
| **Machine Learning** | Scikit-Learn | `TfidfVectorizer` and `cosine_similarity` computations |
| **Visualization** | Plotly Express | Interactive charts (genre counts, ratings, timeline) |
| **Styling** | Custom CSS3 | Dark cinematic layout, responsive cards, badge styling |

---

## 🧠 How the Recommendation Algorithm Works

CineMatch uses a **Content-Based Filtering** approach. Here is the step-by-step workflow:

```text
+-----------------------+     +--------------------------+
|  data/movies.csv      | --> |  Data Cleaning (Pandas)  |
+-----------------------+     +--------------------------+
                                           |
                                           v
                              +--------------------------+
                              |   Metadata Soup Builder  |
                              | Genres*2 + Dir + Summary |
                              +--------------------------+
                                           |
                                           v
                              +--------------------------+
                              |   TfidfVectorizer (NLP)  |
                              |  Stop words removal      |
                              |  Unigrams & Bigrams      |
                              +--------------------------+
                                           |
                                           v
                              +--------------------------+
                              |  Cosine Similarity Matrix|
                              |    Dim: (N x N) Matrix   |
                              +--------------------------+
                                           |
                                           v
                              +--------------------------+
                              |  Top-N Ranking Engine    |
                              | (Excludes Target Movie)  |
                              +--------------------------+
```

### 1. Metadata Soup Construction
For every film $i$, text features are combined into a weighted string:
$$\text{Soup}_i = (\text{Genres} \times 2) + \text{Director} + \text{Overview}$$
*Note: Repeating genres doubles their mathematical significance so genre affinity is preserved.*

### 2. TF-IDF Calculation
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- **$\text{TF}(t, d)$**: Frequency of term $t$ in movie $d$.
- **$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$**: Penalizes common terms ("the", "film") and rewards unique descriptors ("superhero", "subconscious", "mafia").

### 3. Cosine Similarity Formula
Between two movie vectors $\vec{u}$ and $\vec{v}$:
$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}$$
Cosine similarity measures the angle between vectors rather than magnitude, ensuring long synopses and short synopses can be compared accurately.

---

## 💻 Windows Installation and Execution Guide

Follow these step-by-step instructions in Windows PowerShell or Command Prompt.

### Step 1: Open Terminal and Navigate to Project
```powershell
# Open Windows PowerShell or Command Prompt
cd "C:\Users\kanishka sharma\.gemini\antigravity\scratch\CineMatch"
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error in PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the command again).*

- **In Command Prompt (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Launch the Application
```powershell
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 Testing and Verification

Verify the system using these test scenarios:

| Test Case | Steps to Execute | Expected Result | Status |
| :--- | :--- | :--- | :--- |
| **1. Existing Title Search** | Type "Inception" in Discover search bar | Exactly "Inception" displays with 8.8 rating | Passed |
| **2. Non-existent Search** | Type "NonexistentXYZMovie" | "No Movies Found" friendly empty-state card appears | Passed |
| **3. Multi-Genre Filter** | Select "Sci-Fi" and "Animation" | Only films belonging to these categories are listed | Passed |
| **4. Content Similarity** | Select "The Dark Knight", click Generate | Recommends *Joker*, *The Silence of the Lambs*, *Gladiator* | Passed |
| **5. Add Favorite** | Click "❤️ Favorite" on *Interstellar* | Watchlist counter increments to 1, badge updates | Passed |
| **6. Remove Favorite** | Navigate to 'My Favorites', click "💔 Remove" | Movie disappears from list, empty message shown | Passed |
| **7. Reset Filters** | Apply filters and click "🔄 Reset Filters" | Resets all criteria to default catalog view | Passed |
| **8. Dataset Resilience** | Rename `data/movies.csv` temporarily | Clean, informative error banner explains file missing | Passed |

---

## 🔧 Troubleshooting Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'streamlit'`
**Solution:** Ensure your virtual environment is activated (`venv\Scripts\activate`) before running `pip install -r requirements.txt`.

### Issue 2: Script Execution Policy Error in PowerShell
**Message:** `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system.`  
**Solution:** Run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Issue 3: Missing Poster Images When Offline
**Solution:** CineMatch has a built-in SVG placeholder generator (`OFFLINE_POSTER_SVG`). If an image URL cannot be reached, the card immediately renders the offline cinema clapper placeholder without throwing errors.

---

## 🎓 Viva Voce Revision Questions & Answers

### 1. What is the difference between Content-Based and Collaborative Filtering?
**Answer:** Content-based filtering recommends items by analyzing intrinsic metadata (genres, keywords, directors, storyline) of items that a user previously liked. Collaborative filtering analyzes patterns of past interactions across many users ("users who liked X also bought Y") without understanding the item's content.

### 2. Why did you use TF-IDF instead of CountVectorizer?
**Answer:** CountVectorizer simply counts term frequencies, meaning generic terms like "story", "man", or "life" dominate the vector simply because they appear frequently. TF-IDF applies an inverse document frequency penalty to common words across all movies and boosts rare, highly descriptive words (e.g., "dystopian", "hobbit", "mafia").

### 3. Why is Cosine Similarity used instead of Euclidean Distance?
**Answer:** Cosine similarity evaluates the angle between vectors rather than Euclidean straight-line distance. This makes the metric independent of document length: a 20-word concise synopsis and a 100-word detailed synopsis about the same subject will still have high cosine similarity.

### 4. What is the range of Cosine Similarity?
**Answer:** In text processing where term frequencies cannot be negative, cosine similarity ranges between $0.0$ (orthogonal vectors / no words in common) and $1.0$ (identical vectors / perfect thematic alignment).

### 5. How does CineMatch prevent recommending the selected movie to itself?
**Answer:** In `recommender.py`, we sort the similarity scores in descending order and filter out the element matching the reference movie's own index (`item[0] != target_idx`).

### 6. What is Streamlit Session State and why is it necessary?
**Answer:** Streamlit operates on an execution model that reruns the entire Python script on every user interaction. Without `st.session_state`, variables would reset back to their initial values. We use `st.session_state.favorites` to persist the user's watchlist across reruns.

### 7. What is `@st.cache_resource` and why did you use it?
**Answer:** `@st.cache_resource` is a Streamlit decorator that caches global singleton objects like machine learning models and database connections. It prevents CineMatch from re-reading `movies.csv` and recalculating the TF-IDF matrix on every user click, making the app blazingly fast.

### 8. What is the "Cold Start Problem" and does CineMatch suffer from it?
**Answer:** The cold start problem occurs when a recommendation system cannot make recommendations for new items or new users due to a lack of rating history. Content-based systems do NOT suffer from the item cold start problem: as soon as a new movie is added with a genre and synopsis, it can immediately be vectorized and recommended.

### 9. What are Python's OOP principles used in this project?
**Answer:** The project uses Encapsulation by packaging data attributes (`df`, `tfidf_matrix`, `cosine_sim`) and functional operations (`recommend_similar_movies`, `filter_movies`, `search_movies`) into the `MovieRecommender` class.

### 10. How are missing values handled in the dataset?
**Answer:** Using Pandas `.fillna()` methods and `pd.to_numeric(..., errors='coerce')` in `recommender.py`. Missing overviews are set to empty strings, missing ratings are defaulted safely, and missing years are assigned 2000 to prevent runtime crashes.

---

## 🔮 Future Enhancements

1. **Hybrid Filtering Engine**: Integrate LightFM or Surprise library to combine content similarity with real user ratings.
2. **REST API Integration**: Connect to The Movie Database (TMDb) live API for real-time trending releases and video trailers.
3. **User Authentication**: Implement user login and register flows backed by SQLite or MongoDB to store user watchlists permanently in the cloud.
4. **Deep Semantic Embeddings**: Upgrade TF-IDF to Transformer-based sentence embeddings (such as `sentence-transformers/all-MiniLM-L6-v2`) to capture context and semantic analogies.

---

## 📜 License & Acknowledgments
Developed for academic assessment and learning purposes in the Python Programming curriculum.
Dataset movie metadata and poster placeholders curated for demonstration.

# ==============================================================================

if "favorites" not in st.session_state:
    st.session_state.favorites = set()


@st.cache_resource(
    show_spinner="Loading CineMatch Recommendation Engine..."
)
def get_recommender() -> MovieRecommender:
    """Load the movie recommendation engine."""

    dataset_file = os.path.join(
        os.path.dirname(__file__),
        "data",
        "movies.csv"
    )

    engine = MovieRecommender(dataset_file)

    return engine


recommender = get_recommender()


# ==============================================================================
# 5. OFFLINE POSTER PLACEHOLDER & LOCAL POSTER LOADER
# ==============================================================================

_RAW_FALLBACK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
<rect width="300" height="450" fill="#141A29"/>
<circle cx="150" cy="180" r="60" fill="#1F293D"/>
<text x="50%" y="190" font-family="Arial,sans-serif" font-size="44" fill="#E50914" text-anchor="middle">🎬</text>
<text x="50%" y="275" font-family="Arial,sans-serif" font-weight="bold" font-size="18" fill="#FFFFFF" text-anchor="middle">CineMatch</text>
<text x="50%" y="305" font-family="Arial,sans-serif" font-size="13" fill="#9CA3AF" text-anchor="middle">Poster Preview</text>
</svg>"""

OFFLINE_POSTER_SVG = f"data:image/svg+xml;base64,{base64.b64encode(_RAW_FALLBACK_SVG.encode()).decode()}"

@st.cache_data
def get_movie_poster(movie_id: int, remote_url: str) -> str:
    """Returns local poster as base64 if present, else remote URL or fallback."""
    local_path = os.path.join(os.path.dirname(__file__), "assets", "posters", f"{movie_id}.jpg")
    if os.path.exists(local_path):
        try:
            with open(local_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            pass
    if remote_url and str(remote_url).startswith(("http://", "https://")):
        return str(remote_url).strip()
    return OFFLINE_POSTER_SVG


# ==============================================================================
# 6. HERO BANNER
# ==============================================================================

def render_hero_banner():
    """Display the main CineMatch banner."""

    banner_html = """
    <div class="cinematch-hero">
        <h1 class="hero-title">CINEMATCH</h1>

        <p class="hero-tagline">
            Discover your next cinematic obsession.
            Powered by TF-IDF Natural Language Processing
            and Cosine Similarity, CineMatch pairs your
            tastes with world-class cinema.
        </p>
    </div>
    """

    st.markdown(
        clean_html(banner_html),
        unsafe_allow_html=True
    )


# ==============================================================================
# 7. MOVIE CARD
# ==============================================================================

def render_movie_card(
    movie_row,
    key_suffix: str = "",
    show_similarity: bool = False
):
    """Display a movie card with details and favorite buttons."""

    movie_id = int(movie_row["movie_id"])
    title = str(movie_row["title"])
    genres = str(movie_row["genres"])
    release_year = int(movie_row["release_year"])
    rating = float(movie_row["rating"])

    director = str(
        movie_row.get("director", "Unknown")
    )

    # Clean movie description
    overview = str(
        movie_row.get(
            "overview",
            "No synopsis available."
        )
    )

    # Remove HTML tags and decode HTML entities
    overview = html.unescape(
        re.sub(r"<[^>]*>", " ", overview)
    )

    # Correct whitespace handling
    overview = re.sub(r"\s+", " ", overview).strip()

    # Replace invalid descriptions
    if (
        not overview
        or "class=" in overview.lower()
        or overview.lower() in {"nan", "none"}
    ):
        overview = "No synopsis available."

    # Escape text before inserting it into HTML
    overview = html.escape(
        overview,
        quote=True
    )

    poster_raw = str(movie_row.get("poster_url", "")).strip()
    poster_src = get_movie_poster(movie_id, poster_raw)

    # Prepare genre labels
    genre_list = [
        genre.strip()
        for genre in genres.split(",")
    ][:3]

    genre_html = "".join(
        f'<span class="genre-chip">{html.escape(g)}</span>'
        for g in genre_list
    )

    # Similarity badge
    similarity_html = ""

    if (
        show_similarity
        and "similarity_score" in movie_row
    ):
        score = movie_row["similarity_score"]

        similarity_html = (
            f'<div class="similarity-badge">'
            f'🎯 {score}% Match'
            f'</div>'
        )

    # Movie card HTML
    card_html = f"""
    <div class="movie-card">

        <div class="movie-poster-container">

            <img
                class="movie-poster-img"
                src="{poster_src}"
                alt="{html.escape(title, quote=True)} Poster"
                onerror="this.onerror=null; this.src='{OFFLINE_POSTER_SVG}';"
            />

            <div class="poster-rating-badge">
                ★ {rating}
            </div>

            <div class="poster-year-badge">
                {release_year}
            </div>

        </div>

        <div class="movie-info">

            {similarity_html}

            <div class="movie-title">
                {html.escape(title)}
            </div>

            <div class="movie-director">
                Dir. {html.escape(director)}
            </div>

            <div class="genre-tags">
                {genre_html}
            </div>

            <div class="movie-overview">
                {overview}
            </div>

        </div>

    </div>
    """

    st.markdown(
        clean_html(card_html),
        unsafe_allow_html=True
    )

    # Add/remove favorites
    is_favorite = (
        movie_id in st.session_state.favorites
    )

    if is_favorite:

        if st.button(
            "💔 Remove Favorite",
            key=f"fav_btn_{movie_id}_{key_suffix}",
            use_container_width=True
        ):
            st.session_state.favorites.remove(
                movie_id
            )

            st.rerun()

    else:

        if st.button(
            "❤️ Add to Favorites",
            key=f"fav_btn_{movie_id}_{key_suffix}",
            use_container_width=True
        ):
            st.session_state.favorites.add(
                movie_id
            )

            st.rerun()

    # Movie details expander
    with st.expander("ℹ️ Details & Plot"):

        st.markdown(
            f"**Directed by:** {director}"
        )

        st.markdown(
            f"**Release Year:** {release_year}"
        )

        st.markdown(
            f"**IMDb Rating:** ⭐ {rating} / 10.0"
        )

        st.markdown(
            f"**Genres:** `{genres}`"
        )

        if (
            show_similarity
            and "similarity_score" in movie_row
        ):
            st.markdown(
                "**Similarity Match:** "
                f"{movie_row['similarity_score']}%"
            )

        st.markdown("**Overview:**")

        # Display cleaned description as plain text
        st.write(
            html.unescape(overview)
        )


# ==============================================================================
# 8. MOVIE GRID
# ==============================================================================

def render_movie_grid(
    df: pd.DataFrame,
    key_prefix: str = "grid",
    cols_per_row: int = 4,
    show_similarity: bool = False
):
    """Display movie cards in a responsive grid."""

    if df.empty:

        empty_html = """
        <div class="empty-state-box">

            <div class="empty-state-icon">
                🎬
            </div>

            <h3>No Movies Found</h3>

            <p>
                Try adjusting your search query,
                genre selection, or rating threshold.
            </p>

        </div>
        """

        st.markdown(
            clean_html(empty_html),
            unsafe_allow_html=True
        )

        return

    rows = [
        df.iloc[i:i + cols_per_row]
        for i in range(
            0,
            len(df),
            cols_per_row
        )
    ]

    for row_index, row_df in enumerate(rows):

        columns = st.columns(cols_per_row)

        for column_index, (_, movie) in enumerate(
            row_df.iterrows()
        ):

            with columns[column_index]:

                render_movie_card(
                    movie,
                    key_suffix=(
                        f"{key_prefix}_"
                        f"{row_index}_{column_index}"
                    ),
                    show_similarity=show_similarity
                )


# ==============================================================================
# 9. DATASET ERROR SAFEGUARD
# ==============================================================================

if not recommender.is_ready:

    st.error(
        "⚠️ **Dataset Error**: "
        "Unable to initialize CineMatch."
    )

    st.markdown(
        f"> **Reason:** `{recommender.error_message}`"
    )

    st.info(
        "💡 **How to fix:** Ensure that "
        "`data/movies.csv` exists with valid movie headers."
    )

    st.stop()
    # ==============================================================================
# 10. SIDEBAR NAVIGATION & STATISTICS
# ==============================================================================

sidebar_logo_html = """
<div style="text-align: center; padding: 0.8rem 0;">

    <h2 style="
        font-family: 'Cinzel', serif;
        color: #E50914;
        margin: 0;
        letter-spacing: 2px;
    ">
        CINEMATCH
    </h2>

    <span style="
        font-size: 0.75rem;
        color: #9CA3AF;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    ">
        College Mini Project
    </span>

</div>
"""

st.sidebar.markdown(
    clean_html(sidebar_logo_html),
    unsafe_allow_html=True
)


# Navigation menu
nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "🔍 Discover Movies",
        "🎯 Recommendations",
        f"❤️ My Favorites ({len(st.session_state.favorites)})",
        "📊 Movie Analytics",
        "ℹ️ About & Viva Guide"
    ],
    index=0
)


# Sidebar statistics
st.sidebar.markdown("---")

stats = recommender.get_stats()

st.sidebar.markdown(
    f"**📚 Catalog:** {stats['total_movies']} Titles"
)

st.sidebar.markdown(
    f"**🏷️ Genres:** {stats['total_genres']} Categories"
)

st.sidebar.markdown(
    f"**⭐ Avg Rating:** {stats['avg_rating']} / 10"
)

st.sidebar.markdown(
    f"**📅 Era:** {stats['year_span']}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Python Programming Mini Project\n"
    "Streamlit • Pandas • Scikit-Learn"
)


# ==============================================================================
# 11. HOME PAGE
# ==============================================================================

if nav_selection == "🏠 Home":

    # Main hero banner
    render_hero_banner()

    st.write("")

    # Dashboard metrics
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            clean_html(f"""
            <div class="cinematch-metric-card">
                <div class="metric-value">
                    {stats['total_movies']}
                </div>
                <div class="metric-label">
                    Curated Movies
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            clean_html(f"""
            <div class="cinematch-metric-card">
                <div class="metric-value">
                    {stats['total_genres']}
                </div>
                <div class="metric-label">
                    Unique Genres
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            clean_html(f"""
            <div class="cinematch-metric-card">
                <div class="metric-value">
                    ★ {stats['avg_rating']}
                </div>
                <div class="metric-label">
                    Average Score
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with m4:
        st.markdown(
            clean_html(f"""
            <div class="cinematch-metric-card">
                <div class="metric-value" style="color: #E50914;">
                    {len(st.session_state.favorites)}
                </div>
                <div class="metric-label">
                    Saved Favorites
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    st.write("")

    # Highest-rated movies section
    header_html1 = """
    <div class="section-header">
        <div class="section-accent-line"></div>
        <h2>🔥 Highest Rated Masterpieces</h2>
    </div>
    """

    st.markdown(
        clean_html(header_html1),
        unsafe_allow_html=True
    )

    top_movies = recommender.get_top_rated(
        top_n=8
    )

    render_movie_grid(
        top_movies,
        key_prefix="home_top",
        cols_per_row=4
    )

    st.write("")

    # Quick recommendation section
    header_html2 = """
    <div class="section-header">
        <div class="section-accent-line"></div>
        <h2>⚡ Quick Pick: What Do You Feel Like Watching?</h2>
    </div>
    """

    st.markdown(
        clean_html(header_html2),
        unsafe_allow_html=True
    )

    col_pick, col_res = st.columns([1, 2])

    with col_pick:

        all_titles = (
            recommender.get_all_movies()["title"]
            .tolist()
        )

        quick_choice = st.selectbox(
            "Select a film you love:",
            all_titles,
            index=0
        )

        st.write(
            "CineMatch analyzes movie descriptions, "
            "genres, and directors using "
            "**TF-IDF Vectorization** to find "
            "similar movies."
        )

    with col_res:

        if quick_choice:

            quick_recs = (
                recommender.recommend_similar_movies(
                    quick_choice,
                    top_n=2
                )
            )

            st.markdown(
                f"**Top Matches for '{quick_choice}':**"
            )

            render_movie_grid(
                quick_recs,
                key_prefix="home_quick",
                cols_per_row=2,
                show_similarity=True
            )


# ==============================================================================
# 12. DISCOVER & SEARCH MOVIES
# ==============================================================================

elif nav_selection == "🔍 Discover Movies":

    discover_header = """
    <div class="section-header">
        <div class="section-accent-line"></div>
        <h2>🔍 Discover & Search Catalog</h2>
    </div>
    """

    st.markdown(
        clean_html(discover_header),
        unsafe_allow_html=True
    )

    st.write(
        "Search your favorite movies and discover "
        "something new to watch."
    )

    # Search and sorting controls
    with st.container():

        f_col1, f_col2 = st.columns([2, 1])

        with f_col1:

            search_query = st.text_input(
                "🔎 Search by Title, Director, or Keyword:",
                placeholder="e.g. Inception, Nolan, Space, Heist..."
            )

        with f_col2:

            sort_choice = st.selectbox(
                "Sort Results By:",
                [
                    "Highest Rating",
                    "Release Year (Newest First)",
                    "Release Year (Oldest First)",
                    "Title (A-Z)"
                ]
            )

        # Genre, rating, and reset controls
        c1, c2, c3 = st.columns([2, 1.5, 1])

        with c1:

            all_genres = recommender.get_all_genres()

            selected_genres = st.multiselect(
                "Filter by Genres:",
                options=all_genres,
                default=[]
            )

        with c2:

            min_rating_val = st.slider(
                "Minimum Rating (IMDb):",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1
            )

        with c3:

            st.write("")
            st.write("")

            reset_clicked = st.button(
                "🔄 Reset Filters",
                use_container_width=True
            )

            if reset_clicked:
                st.rerun()

    # Determine sorting column
    sort_column = "rating"
    is_ascending = False

    if sort_choice == "Release Year (Newest First)":

        sort_column = "release_year"
        is_ascending = False

    elif sort_choice == "Release Year (Oldest First)":

        sort_column = "release_year"
        is_ascending = True

    elif sort_choice == "Title (A-Z)":

        sort_column = "title"
        is_ascending = True

    # Apply search and filters
    if search_query:

        matched_df = recommender.search_movies(
            search_query
        )

        # Filter by selected genres
        if selected_genres:

            def matches_genre(movie_genres):

                movie_genre_list = [
                    genre.strip().lower()
                    for genre in str(
                        movie_genres
                    ).split(",")
                ]

                return any(
                    genre.lower() in movie_genre_list
                    for genre in selected_genres
                )

            matched_df = matched_df[
                matched_df["genres"].apply(
                    matches_genre
                )
            ]

        # Filter by minimum rating
        matched_df = matched_df[
            matched_df["rating"] >= min_rating_val
        ]

        # Sort results
        results_df = matched_df.sort_values(
            by=sort_column,
            ascending=is_ascending
        )

    else:

        results_df = recommender.filter_movies(
            genres=selected_genres,
            min_rating=min_rating_val,
            sort_by=sort_column,
            ascending=is_ascending
        )

    # Display results
    st.markdown("---")

    st.markdown(
        f"### 🎬 Showing {len(results_df)} matching movies"
    )

    render_movie_grid(
        results_df,
        key_prefix="discover",
        cols_per_row=4
    )
    # ==============================================================================
# 13. VIEW: RECOMMENDATIONS
# ==============================================================================

elif nav_selection == "🎯 Recommendations":

    recs_header = """
    <div class="section-header">
        <div class="section-accent-line"></div>
        <h2>🎯 Content-Based Movie Recommender</h2>
    </div>
    """

    st.markdown(
        clean_html(recs_header),
        unsafe_allow_html=True
    )

    st.write(
        "Select any movie from our catalog. "
        "CineMatch uses TF-IDF Vectorization and "
        "Cosine Similarity to recommend similar movies."
    )

    st.info(
        "💡 Recommendations are generated by comparing "
        "movie genres, descriptions, and other available metadata."
    )

    # Get all movie titles
    all_titles = (
        recommender.get_all_movies()["title"]
        .tolist()
    )

    # Recommendation controls
    rec_col1, rec_col2, rec_col3 = st.columns(
        [2, 1, 1]
    )

    with rec_col1:

        selected_movie_title = st.selectbox(
            "Choose a Reference Movie:",
            all_titles,
            index=0
        )

    with rec_col2:

        top_n_count = st.slider(
            "Recommendations Count:",
            min_value=2,
            max_value=8,
            value=4,
            step=1
        )

    with rec_col3:

        st.write("")
        st.write("")

        get_rec_btn = st.button(
            "✨ Generate Recommendations",
            use_container_width=True
        )

    # Display selected movie details
    ref_movie = recommender.get_movie_by_title(
        selected_movie_title
    )

    if ref_movie:

        with st.expander(
            f"📌 Reference Film Details: {selected_movie_title}"
        ):

            c_img, c_info = st.columns([1, 3])

            with c_img:

                ref_poster = get_movie_poster(int(ref_movie["movie_id"]), ref_movie.get("poster_url", ""))
                st.image(ref_poster, width=140)

            with c_info:

                st.write(
                    f"**Year:** {ref_movie['release_year']}"
                )

                st.write(
                    f"**Rating:** ⭐ {ref_movie['rating']}"
                )

                st.write(
                    f"**Director:** "
                    f"{ref_movie.get('director', 'Unknown')}"
                )

                st.write(
                    f"**Genres:** {ref_movie['genres']}"
                )

                st.write(
                    f"**Overview:** "
                    f"{ref_movie.get('overview', '')}"
                )

    st.markdown("---")

    # Generate recommendations
    rec_df = recommender.recommend_similar_movies(
        selected_movie_title,
        top_n=top_n_count
    )

    if not rec_df.empty:

        st.markdown(
            f"### 🍿 Recommended Movies for Fans of "
            f"*{selected_movie_title}*"
        )

        st.write(
            "Here are movies with similar content "
            "according to the recommendation algorithm."
        )

        render_movie_grid(
            rec_df,
            key_prefix="ai_rec",
            cols_per_row=4,
            show_similarity=True
        )

    else:

        st.warning(
            "No recommendations could be generated "
            "for the selected title."
        )
        # ==============================================================================
# 16. VIEW: ABOUT & VIVA GUIDE
# ==============================================================================

elif nav_selection == "ℹ️ About & Viva Guide":

    about_header = """
    <div class="section-header">
        <div class="section-accent-line"></div>
        <h2>ℹ️ Project Documentation & College Viva Guide</h2>
    </div>
    """

    st.markdown(
        clean_html(about_header),
        unsafe_allow_html=True
    )

    # Documentation tabs
    doc_tab1, doc_tab2, doc_tab3, doc_tab4 = st.tabs(
        [
            "📖 Project Overview",
            "⚙️ Recommendation Algorithm",
            "🎓 Viva Voce Q&A",
            "🚀 Future Enhancements"
        ]
    )

    # --------------------------------------------------------------------------
    # TAB 1: PROJECT OVERVIEW
    # --------------------------------------------------------------------------

    with doc_tab1:

        st.markdown(
            """
            ### 📌 Project Title

            **CineMatch – Movie Recommendation System Using Python**

            ### 🎯 Problem Statement

            With thousands of movies available, viewers often
            spend considerable time deciding what to watch.

            Traditional movie listings do not always provide
            personalized recommendations.

            CineMatch addresses this problem by recommending
            movies based on their content and metadata.

            ### 🎯 Project Objectives

            - Develop a movie recommendation application.
            - Provide an interactive user interface.
            - Allow users to search for movies.
            - Filter movies by genre and rating.
            - Recommend similar movies using machine learning.
            - Allow users to maintain a favorites list.
            - Display movie analytics using interactive charts.

            ### 🛠️ Technology Stack

            - **Python:** Core programming language.
            - **Streamlit:** Interactive web application.
            - **Pandas:** Dataset processing and manipulation.
            - **Scikit-Learn:** TF-IDF and cosine similarity.
            - **Plotly Express:** Interactive data visualization.
            - **HTML and CSS:** Custom interface styling.
            - **CSV:** Movie dataset storage.

            ### ✨ Main Features

            1. Home dashboard
            2. Movie search and discovery
            3. Content-based recommendations
            4. Favorites and watchlist
            5. Interactive movie analytics
            6. Project documentation and viva guide
            """
        )

    # --------------------------------------------------------------------------
    # TAB 2: RECOMMENDATION ALGORITHM
    # --------------------------------------------------------------------------

    with doc_tab2:

        st.markdown(
            """
            ### ⚙️ Recommendation Algorithm

            CineMatch uses a content-based filtering approach
            to identify movies with similar characteristics.

            ### 1. Content-Based Filtering

            The system compares movie information such as
            genres, directors, and descriptions.

            ### 2. TF-IDF Vectorization

            TF-IDF stands for Term Frequency-Inverse Document
            Frequency.

            It converts textual information into numerical
            feature vectors.

            **Term Frequency (TF):**

            Measures how frequently a term occurs in a document.

            **Inverse Document Frequency (IDF):**

            Gives lower importance to terms appearing in many
            documents and higher importance to more distinctive
            terms.

            ### 3. Cosine Similarity

            Cosine similarity measures the similarity between
            two numerical vectors.

            Formula:

            Cosine Similarity(A, B) =
            (A · B) / (||A|| × ||B||)

            A higher similarity score indicates that two
            movie representations are more similar.

            ### 4. Recommendation Process

            1. Load the movie dataset.
            2. Process available movie metadata.
            3. Convert text into TF-IDF vectors.
            4. Calculate similarities between movies.
            5. Rank similar movies.
            6. Display the top recommendations.

            ### Advantages

            - Recommendations depend on movie content.
            - New movies can be considered when their metadata
              is available.
            - No user-rating history is required for the
              basic content-based approach.
            - Recommendations are easy to interpret using
              movie genres and descriptions.
            """
        )

    # --------------------------------------------------------------------------
    # TAB 3: VIVA QUESTIONS AND ANSWERS
    # --------------------------------------------------------------------------

    with doc_tab3:

        st.markdown(
            "### 🎓 Frequently Asked Viva Questions & Answers"
        )

        viva_qa = [

            (
                "Q1. What is CineMatch?",
                "CineMatch is a movie recommendation system "
                "developed using Python and Streamlit. "
                "It helps users discover movies similar "
                "to their interests."
            ),

            (
                "Q2. What is a recommendation system?",
                "A recommendation system suggests relevant "
                "items to users based on their preferences "
                "or item characteristics."
            ),

            (
                "Q3. What is content-based filtering?",
                "Content-based filtering recommends items "
                "by comparing their features, such as genres, "
                "descriptions, and directors."
            ),

            (
                "Q4. What is TF-IDF?",
                "TF-IDF stands for Term Frequency-Inverse "
                "Document Frequency. It converts text into "
                "numerical features and helps identify "
                "important words."
            ),

            (
                "Q5. What is cosine similarity?",
                "Cosine similarity measures the cosine of "
                "the angle between two vectors. It is used "
                "to determine how similar two movie "
                "representations are."
            ),

            (
                "Q6. Why is Python used in this project?",
                "Python provides libraries for data "
                "processing, machine learning, and web "
                "application development."
            ),

            (
                "Q7. What is Streamlit?",
                "Streamlit is a Python framework used to "
                "create interactive web applications "
                "without developing a separate frontend "
                "using JavaScript."
            ),

            (
                "Q8. What is Pandas?",
                "Pandas is a Python library used to load, "
                "manipulate, filter, and analyze structured "
                "data."
            ),

            (
                "Q9. What is a CSV file?",
                "CSV stands for Comma-Separated Values. "
                "It stores tabular data in a text format "
                "that can be loaded using Pandas."
            ),

            (
                "Q10. What is collaborative filtering?",
                "Collaborative filtering recommends items "
                "using user interactions, ratings, or "
                "preferences from multiple users."
            ),

            (
                "Q11. What is the difference between "
                "content-based and collaborative filtering?",
                "Content-based filtering compares item "
                "features. Collaborative filtering uses "
                "patterns in user behavior and interactions."
            ),

            (
                "Q12. What is a movie dataset?",
                "A movie dataset is a collection of movie "
                "records containing information such as "
                "titles, genres, ratings, and descriptions."
            ),

            (
                "Q13. Why is data preprocessing important?",
                "Data preprocessing helps handle missing "
                "values, inconsistent formats, and "
                "unwanted information before analysis."
            ),

            (
                "Q14. What is Plotly?",
                "Plotly is a visualization library used "
                "to create interactive charts and graphs."
            ),

            (
                "Q15. What is Streamlit session state?",
                "Session state stores information across "
                "Streamlit reruns during a user's session. "
                "CineMatch uses it to maintain the favorites "
                "list during that session."
            ),

            (
                "Q16. What is the cold-start problem?",
                "The cold-start problem occurs when a "
                "recommendation system lacks sufficient "
                "information about a new user or item."
            ),

            (
                "Q17. What is the purpose of the favorites feature?",
                "It allows users to save movies they like "
                "and view them in a dedicated watchlist."
            ),

            (
                "Q18. What are the limitations of CineMatch?",
                "Recommendations depend on the available "
                "dataset and metadata. The basic system "
                "does not automatically learn individual "
                "user preferences from a large user community."
            ),

            (
                "Q19. How can this project be improved?",
                "It can be extended with user authentication, "
                "a database, external movie APIs, and "
                "hybrid recommendation algorithms."
            ),

            (
                "Q20. What is the main objective of this project?",
                "The main objective is to help users discover "
                "relevant movies through an interactive "
                "application and content-based recommendations."
            )
        ]

        for question, answer in viva_qa:

            with st.expander(question):

                st.write(answer)

    # --------------------------------------------------------------------------
    # TAB 4: FUTURE ENHANCEMENTS
    # --------------------------------------------------------------------------

    with doc_tab4:

        st.markdown(
            """
            ### 🚀 Future Scope & Enhancements

            **1. User Authentication**

            Add registration and login functionality
            for individual users.

            **2. Database Integration**

            Store movie information and user favorites
            in a database such as MySQL or PostgreSQL.

            **3. External Movie APIs**

            Integrate movie APIs to retrieve additional
            information, posters, and trailers.

            **4. Hybrid Recommendation System**

            Combine content-based filtering with
            collaborative filtering.

            **5. Deep Learning**

            Explore transformer-based text embeddings
            to improve semantic movie similarity.

            **6. User Reviews and Ratings**

            Allow users to provide feedback and ratings.

            **7. Deployment**

            Deploy the application so users can access
            it through a web browser.

            **8. Improved Personalization**

            Use saved preferences and viewing history
            to provide more personalized suggestions.
            """
        )


# ==============================================================================
# 17. FOOTER
# ==============================================================================

st.markdown("---")

footer_html = """
<div style="
    text-align: center;
    color: #6B7280;
    font-size: 0.85rem;
    padding: 1rem 0;
">
    🎬 <b>CineMatch</b> —
    Python Programming Mini Project
    <br>
    Developed with Streamlit, Pandas and Scikit-Learn
</div>
"""

st.markdown(
    clean_html(footer_html),
    unsafe_allow_html=True
)