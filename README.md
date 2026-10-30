# 🎬 CineMatch – Movie Recommendation System Using Python

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
