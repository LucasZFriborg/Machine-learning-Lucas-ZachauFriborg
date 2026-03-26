# 🎬 Movie Recommendation System

## 📖 Overview

This project implements a **content-based movie recommendation system** using the MovieLens dataset. The system recommends five similar movies based on an input title by analyzing genres and user-generated tags.

---

## 📦 Project Structure
├── recommend.py # Main recommendation script
├── data_exploration.ipynb
├── report.md # Detailed report and method explanation
├── README.md
└── .gitignore

---

## Dataset

The dataset is **not included in this repository** due to file size limitations.

You must download it manually from MovieLens:

👉 https://grouplens.org/datasets/movielens/

Scroll to:

**"recommended for education and development"**

Then download:

- **ml-latest.zip** (NOT the small version)

This corresponds to the **full dataset (~335 MB)**.

---

### 📁 Extract the dataset

After downloading, extract the zip file and place it like this:
Data/ml-latest/movies.csv
Data/ml-latest/tags.csv
Data/ml-latest/ratings.csv

---

### ℹ️ Notes

- The files `genome-tags.csv` and `genome-scores.csv` are included in the dataset but **are not used in this project**.
- These files come from a separate machine learning project and can be safely ignored for this assignment.

---

⚠️ Important:
- Do NOT rename the folder (`ml-latest`)
- The script expects this exact folder structure

---

## 🧠 Method

This system uses:
- Content-Based Filtering  
- TF-IDF Vectorization  
- Cosine Similarity  

Movie features are created by combining:
- Genres (from `movies.csv`)
- Tags (from `tags.csv`)

---

## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/LucasZFriborg/Machine-learning-Lucas-ZachauFriborg.git
cd Machine-learning-Lucas-ZachauFriborg

2. Create virtual environment (uv)
uv venv
source .venv/bin/activate

3. Install dependencies
```bash
pip install pandas scikit-learn

▶️ Run the program
python recommend.py

Then enter a movie title:
Enter movie title: Toy Story

The system will:
1. Suggest matching titles
2. Let you choose one
3. Output top 5 similar movies

📊 Documentation
- data_exploration.ipynb contains the exploratory data analysis
- report.md contains a detailed explanation of:
    - method choice
    - feature engineering
    - limitations
    - design decisions

⚠️ Limitations
- Does not use user ratings (no collaborative filtering)
- Performance depends on available tags
- Movies without tags rely only on genres

🚀 Future Improvements
- Incorporate ratings (collaborative filtering)
- Use clustering for more diverse recommendations
- Build a web interface (e.g. Dash)

👤 Author
Lucas Zachau Friborg