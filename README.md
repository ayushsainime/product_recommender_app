# 🛍️ Product Recommendation System

An end-to-end **Python** and **Machine Learning** project that suggests products to users based on ratings, product content, and peer preferences. Deployed as an interactive web app using **Streamlit**.

---

## 📖 Overview

This system offers four complementary recommendation modes:

1. **Rating-Based**  
   Ranks products by their average review score (“What’s rated highest?”).

2. **Content-Based**  
   Uses TF-IDF vectorization of product tags (category, brand, description) and cosine similarity to find “products like this one.”

3. **User-Based**  
   Builds a user–item ratings matrix, finds users with similar tastes via cosine similarity, and recommends items they enjoyed.

4. **Hybrid**  
   Merges content- and user-based results for balanced suggestions: “Items like this” + “Items people like you loved.”

---

## 📦 Dataset

- **Source:** Walmart Product Review Dataset  
- **Size:** ~5,000 reviews (Jul–Dec 2020)  
- **Download:**  
  [https://www.kaggle.com/datasets/promptcloud/walmart-product-review-dataset](https://www.kaggle.com/datasets/promptcloud/walmart-product-review-dataset)  

---

## 🧰 Tech Stack & Libraries

- **Language:** Python 3.x  
- **Data:** pandas, NumPy  
- **ML:** scikit-learn (TF-IDF, cosine similarity)  
- **NLP (optional):** spaCy for text cleaning  
- **Deployment:** Streamlit  
- **Caching:** joblib / Streamlit cache  

---


![Screenshot 2025-05-10 163456](https://github.com/user-attachments/assets/2193739d-430a-40f5-b180-96fb10d69214)
![Screenshot 2025-05-10 163621](https://github.com/user-attachments/assets/5cb00fbd-0ac3-465c-96a0-2d2255865b81)


## 🚀 Quick Start

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/product-recommender-app.git
   cd product-recommender-app
