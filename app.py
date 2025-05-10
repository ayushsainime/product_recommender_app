import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#Data Loading & Preprocessing
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep='\t')
        # seleet and rename colmn 
        mapping = {
            'Uniq Id': 'id',
            'Product Id': 'prod_id',
            'Product Rating': 'rating',
            'Product Reviews Count': 'review_count',
            'Product Category': 'category',
            'Product Brand': 'brand',
            'Product Name': 'name',
            'Product Image Url': 'image_url',
            'Product Description': 'description',
            'Product Tags': 'raw_tags'
        }
        df = df[list(mapping.keys())]
        df.rename(columns=mapping, inplace=True)
        #  cleaning 
        df['id'] = df['id'].str.extract(r'(\d+)') 
        df['prod_id'] = df['prod_id'].str.extract(r'(\d+)') 
        # filling missing vallue s
        df['rating'].fillna(0, inplace=True)
        df['review_count'].fillna(0, inplace=True)
        df[['category', 'brand', 'description', 'raw_tags']] = df[['category', 'brand', 'description', 'raw_tags']].fillna('')
        #  combinee all the tags in one  column for easy access  . 
        df['tags'] = df[['category', 'brand', 'description', 'raw_tags']].apply(lambda row: ', '.join([x for x in row if x]), axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Recommendation Functions 
@st.cache_data
def rating_based(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    try:
        avg_ratings = (
            df
            .groupby(['name', 'review_count', 'brand', 'image_url'])['rating']
            .mean()
            .reset_index()
            .sort_values('rating', ascending=False)
        )
        avg_ratings['rating'] = avg_ratings['rating'].astype(int)
        avg_ratings['review_count'] = avg_ratings['review_count'].astype(int)
        return avg_ratings.head(top_n)
    except Exception as e:
        st.error(f"Error in rating-based recommendations: {e}")
        return pd.DataFrame()

@st.cache_data
def content_based(df: pd.DataFrame, item_name: str, top_n: int) -> pd.DataFrame:
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['tags'])
        sim_matrix = cosine_similarity(tfidf_matrix)
        indices = pd.Series(df.index, index=df['name']).drop_duplicates()
        if item_name not in indices:
            return pd.DataFrame()
        idx = indices[item_name]
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        rec_indices = [i for i, _ in sim_scores]
        cols = ['name', 'review_count', 'brand', 'image_url', 'rating']
        return df.iloc[rec_indices][cols]
    except Exception as e:
        st.error(f"Error in content-based recommendations: {e}")
        return pd.DataFrame()

@st.cache_data
def collaborative(df: pd.DataFrame, user_id: int, top_n: int) -> pd.DataFrame:
    try:
        user_item = df.pivot_table(index='id', columns='prod_id', values='rating', aggfunc='mean').fillna(0).astype(int)
        sim = cosine_similarity(user_item)
        sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)
        if user_id not in sim_df.index:
            return pd.DataFrame()
        scores = sim_df[user_id].sort_values(ascending=False)[1:]
        recs = []
        for u in scores.index:
            unseen = (user_item.loc[u] == 0) & (user_item.loc[user_id] == 0)
            recs.extend(user_item.columns[unseen][:top_n])
        cols = ['name', 'review_count', 'brand', 'image_url', 'rating']
        return df[df['prod_id'].isin(recs)][cols].drop_duplicates().head(top_n)
    except Exception as e:
        st.error(f"Error in collaborative filtering: {e}")
        return pd.DataFrame()

@st.cache_data
def hybrid(df: pd.DataFrame, user_id: int, item_name: str, top_n: int) -> pd.DataFrame:
    try:
        cb = content_based(df, item_name, top_n)
        cf = collaborative(df, user_id, top_n)
        combined = pd.concat([cb, cf]).drop_duplicates()
        combined['score'] = combined['rating'] * 0.7 + combined['review_count'] * 0.3  # Weighted scoring
        return combined.sort_values('score', ascending=False).head(top_n)
    except Exception as e:
        st.error(f"Error in hybrid recommendations: {e}")
        return pd.DataFrame()

@st.cache_data
def filter_by_rating(df: pd.DataFrame, selected_rating: float) -> pd.DataFrame:
    try:
        filtered = df[df['rating'] == selected_rating]
        return filtered[['name', 'review_count', 'brand', 'image_url', 'rating']].drop_duplicates()
    except Exception as e:
        st.error(f"Error filtering by rating: {e}")
        return pd.DataFrame()

# Streamlit UI 

st.set_page_config(page_title='Product Recommender', layout='wide')
st.title(':) Product Recommendation System')

# Load data
data_path = 'marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv'
df = load_data(data_path)

st.sidebar.header('Settings')
mode = st.sidebar.radio('Choose Recommendation Type', ['Rating-Based', 'Content-Based', 'USER BASED', 'Hybrid', 'Filter by Rating'])
top_n = st.sidebar.slider('Number of Recommendations', 1, 20, 5)

# Function to display images in a grid
def display_images_in_grid(results):
    if not results.empty:
        cols_per_row = 5  # Number of columns in a row
        rows = (len(results) + cols_per_row - 1) // cols_per_row
        for i in range(rows):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i * cols_per_row + j
                if idx < len(results):
                    row = results.iloc[idx]
                    with col:
                        st.image(row['image_url'], use_container_width=True)  # Updated parameter
                        st.caption(f"{row['name']} (Brand: {row['brand']})")
                        st.write(f"⭐ {row['rating']} | Reviews: {row['review_count']}")

if mode == 'Rating-Based':
    st.subheader('Top Rated Products')
    res = rating_based(df, top_n)
    if not res.empty:
        display_images_in_grid(res)
elif mode == 'Content-Based':
    st.subheader('Content-Based Recommendations')
    prod = st.selectbox('Select Product', df['name'].unique())
    res = content_based(df, prod, top_n)
    if res.empty:
        st.warning('No recommendations found or product not in dataset')
    else:
        display_images_in_grid(res)
elif mode == 'USER BASED':
    st.subheader('USER BASED Filtering Recommendations')
    uid = st.selectbox('Select User ID', df['id'].unique())
    res = collaborative(df, uid, top_n)
    if  res.empty:
        st.warning('No recommendations for this user')
    else:
        display_images_in_grid(res)
elif mode == 'Hybrid':
    st.subheader('Hybrid Recommendations')
    uid = st.selectbox('Select User ID', df['id'].unique(), key='hy_uid')
    prod = st.selectbox('Select Product', df['name'].unique(), key='hy_prod')
    res = hybrid(df, uid, prod, top_n)
    if res.empty:

        st.warning('No hybrid recommendations available')
    else:
        display_images_in_grid(res)
elif mode == 'Filter by Rating':
    st.subheader('Filter Products by Rating')
    selected_rating = st.slider('Select Rating', 0.0, 5.0, 0.0, step=0.5)
    res = filter_by_rating(df, selected_rating)
    if res.empty:
        st.warning(f'No products found with a rating of {selected_rating}')
    else:
        display_images_in_grid(res)

st.sidebar.write('---')
st.sidebar.caption(' MADE BY AYUSH SAINI . NIT TRICHY . ')