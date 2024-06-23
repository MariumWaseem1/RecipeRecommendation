import time
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

def app(data):
    st.title('The Recipe Wizard 🔮')
    st.markdown(
        "<p style='font-family: Algerian, cursive; color: #FFC0CB; font-size: 28px;'>🔍 Unveil Your Favorite Recipe - Your Similar Recipe Portal!🔍 </p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-family: Algerian, cursive; color: #FF69B4; font-size: 20px;'>Select the recipe you love the most, and behold as the Recipe Wizard conjures up a collection of similar culinary wonders just for you!</p>",
        unsafe_allow_html=True
    )

    st.markdown('Made by using the Matrix Factorization model')

    st.image('images/Recipewizard.png', use_column_width=True)

    fav_food = st.multiselect(
        label='🔮 Select your favorite enchanted recipe! 🔮',
        options=data.loc[:, 'Name'].values,
    )

    count = st.slider(
        label='How many recipes do you want the RecipeWizard to display? 🔮',
        min_value=1,
        max_value=15,
        value=7,
        step=1
    )

    submit = st.button('Submit')

    if submit:
        if not fav_food:
            st.subheader('The RecipeWizard wants you to select at least one food item')
        else:
            with st.spinner('Magic in progress! 🔮'):
                time.sleep(2)

                # Preprocess the ingredient data and remove NaN values
                data['IngredientsString'] = data['RecipeIngredientParts'].apply(lambda x: ', '.join(x))
                data = data.dropna(subset=['IngredientsString'])

                vectorizer = TfidfVectorizer()
                ingredient_vectors = vectorizer.fit_transform(data['IngredientsString'].values)

                # Apply matrix factorization on the ingredient vectors
                mf_model = NMF(n_components=10, init='nndsvd', random_state=42)
                embeddings = mf_model.fit_transform(ingredient_vectors)

                # Normalize embeddings
                normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                fav_indices = []
                for q in fav_food:
                    food_idx = np.where(data['Name'].values == q)[0][0]
                    fav_indices.append(food_idx)

                fav_embeddings = normalized_embeddings[fav_indices]
                mean_embedding = np.mean(fav_embeddings, axis=0)

                # Check for NaN values in normalized_embeddings
                mask = np.isnan(normalized_embeddings)
                normalized_embeddings[mask] = 0.0

                similarities = cosine_similarity(normalized_embeddings, [mean_embedding])

                indices = np.argsort(similarities, axis=0)[::-1].squeeze()

                st.header('Discover these wizard-approved recipes that might captivate your taste buds! 🔮')
                for idx in indices:
                    if count == 0:
                        break

                    if idx not in fav_indices:
                        recipe_name = data.iloc[idx, 0]
                        recipe_ingredients = data.iloc[idx, 1]

                        st.subheader(recipe_name)
                        st.write("Ingredients:")
                        for ingredient in recipe_ingredients:
                            st.write(f"- {ingredient}")

                        count -= 1



