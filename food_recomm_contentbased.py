

import pandas as pd
import time
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

def calculate_coverage(generated_recipes, total_recipes):
    unique_recipes = set(generated_recipes)
    coverage = len(unique_recipes) / total_recipes
    return coverage

def calculate_diversity(generated_recipes):
    unique_recipes = set(generated_recipes)
    diversity = len(unique_recipes) / len(generated_recipes)
    return diversity

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

    st.markdown('Made by using Content-Based Filtering')

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
                
                # Content-based filtering
                selected_indices = []
                for q in fav_food:
                    food_idx = np.where(data['Name'].values == q)[0][0]
                    selected_indices.append(food_idx)

                embeddings = data.iloc[selected_indices, 2:].values
                mean_embedding = np.mean(embeddings, axis=0)

                similarities = cosine_similarity(data.iloc[:, 2:].values, [mean_embedding])

                indices = np.argsort(similarities, axis=0)[::-1].squeeze()

                # List to store generated recipe names
                generated_recipes = []

                st.header('Discover these wizard-approved recipes that might captivate your taste buds! 🔮')
                for idx in indices:
                    if count == 0:
                        break

                    if idx not in selected_indices:
                        recipe_name = data.iloc[idx, 0]
                        recipe_ingredients = data.iloc[idx, 1]
                        
                        st.subheader(recipe_name)
                        st.write("Ingredients:")
                        for ingredient in recipe_ingredients:
                            st.write(f"- {ingredient}")
                        
                        generated_recipes.append(recipe_name)
                        count -= 1

                # Calculate coverage and diversity
                total_recipes = data.shape[0]
                coverage = calculate_coverage(generated_recipes, total_recipes)
                diversity = calculate_diversity(generated_recipes)

                # Display coverage and diversity
                st.write(f"Coverage: {coverage:.2f}")
                st.write(f"Diversity: {diversity:.2f}")


# Load the dataset
data = pd.read_csv('data/updated_recipes.csv')

# Run the Streamlit app
app(data)
