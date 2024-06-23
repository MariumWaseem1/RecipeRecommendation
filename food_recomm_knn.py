

import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors

def calculate_diversity(generated_recipes):
    unique_recipes = set(generated_recipes)
    diversity = len(unique_recipes) / len(generated_recipes)
    return diversity

def app(data):
    st.title('The Recipe Wizard 🔮')

    st.markdown("<p style='font-family: Algerian, cursive; color: #FFC0CB; font-size: 28px;'>🔍 Unveil Your Favorite Recipe - Your Similar Recipe Portal!🔍 </p>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-family: Algerian, cursive; color: #FF69B4; font-size: 20px;'>Select the recipe you love the most, and behold as the Recipe Wizard conjures up a collection of similar culinary wonders just for you!</p>", unsafe_allow_html=True)
    
    st.markdown('Made by using the KNN model')

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
                
                embeddings = []
                for q in fav_food:
                    food = data.loc[data['Name'] == q].values[0][2:]
                    embeddings.append(food)

                embeddings = np.logical_or.reduce(embeddings)

                # KNN model implementation
                nn_model = NearestNeighbors(n_neighbors=data.shape[0], algorithm='brute', metric='cosine')
                nn_model.fit(data.iloc[:, 2:].values.reshape(data.shape[0], -1))

                _, indices = nn_model.kneighbors(embeddings.reshape(1, -1))

                # List to store generated recipe names
                generated_recipes = []

                st.header('Discover these wizard-approved recipes that might captivate your taste buds! 🔮')
                for idx in indices[0]:
                    if count == 0:
                        break

                    if data.iloc[idx, 0] not in fav_food and data.iloc[idx, 0] not in generated_recipes:
                        recipe_name = data.iloc[idx, 0]
                        recipe_ingredients = data.iloc[idx, 1]
                        
                        st.subheader(recipe_name)
                        st.write("Ingredients:")
                        for ingredient in recipe_ingredients:
                            st.write(f"- {ingredient}")
                        
                        generated_recipes.append(recipe_name)
                        count -= 1

                # Calculate coverage
                coverage = len(set(generated_recipes)) / data.shape[0]

                # Calculate diversity
                diversity = calculate_diversity(generated_recipes)

                # Display coverage and diversity
                st.write(f"Coverage: {coverage:.2f}")
                st.write(f"Diversity: {diversity:.2f}")


# Load the dataset
data = pd.read_csv('data/updated_recipes.csv')

# Run the Streamlit app
app(data)
