# import time
# import numpy as np
# import pandas as pd
# import streamlit as st
# from sklearn.metrics.pairwise import cosine_similarity

# import food_recomm_knn
# import food_recomm_matrixfactorization
# import food_recomm_random
# import find_recipe_knn
# import find_recipe_matrixfactorization
# import find_recipe_random
# from find_recipe_matrixfactorization import MatrixFactorization

# st.set_page_config(
#     page_title="RecipeWizard",
#     page_icon=':Dome:'
# )

# @st.cache_data()
# def fetch_and_clean_data(file_path):
#     df = pd.read_csv(file_path)

#     # get all items
#     items = set()
#     for x in df.ingredients:
#         for val in x.split(', '):
#             items.add(val.lower().strip())

#     # create new dataframe
#     new_df = pd.DataFrame(data=np.zeros((256, len(items) + 2), dtype=int), columns=['name', 'ingredients'] + list(items))

#     for i, d in df.iterrows():
#         new_df.loc[i, ['name', 'ingredients']] = d[:2]

#         for val in d[1].split(', '):
#             item = val.lower().strip()
#             new_df.loc[i, item] = 1

#     return new_df

# data = fetch_and_clean_data('data/food_250.csv')

# PAGES = {
#     'Food Recommender (KNN)': food_recomm_knn,
#     'Food Recommender (Matrix Factorization)': food_recomm_matrixfactorization,
#     'Food Recommender (Random)': food_recomm_random,
#     'Recipe Finder (KNN)': find_recipe_knn,
#     'Recipe Finder (Matrix Factorization)': find_recipe_matrixfactorization,
#     'Recipe Finder (Random)': find_recipe_random
# }

# page = st.sidebar.radio(
#     label='Contents',
#     options=list(PAGES.keys())
# )


# content = PAGES[page]
# content.app(data)

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# import food_recomm_knn
# import food_recomm_matrixfactorization
# import food_recomm_contentbased
# import find_recipe_cosineSimilarity
# import find_recipe_jaccardSimilarity
# import find_recipe_random
# # from find_recipe_matrixfactorization import MatrixFactorization
# st.set_page_config(
#     page_title="RecipeWizard",
#     page_icon=':Dome:'
# )


# @st.cache_data()
# def fetch_and_clean_data(file_path, num_rows):
#     df = pd.read_csv(file_path, nrows=num_rows)

#     # Rest of the code remains the same...


#     # Extracting relevant columns from the new dataset
#     new_df = df[['Name', 'RecipeIngredientParts']]

    # Cleaning and transforming data
    # Cleaning and transforming data
            #  new_df['RecipeIngredientParts'] = new_df['RecipeIngredientParts'].apply(lambda x: [val.strip().lower() for val in str(x).split(',') if not isinstance(x, float)])

#rrrrrrrrrrrr
#     # Creating binary columns for each ingredient
#     ingredients = set()
#     for x in new_df['RecipeIngredientParts']:
#         for val in x:
#             ingredients.add(val)

#     # Creating a new dataframe with binary columns for ingredients
#     ingredient_columns = list(ingredients)
#     new_df = pd.concat([new_df, pd.DataFrame(0, index=np.arange(len(new_df)), columns=ingredient_columns)], axis=1)

#     for i, row in new_df.iterrows():
#         for val in row['RecipeIngredientParts']:
#             new_df.loc[i, val] = 1

#     return new_df

# data = fetch_and_clean_data('data/updated_recipes.csv', 1000)


# PAGES = {
#     'Food Recommender (KNN)': food_recomm_knn,
#     'Food Recommender (Matrix Factorization)': food_recomm_matrixfactorization,
#     'Food Recommender (Random)': food_recomm_random,
#     'Recipe Finder (KNN)': find_recipe_knn,
#     'Recipe Finder (Matrix Factorization)': find_recipe_matrixfactorization,
#     'Recipe Finder (Random)': find_recipe_random
# }


# page = st.sidebar.radio(
#     label='Contents',
#     options=list(PAGES.keys())
# )

# content = PAGES[page]
# content.app(data)

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import food_recomm_knn
import food_recomm_matrixfactorization
import food_recomm_contentbased
import find_recipe_cosineSimilarity
import find_recipe_jaccardSimilarity
import find_recipe_pearsoncorrelation


@st.cache_data()
def fetch_and_clean_data(file_path, num_rows):
    df = pd.read_csv(file_path, nrows=num_rows)

st.cache_data()
def fetch_and_clean_data(file_path, num_rows):
    df = pd.read_csv(file_path, nrows=num_rows)

    # Rest of the code remains the same...

    # Extracting relevant columns from the new dataset
    new_df = df[['Name', 'RecipeIngredientParts']]

    # Cleaning and transforming data
    new_df['RecipeIngredientParts'] = new_df['RecipeIngredientParts'].apply(lambda x: [val.strip().lower() for val in str(x).split(',') if not isinstance(x, float) and x is not None])

    # Creating binary columns for each ingredient
    ingredients = set()
    for x in new_df['RecipeIngredientParts']:
        for val in x:
            ingredients.add(val)

    # Creating a new dataframe with binary columns for ingredients
    ingredient_columns = list(ingredients)
    new_df = pd.concat([new_df, pd.DataFrame(0, index=np.arange(len(new_df)), columns=ingredient_columns)], axis=1)

    for i, row in new_df.iterrows():
        for val in row['RecipeIngredientParts']:
            new_df.loc[i, val] = 1

    return new_df

# data = fetch_and_clean_data('data/updated_recipes.csv', 1000)
data = fetch_and_clean_data('data/updated_recipes.csv', 1000)

PAGES = {
    'Food Recommender (KNN)': food_recomm_knn,
    'Food Recommender (Matrix Factorization)': food_recomm_matrixfactorization,
    'Food Recommender (Content based)': food_recomm_contentbased,
    'Recipe Finder (Cosine)': find_recipe_cosineSimilarity,
    'Recipe Finder (Jaccard Similarity)': find_recipe_jaccardSimilarity,
    'Recipe Finder (Pearsoncorrelation)': find_recipe_pearsoncorrelation
}

page = st.sidebar.radio(
    label='Contents',
    options=list(PAGES.keys())
)

content = PAGES[page]
content.app(data)
