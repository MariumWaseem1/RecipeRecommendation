


import time
import numpy as np
import streamlit as st
from sklearn.metrics import jaccard_score
from helper import embed_query

def app(data):
    st.title('The Recipe Wizard 🔮')
    st.markdown("<p style='font-family: Algerian, cursive; color: #ffafc1; font-size: 28px;'>🔍 Welcome to RecipeWizard - Your Ingredient Driven Recipe Portal! 🔍</p>"
                "<p style='font-family: Algerian, cursive; color: #ff8096; font-size: 20px;'>Craving something amazing? I'll find delectable dishes for your taste buds! Let's embark on a flavor adventure! 🌟</p>"
                "<p style='font-family: Algerian, cursive; color: #ffafc1; font-size: 20px;'>Ready to be amazed? Let's get cooking! 👩‍🍳👨‍🍳</p>",
                unsafe_allow_html=True)
    st.markdown('Made by using Jaccard Similarity')

    st.image('images/Recipewizard.png')

    available_items = st.multiselect(
        label='🔮 Discover recipes with the ingredients you have! 🔮',
        options=data.columns[2:],
    )
    
    count = st.slider(
        label='🔮 How many recipes do you want the Recipe Wizard to display?🔮',
        min_value=1,
        max_value=15,
        value=7,
        step=1
    )
    
    submit = st.button('Submit')
    
    if submit:
        if not available_items:
            st.subheader('The Recipe Wizard wants you to enter at least 2 ingredients!')
        else:
            with st.spinner('Magic in progress..🔮'):
                time.sleep(2)
                emb_qy = embed_query(available_items, data.columns[2:].values)
                emb_qy = emb_qy.reshape(1, -1)
                emb_recipes = data.iloc[:, 2:].values
                similarities = []
                for i in range(emb_recipes.shape[0]):
                    similarity = jaccard_score(emb_qy.ravel(), emb_recipes[i].ravel())
                    similarities.append(similarity)
                
                similarities = np.array(similarities)
                idx_sorted = np.argsort(similarities)[::-1]
                
                st.markdown("<h2 style='font-family: Arial; color: #ffafc1; font-size: 28px;'>✨ Unveiling Extraordinary Recipes - Prepare to be Amazed! ✨</h2>",
                            unsafe_allow_html=True)
                for val, idx in zip(similarities[idx_sorted], idx_sorted):
                    if count and 0 < val < 1.0:
                        st.info(f'**{data.iloc[int(idx), 0]}**')
                        st.write("Ingredients:")
                        ingredients = data.iloc[int(idx), 1]
                        for ingredient in ingredients:
                            st.write(f"- {ingredient}")
                        count -= 1
