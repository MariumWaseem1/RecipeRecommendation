
RecipeWizard is a machine learning-based intelligent recipe recommendation system that helps users make healthier dietary choices. By analyzing user inputs and nutritional data, the system suggests suitable recipes while considering calorie intake, dietary preferences, and nutritional balance.

Features
- Personalized recipe recommendations based on user health profiles
- Nutritional filtering for calories, protein, fats, and carbohydrates
- ML models trained on a curated dataset for food and health
- Evaluation and comparison of multiple classification models
- User-friendly interface for dietary input and recipe exploration

Machine Learning Models Used
The system evaluates and compares multiple classification models for predicting the healthiness of recipes:

- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes Classifier

Performance metrics such as accuracy, F1-score, precision, and recall were used to compare models and identify the most effective one for recommendation.

Technology Stack
- Programming Language: Python
- Libraries and Frameworks:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit (optional for UI)

Dataset
- Recipes and nutritional information sourced and cleaned from Food.com and other open datasets
- Features include:
- Ingredients
- Nutritional values (calories, protein, fat, carbohydrates)
- User ratings and tags
