import pandas as pd
import random
import category_encoders as ce
import joblib

# ==============================================================
# DATA PREPARATION

df = pd.read_csv('dataset.csv')
# print(df.isnull().sum())

df = df.dropna(subset=['genre', 'country', 'actors', 'director'])

dfFilm = df.drop(['filmtv_ID', 'director', 'actors', 'votes', 'country'], axis= 'columns')
dfFilm = dfFilm[dfFilm['year'] > 2000]
dfFilm = dfFilm[dfFilm['main_country'] == 'United States']
dfFilm = dfFilm.drop(['main_country'], axis= 'columns')
dfFilm = dfFilm.reset_index()

# print(dfFilm.head())
# print(len(dfFilm))

dfFilm['genre'] = dfFilm['genre'].replace('Mélo', 'Melodrama')
dfFilm['genre'] = dfFilm['genre'].replace('Sperimental', 'Experimental')

# print(dfFilm['genre'].unique())

dfFilm.to_csv('dfFilm.csv')

dfX = dfFilm.drop(['index', 'film_title', 'year'], axis= 'columns')
dfY = dfFilm['film_title']

# ==============================================================
# MODELLING

# create dummy by genre
ohe = ce.OneHotEncoder(use_cat_names= True, handle_unknown= 'ignore')
dfX = ohe.fit_transform(dfX)

print(dfX.columns)

# model
from sklearn.tree import DecisionTreeClassifier
modelDT = DecisionTreeClassifier()
modelDT.fit(dfX, dfY)
print(modelDT.score(dfX, dfY))
# print(modelDT.predict([dfX.iloc[15]])[0])

# test model
film_input = modelDT.predict([[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100,7]])[0]
print(film_input)

print(dfFilm[dfFilm['film_title'] == film_input])
index_input = dfFilm[dfFilm['film_title'] == film_input].index.values[0]
print(index_input)

# ==============================================================
# RECOMMENDATION SYSTEM

# content-based + cosine similarities
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer= lambda x : x.split(' '),
    analyzer= 'word'
)

matrix = model.fit_transform(dfFilm['genre'])
feature = model.get_feature_names()
# print(feature)
# print(len(feature))

from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrix)

all_films = list(enumerate(score[index_input]))

rec_films = sorted(
    all_films,
    key= lambda i : i[1],
    reverse= True
)

similar_films = []
for i in rec_films:
    if i[1] > 0.5:
        similar_films.append(i)

random_rec = random.choices(similar_films, k= 3)
# print(random_rec)
for i in random_rec:
    print(dfFilm.iloc[i[0]]['film_title'])


joblib.dump(modelDT, 'model')