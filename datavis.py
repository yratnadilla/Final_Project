import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv('dataset.csv')

# print(df.head())
# print(df.isnull().sum())
# print(len(df))

df = df.dropna(subset=['genre', 'country', 'actors', 'director'])
# print(df.isnull().sum())
# print(len(df))

# print(df.head())
dfFilm = df.drop(['filmtv_ID', 'director', 'actors', 'votes', 'country'], axis= 'columns')
dfFilm = dfFilm[dfFilm['year'] > 2000]
# print(dfFilm.head())
# print(dfFilm['genre'].value_counts())
# print(dfFilm['main_country'].value_counts())

dfGenre = pd.DataFrame(dfFilm['genre'].value_counts())
dfCountry = pd.DataFrame(dfFilm['main_country'].value_counts())

# print(dfGenre)
# print(dfCountry)

# plt.subplot(121)
plt.figure('Film Count by Genre', figsize= (10,5))
sn.barplot(
    x= dfGenre['genre'],
    y= dfGenre.index,
)

plt.savefig('Genre.jpg')

# plt.subplot(122)
plt.figure('Film Count by Country', figsize= (10,5))
sn.barplot(
    x= dfCountry['main_country'][:30],
    y= dfCountry.index[:30],
)

plt.savefig('Country.jpg')
# plt.show()