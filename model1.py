import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pandas.plotting import register_matplotlib_converters
from pygal_maps_world.maps import World
import category_encoders as ce
import joblib

register_matplotlib_converters()

# =======================================
# creating initial dataframe
dfMatch = pd.read_csv(
    'results.csv',
    index_col= 'date',
    parse_dates= ['date']
)

# =======================================
# identifying nan values

# print(df.isnull().sum())
dfMatch = dfMatch.dropna()
dfMatch = dfMatch.drop(['city', 'country', 'neutral'], axis= 'columns')

# =======================================
# plotting top 30 match winners
# dfWins = pd.DataFrame(dfMatch['winner'].value_counts())
# dfWins = dfWins.drop(['Draw'], axis= 'index')

# dfWinPlot = dfWins[:30]

# plt.title('Top 30 Football Match Winners of All Time')
# sn.barplot(
#     x= dfWinPlot['winner'],
#     y= dfWinPlot.index,
#     data= dfWinPlot
# )

# plt.xticks(rotation= 90)
# plt.show()

# =======================================
# plotting match winners distribution


# =======================================
# modelling

dfX = dfMatch.drop(['winner'], axis= 'columns')
ohe = ce.OneHotEncoder(use_cat_names= True, handle_unknown= 'ignore')
dfX = ohe.fit_transform(dfX)

dfY = dfMatch['winner']


from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    dfX,
    dfY,
    test_size= 0.4
)

from sklearn.linear_model import LogisticRegression
modelLog = LogisticRegression(solver= 'liblinear', multi_class= 'auto')
modelLog.fit(xtr, ytr)
print(modelLog.score(xts, yts))

# from sklearn.ensemble import RandomForestClassifier
# modelRF = RandomForestClassifier(n_estimators=20)
# modelRF.fit(xtr, ytr)
# print(modelRF.score(xts, yts))