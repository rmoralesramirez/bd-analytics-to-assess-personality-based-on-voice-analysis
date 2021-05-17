# IMPORTS
import pandas as pd
import glob
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns

# PERSONALITY DATA 2019
# Obtaining a list of the ids
dataunfiltered_2019 = pd.read_csv(r"...\TFG\AÑO PASADO\datos_def\ids_equivalence.csv")
data1_2019 = dataunfiltered_2019[dataunfiltered_2019["genero"] == 1]  # Women's ids
idslist = data1_2019["id_rodrigo"].to_list()

# Obtaining personality data from experts
data2_2019 = pd.read_excel(r"...\TFG\AÑO PASADO\datos_def\personalidad_2019.xlsx")
list1_as_set_2019 = set(data1_2019["id_rodrigo"])
intersection_2019 = list1_as_set_2019.intersection(data2_2019["id_rodrigo"])
ids_2019 = list(intersection_2019)
all_data_2019 = data2_2019[data2_2019["id_rodrigo"].isin(ids_2019)]

# PERSONALITY FEATURES 2019
# Obtaining all audio features
featuresunfiltered_2019 = pd.read_excel(r"...\TFG\AÑO PASADO\datos_def\audio_2019.xlsx")
all_features_2019 = featuresunfiltered_2019[
    featuresunfiltered_2019["id_rodrigo"].isin(ids_2019)
]

# PERSONALITY FEATURES 2020
path = r"...\TFG\SCRIPTS\NuevosDatos2020PersonalityDetection"

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
all_features_2020 = pd.concat(li, axis=0, ignore_index=True)

# PERSONALITY DATA 2020
all_data_2020 = pd.read_excel(
    r"...\TFG\SCRIPTS\NuevosDatos2020Personalidad\PuntuacionesPersonalidad.xlsx"
)

# FILTERING
personality_treat = "RESP"

# 2019
filter_col2_2019 = [
    col for col in all_data_2019 if col.startswith("Exp1_" + personality_treat)
]
data_2019 = all_data_2019[filter_col2_2019].reset_index().drop("index", axis=1)
features_2019 = all_features_2019

# 2020
features_2020 = all_features_2020

filter_col2_2020 = [
    col for col in all_data_2020 if col.startswith("total" + personality_treat + "auto")
]
data_2020 = all_data_2020[filter_col2_2020]

# 2020 DATAFRAME CREATION
min_max_scaler = preprocessing.MinMaxScaler()  # Normalization

df_2020 = pd.DataFrame(features_2020)
df_2020["total" + personality_treat + "auto"] = data_2020
df_2020 = df_2020.drop("name", axis=1)

# Normalization
df_2020_normalized = min_max_scaler.fit_transform(df_2020.values)

df_2020 = pd.DataFrame(df_2020_normalized)

# Dropping the last column to be the independent variable
df_2020 = df_2020.rename(columns={len(df_2020.columns) - 1: "personality"})

x_2020 = df_2020.drop("personality", axis=1) * 100
y_2020 = df_2020.personality * 100

# 2019 DATAFRAME CREATION

df_2019 = pd.DataFrame(features_2019)
df_2019["Exp1_" + personality_treat] = data_2019
df_2019 = df_2019.drop("name", axis=1)
df_2019 = df_2019.drop("id_rodrigo", axis=1)

# Normalization
df_2020_normalized = min_max_scaler.fit_transform(df_2020.values)

df_2020 = pd.DataFrame(df_2020_normalized)

# Dropping the last column to be the independent variable
df_2020 = df_2020.rename(columns={len(df_2020.columns) - 1: "personality"})

x_2020 = df_2020.drop("personality", axis=1) * 100
y_2020 = df_2020.personality * 100

# 2019 DATAFRAME CREATION

df_2019 = pd.DataFrame(features_2019)
df_2019["Exp1_CORD"] = data_2019
df_2019 = df_2019.drop("name", axis=1)
df_2019 = df_2019.drop("id_rodrigo", axis=1)

# Normalization
df_2019_normalized = min_max_scaler.fit_transform(df_2019.values)

df_2019 = pd.DataFrame(df_2019_normalized)

# Dropping the last column to be the independent variable
df_2019 = df_2019.rename(columns={len(df_2020.columns) - 1: "personality"})

x_2019 = df_2019.drop("personality", axis=1) * 100
y_2019 = df_2019.personality * 100

# SPLIT INTO TRAINING MODEL

x_train = x_2019.fillna(0).astype("int")
x_test = x_2020.fillna(0).astype("int")
y_train = y_2019.fillna(0).astype("int")
y_test = y_2020.fillna(0).astype("int")

# LOGISTIC REGRESSION

logisticRegr = LogisticRegression(max_iter=10000000)
logisticRegr.fit(x_train, y_train)

y_pred = logisticRegr.predict(x_test)

# PERCENTIL DECLARATION

low_scorers_real = 0
medium_scorers_real = 0
high_scorers_real = 0

low_scorers_pred = 0
medium_scorers_pred = 0
high_scorers_pred = 0

percentil25 = np.percentile(y_test, 25)
percentil75 = np.percentile(y_test, 75)

y_len = len(y_test)

# PREDICTIONS' PERCENTILES CALCULATION
i1 = 0

d1 = [None] * y_len

while i1 < y_len:
    if y_pred[i1] <= percentil25:
        d1[i1] = 0
    elif y_pred[i1] <= percentil75:
        d1[i1] = 1
    else:
        d1[i1] = 2
    i1 += 1
# TEST' PERCENTILES CALCULATION
i2 = 0

d2 = [None] * y_len

while i2 < y_len:
    if y_test[i2] <= percentil25:
        d2[i2] = 0
    elif y_test[i2] <= percentil75:
        d2[i2] = 1
    else:
        d2[i2] = 2
    i2 += 1
# STATISTICS CALCULATIONS
cf_matrix = confusion_matrix(d1, d2)

score = metrics.accuracy_score(d2, d1)
print(score)

rms = mean_squared_error(d2, d1, squared=False)
print(rms)

# PLOTS
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


sns.heatmap(cf_matrix, annot=True, cmap="Blues")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
