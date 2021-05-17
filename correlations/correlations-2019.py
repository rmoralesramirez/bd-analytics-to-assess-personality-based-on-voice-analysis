# IMPORTS
import pandas as pd
from scipy.stats.stats import pearsonr
import zipfile

# EXTRA FUNCTIONS
# Function to iterate through columns
def getcolumn(matrix, col):
    columna = []
    for row in matrix:
        columna.append(row[col])
    return columna


# PERSONALITY DATA
# Obtaining a list of the ids
dataunfiltered = pd.read_csv(r"...\TFG\AÑO PASADO\datos_def\ids_equivalence.csv")
data1 = dataunfiltered[dataunfiltered["genero"] == 1]  # Women's ids
idslist = data1["id_rodrigo"].to_list()

# Obtaining personality data from experts
data2 = pd.read_excel(r"...\TFG\AÑO PASADO\datos_def\personalidad_2019.xlsx")

list1_as_set = set(data1["id_rodrigo"])
intersection = list1_as_set.intersection(data2["id_rodrigo"])
ids = list(intersection)
data = data2[data2["id_rodrigo"].isin(ids)]

# PERSONALITY FEATURES
# Obtaining all audio features
featuresunfiltered = pd.read_excel(r"...\TFG\AÑO PASADO\datos_def\audio_2019.xlsx")
all_features = featuresunfiltered[featuresunfiltered["id_rodrigo"].isin(ids)]
features = all_features

# VARIABLES

m_data = data.values
m_features = features.values

c_data = data.columns
c_features = features.columns

l_data = len(m_data[0])
l_features = len(m_features[0])

# CORRELATIONS

correlations = {}
i, j = 2, 1

while i < l_data:
    col_data = getcolumn(m_data, i)
    while j < l_features:
        col_features = getcolumn(m_features, j)
        correlations[str(c_data[i]) + "__" + str(c_features[j])] = pearsonr(
            col_data, col_features
        )

        j += 1
    i += 1
    j = 1
result = pd.DataFrame.from_dict(correlations, orient="index")
result.columns = ["PCC", "p-value"]

potentialcorrelations = result.sort_index()[
    result.sort_index()["p-value"].between(0, 0.05)
]

# EXPORTATION

with zipfile.ZipFile("NEW-correlations2019.zip", "w") as csv_zip:
    csv_zip.writestr("NEW-all-correlations2019.csv", result.sort_index().to_csv())
    csv_zip.writestr(
        "NEW-potential-correlations2019.csv",
        potentialcorrelations.sort_index().to_csv(),
    )
