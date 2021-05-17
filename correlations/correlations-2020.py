# IMPORTS
import pandas as pd
import glob
from scipy.stats.stats import pearsonr
import zipfile

# EXTRA FUNCTIONS
# Function to iterate through columns
def getcolumn(matrix, col):
    columna = []
    for row in matrix:
        columna.append(row[col])
    return columna


# PERSONALITY FEATURES
path = r"...\TFG\SCRIPTS\NuevosDatos2020PersonalityDetection"

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)
all_features = pd.concat(li, axis=0, ignore_index=True)
features = all_features

# PERSONALITY DATA
data = pd.read_excel(
    r"...\TFG\SCRIPTS\NuevosDatos2020Personalidad\PuntuacionesPersonalidad.xlsx"
)

# VARIABLES

m_data = data.values
m_features = features.values

c_data = data.columns
c_features = features.columns

l_data = len(m_data[0])
l_features = len(m_features[0])

# CORRELATIONS

correlations = {}
i, j = 3, 1

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

with zipfile.ZipFile("NEW-correlations.zip", "w") as csv_zip:
    csv_zip.writestr("NEW-all-correlations.csv", result.sort_index().to_csv())
    csv_zip.writestr(
        "NEW-potential-correlations.csv", potentialcorrelations.sort_index().to_csv()
    )
