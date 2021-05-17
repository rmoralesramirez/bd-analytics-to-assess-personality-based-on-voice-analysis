# IMPORTS
import pandas as pd
import zipfile


# PERSONALITY DATA

data20 = pd.read_csv(
    r"...\TFG\SCRIPTS\Correlations\RESULTADOS\NEW-potential-correlations.csv"
)

data19 = pd.read_csv(
    r"...\TFG\SCRIPTS\Correlations\RESULTADOS\NEW-potential-correlations2019.csv"
)

list1_as_set = set(data20["correlation"])
intersection = list1_as_set.intersection(data19["correlation"])

features = list(intersection)
print(features)

data2020 = data20[(data20["correlation"]).isin(features)]
data2019 = data19[(data19["correlation"]).isin(features)]


with zipfile.ZipFile("last-year-COMPARATION.zip", "w") as csv_zip:
    csv_zip.writestr("coincidences-2020.csv", data2020.to_csv())
    csv_zip.writestr("coincidences-2019.csv", data2019.to_csv())
