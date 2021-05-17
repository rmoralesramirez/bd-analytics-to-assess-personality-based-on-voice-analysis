# Correlations code

Use of pandas and scipy libraries in order to obtain dataframes and search for possible correlations when comparing personality trait ratings and speech features. 
There is also a function declared to iterate through dataframe columns, and the results are finally exported to a csv file.

There was data from two datasets, one from 2019 and another from 2020, from which only women were selected to perform this analysis as it seems like gender produces big differences 
in both features and personality.

Correlations from each year were extracted in different files but then only the consistent ones were compared and selected for both years data, being measured with the Pearson Correlation Coefficient (PCC), which measures the linear relationship between two datasets, and its associated p-value, which indicates the level of statistical significance.
