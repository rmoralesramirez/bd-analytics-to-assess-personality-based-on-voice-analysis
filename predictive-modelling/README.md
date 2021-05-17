# Predictive modelling

Use of pandas, scikit-learn, numpy, matplotlib and seaborn in order to obtain a linear, logistic regression and random forest models which predicted a personality trait based 
on specific features or on all features.

Data from 2019 was used to train the model, and 2020’s to test it. Because personality from different years was evaluated in different ways, all the ratings and feature values 
were normalized in mean and variance. Furthermore, as the purpose was to analyse the capability of estimating people’s personality in general, and not the exact value, results 
were split in three percentiles: percentile 25, percentile 75 and over percentile 75; corresponding to low scorer (0), medium scorer (1) and high scorer (2). In this way, we 
transform the regression problem into a 3-class classification problem for which we can compute the percentage of subjects correctly classified.

Plots have also been used to present the results in a clearer and more understandable way with a scatter plot, representing predictions values vs true values, and a confusion 
matrix for the 3-class classification problem. The predictions have been evaluated using two statistics: Accuracy, which is how many predictions have been correct for the 
percentile’s 3-class classification problem, and Root Mean Squared Error (RMSE), a very common metric that informs about the actual size of an error produced by a regression 
model.
