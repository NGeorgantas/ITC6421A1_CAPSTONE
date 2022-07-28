Trading_Application_1.ipynb and Trading_Application_2.ipynb are the two trading bots the user only needs to run the code in python

In order fro the code to run the following files needs to be in the same directory:

params.pkl (For Trading_Application_2.ipynb )
Random_Forest_3.sav (The trained random forest model for Trading_Application_2.ipynb )
Logistic_Regression_model.sav (The trained logistic regression model for Trading_Application_2.ipynb )
Folder DNN_model_3 (The trained Neural Network model for Trading_Application_2.ipynb )

The oanda.cfg is the OANDA configuration file CONTAINS the account_id and the access_token of the user
Because this is very sensitive information the students' details have been removed from the configuration file and have been replaced with XXXXX
The user will have to set up a practise account in OANDA and provide his own account_id and access_token 
Another file provides links and details regarding how to set up a practise account in OANDA
If for some reason the user cannot set up his own OANDA account and find his account_id and access_token then we can provide our own
so that you can check the code running.

Folder Dataset and Model Development consists of the dataset that was used for thraining the ML models
as well as the code for training the machine learning models and the ARIMA model

Back Testing folder provides the code and the examples for the BackTesting of all strategies