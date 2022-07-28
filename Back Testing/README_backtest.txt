For the ARIMA.ipynb you can change in the 6th cell the order variables for backtesting various ARIMA models
	trading costs are set to tc=0.000059 you can change that to 0 if you want to see the results without considering
	trading costs

For Bollinger_Backtest.ipynb, SMA_Backtest.ipynb and Contrarian_Backtest the logic is the same. You load teh model with 
	some random variables in cell3.
	In cell 3 you you the range of the variables you want to backtest and find the optimal
	In cell 4 you can Out of sample test with the optimal variables you founf in the previous cells

For ML_Forest_LogReg_Backtest.ipynb you load the model in cell 2 along with the trading costs you want the results to see.
	Tests are run the the next cell for optimal lags
	After finging the optimal lags you can run the test again applying the optimal number of lags and the visualise.
	You can do this for Logistic regression first and Random forest after.
	Log reg is run for 21 lags while random forest only for 10 because random forest is too heavy and requires 
	a lot of time to complete

For trader1_backtest.ipynb in cell 8 you load the ARIMA and can change the variables or order, in cell 9 you load the logistic regression
	and lags are set to 5 but can be changed
	In cell 12 you can set the ranged for the contrarian bollinger and SMA are the run the tests in cell 12
	This backtests takes very long time to finish.

For trader2_backtest.ipynb you can just run the backtest and in cell 7 you can change the lags that are set to 5
	This file exists in the main folder and not in Back Testing folder as you need to have the folder DNN_model_3,
	Logistic_regression_model.sav, Random_Forest_3.sav and params.pkl and in order to save some file size we 
	did not pack those models in both folders