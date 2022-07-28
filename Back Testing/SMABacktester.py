import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class SMABacktester():

    def __init__(self, symbol, SMA_S, SMA_L, start, end, tc, granularity):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = str(start)
        self.gran = granularity
        self.end = end
        self.results = None
        self.tc = tc
        
        self.get_data()
        self.prepare_data()
    
    #Auto to func apla pairnei dedomena apo ena .csv kai tous kanei SMA
    def get_data(self):
        raw = pd.read_csv("EUR_USD.csv", parse_dates = ["time"], index_col = "time", usecols=[0,1])
        raw = raw[self.symbol].to_frame().dropna() 
        raw = raw.loc[f'{self.start} 00:00:00': f'{self.end} 23:59:00'].copy()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw = raw.resample(self.gran, label='right').last().ffill().iloc[:-1]
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
        return raw
    
    def prepare_data(self):
        data = self.data.copy()
        data["SMA_S"] = data["price"].rolling(self.SMA_S).mean() # add short sma
        data["SMA_L"] = data["price"].rolling(self.SMA_L).mean()
        self.data = data

    
    #Auto to function to ftiaxnoume mono kai mono gia na allazei to SMA_S kai SMA_L entos tou class
    #gia na mporoun na ginoun ta diafora backstest strats
    def set_parameters(self, SMA_S = None, SMA_L = None):
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()
    
    # #to strategy fainetai kai sto allo notebook, vale f string sta results 
    def test_strategy(self):
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)

        # vriskw arithmo trades
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy - data.trades * self.tc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance
        outperf = perf - data["creturns"].iloc[-1] # outperformance 
        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | SMA_S = {} | SMA_L = {}".format(self.symbol, self.SMA_S, self.SMA_L)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))  

    def optimize_parameters(self, SMA_S_range, SMA_L_range): 
        combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))
        
        # test all combinations
        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])
        
        best_perf = np.max(results) # best performance
        opt = combinations[np.argmax(results)] # optimal parameters
        
        # run/set the optimal strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()
        
        # create a df with many results
        many_results =  pd.DataFrame(data = combinations, columns = ["SMA_S", "SMA_L"])
        many_results["performance"] = results
        self.results_overview = many_results
                            
        return opt, best_perf