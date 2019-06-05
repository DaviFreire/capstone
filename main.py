import sys
import pandas as pd
import numpy as np
import csv
import os
import yahoo_finance_pynterface as yahoo
import matplotlib.pyplot        as plt
import matplotlib.dates         as mdates
import matplotlib.ticker        as mticker
from agent import DDPG
from task import Task

def get_prices(ticker, date):
    return yahoo.Get.Prices(ticker, period=date)

prices = get_prices('AAPL', ['2018-04-01','2019-04-30']) 

print(prices.head())

sys.exit()

first = prices['2017-09-01']
state = first
prices = prices.iloc[1:,]

size = len(prices)
task = Task(state, '2018-08-31') 
agent = DDPG(task) 
labels = ['episode', 'total_rewards']
results = {x : [] for x in labels}

os.remove("reward.txt")
with open('reward.txt', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    for i in range(1000):
        day = 0
        score = 0
        state = first
        for date in prices.index:
            date = date.strftime('%Y-%m-%d')
            stock = prices[date]
            action = agent.act(state) 
            next_state, reward, done = task.step(action,stock)
            agent.step(action, reward, next_state, done)
            state = next_state
            score =+ reward
            day += 1
            if day == size:
                to_write = [day] + [score]
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)

                print(score)
            
            
        sys.stdout.flush()
