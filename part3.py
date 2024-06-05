import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from scipy.signal import tf2zpk, lsim, freqs_zpk


# Vẽ hàm số 
def plot_func(X, func, color, title, legend, labels):
    ymin = min(func) 
    ymax = max(func) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func.real, color=f'{color}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


# So sánh hàm số 
def cmp_func(X, funcs, title, legend=['Source function', 'Restored function'], labels=['t']):
    ymin = min([min(func) for func in funcs])
    ymax = max([max(func) for func in funcs])

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(12, 7)) 
    colors = ['black', 'red', 'cyan']
    plt.ylim(ymin, ymax)
    for i in range(len(funcs)):
        plt.plot(X, funcs[i].real, color=f'{colors[i]}')
        plt.xlabel(labels[0])
        plt.legend(legend, loc='upper right')
        plt.title(title)
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


get_order_filter = lambda T: tf2zpk([0, 1], [T, 1]) 


def filter_quotes(file_path, T):
    file = open(file_path)
    header = file.readline()

    timeClose = []
    CLOSE = []
    line = file.readline()
    while line != "":
        line = line.split(';')
        timeClose.append(line[2])
        CLOSE.append(line[8])
        line = file.readline()

    date_array = []
    price_array = []
    for i in range(len(timeClose)):
        # convert date from mmddyy to timestamp
        timestamp = datetime.datetime.strptime(timeClose[i].strip(), '%m/%d/%Y')
        date_array = np.append(date_array, timestamp)
        price_array = np.append(price_array, float(CLOSE[i]))       

    t_arr = np.linspace(0, len(price_array), len(price_array))
    # plot_func(t_arr, price_array, color='black', title='ETH/USDT', legend=['ETHUSDT'], labels=['', ''])

    filter = get_order_filter(T)
    filtered_price = lsim(filter, price_array, t_arr, X0=price_array[0] * T)[1]
    # plot_func(date_array, filtered_price, color='black', title='Фильтрованной сигнал ETH/USDT', legend=['ETHUSDT'], labels=['Date', 'Price'])
    cmp_func(date_array, funcs=[price_array, filtered_price], title='Сравнение исходных и фильтрованных цитат сигналов', legend=['Исходный сигнал', 'Фильтрованной сигнал'], labels=['Дата', 'Цена'])




filter_quotes(file_path='ETH.csv', T=1)
filter_quotes(file_path='ETH.csv', T=7)
filter_quotes(file_path='ETH.csv', T=30)
filter_quotes(file_path='ETH.csv', T=90)
filter_quotes(file_path='ETH.csv', T=356)

plt.show()