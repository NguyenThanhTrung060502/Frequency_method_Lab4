import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2zpk, lsim, freqs_zpk


pi = np.pi

# Vẽ hàm số 
def plot_func(X, func, color, title, legend):
    ymin = min(func) 
    ymax = max(func) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func.real, color=f'{color}')
    plt.xlabel('t')
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


# So sánh hàm số 
def cmp_func(X, funcs, title, legend=['Source function', 'Restored function'], labels=['t']):
    ymin = min([min(func) for func in funcs])
    ymax = max([max(func) for func in funcs])

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 
    colors = ['black', 'red', 'cyan']
    plt.ylim(ymin, ymax)
    for i in range(len(funcs)):
        plt.plot(X, funcs[i].real, color=f'{colors[i]}')
        plt.xlabel(labels[0])
        plt.legend(legend, loc='upper right')
        plt.title(title)
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


# Vẽ biến đổi Fourier của hàm số f(t) 
def plot_image(X, func, title):
    ymin = min(func.real.min(), func.imag.min())
    ymax = max(func.real.max(), func.imag.max())

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func.real, color='seagreen')
    plt.plot(X, func.imag, color='tomato')
    plt.xlabel('\u03C9')
    # plt.ylabel('f(\u03C9)')
    plt.legend(['Real', 'Imag'], loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


def plot_freq_response(X, fr, title, legend=['Frequency response'], scale='linear'):
    ymin = min(fr) 
    ymax = max(fr) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.xscale(scale)
    plt.yscale(scale)
    plt.plot(X, fr.real)
    plt.xlabel('\u03C9')
    plt.ylabel('|W(i\u03C9)|')
    if scale == 'linear':
        plt.ylim(ymin, ymax)
        plt.xlim(min(X), max(X))
        difference_array = np.absolute(fr - 1 / np.sqrt(2))
        index = difference_array.argmin()
        plt.plot(np.linspace(min(X), X[index], 100), [1 / np.sqrt(2)] * 100, 'r--', linewidth=1)
        plt.plot([X[index]] * 100, np.linspace(0, fr[index], 100), 'r--', linewidth=1)
    plt.legend(legend + ['1 / \u221A2'], loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


# Tích vô hướng      
def dotProduct(X, f, g):
    dx = X[1] - X[0]
    return np.dot(f, g) * dx

get_fourier_image = lambda X, V, func: np.array([1 / (np.sqrt(2 * np.pi)) * dotProduct(X, func, (lambda t: np.e ** (-1j * i * t))(X)) for i in V])
get_fourier_function = lambda X, V, image: np.array([1 / (np.sqrt(2 * np.pi)) * dotProduct(V, image, (lambda t: np.e ** (1j * x * t))(V)) for x in X])

get_func = lambda a, t1, t2: np.vectorize(lambda t: a if t1 <= t <= t2 else 0, otypes=[complex])
noised = lambda X, func, b, c, d: func + b * (np.random.rand(X.size) - 0.5) + c * np.sin(d * X)

differentiate = lambda X, func: np.array([0 if i == 0 else (func[i] - func[i - 1]) / (X[i] - X[i - 1]) for i in range(len(func))])



get_order_filter = lambda T1, T2, T3: tf2zpk([T1 ** 2, 2 * T1, 1], [T2 * T3, T2 + T3, 1])



def filtering(a, b, c, d, t1, t2, filter):

    t_arr = np.linspace(0, 8, 8888)

    # Cоздать волновую функцию
    wave_func = get_func(a, t1, t2)(t_arr)
    noised_wave_func = noised(t_arr, wave_func, b, c, d)

    plot_func(t_arr, wave_func, color='black', title='Исходная функция', legend=['g(t)'])
    plot_func(t_arr, noised_wave_func, color='black', title='Зашумлённый (исходный) сигнал', legend=['u(t)'])

    t_filtered, noised_wave_func_filtered,_ = lsim(filter, noised_wave_func, t_arr)
    
    plot_func(t_filtered, noised_wave_func_filtered, color='seagreen', title='Зашумлённый (исходный) сигнал после фильтра', legend=['u*(t)'])
    cmp_func(t_arr, funcs=[wave_func, noised_wave_func_filtered], title='Сравнение исходной функции и фильтрованного сигнала', legend=['g(t)', 'u*(t)'])

    # Образ сигнала и фильтрованного сигнала
    v_arr = np.linspace(-8, 8, 8888)
    wave_func_image = get_fourier_image(t_arr, v_arr, wave_func)
    noised_wave_func_filtered_image = get_fourier_image(t_arr, v_arr, noised_wave_func_filtered)

    plot_image(v_arr, wave_func_image, title='Фурье-образ исходного сигнала')
    plot_image(v_arr, noised_wave_func_filtered_image, title='Фурье-образ фильтрованного сигнала')

    wave_func_image_abs = np.absolute(wave_func_image)
    noised_wave_func_filtered_image_abs = np.absolute(noised_wave_func_filtered_image)

    plot_func(v_arr, wave_func_image_abs, color='tomato', title='Модуль Фурье-образ исходного сигнала', legend=[r'$|\hat{u}(t)|$'])
    plot_func(v_arr, noised_wave_func_filtered_image_abs, color='tomato', title='Модуль Фурье-образ фильтрованного сигнала', legend=[r'$|\hat{u}*(t)|$'])

    cmp_func(v_arr, funcs=[wave_func_image_abs, noised_wave_func_filtered_image_abs], title='Сравнение модулей исходных и фильтрованных образов сигналов', legend=[r'$|\hat{u}(t)|$', r'$|\hat{u}*(t)|$'])


    # Найти АЧХ фильтра первого порядка
    z, p, k = filter
    w, h = freqs_zpk(z, p, k, worN=np.linspace(0, 15, 1000))
    plot_freq_response(w, abs(h), title='АЧХ фильтра', scale='linear')
    


## THIRD TASK

# T1 = 0.1
# T2 = 0.2
# T3 = 0.3

# filtering(a=5, b=0, c=0.5, d=100, t1=2, t2=4, filter=get_order_filter(0.1, 0.2, 0.3))

# T1 = 0.01
# T2 = 0.02
# T3 = 0.03

# filtering(a=5, b=0, c=0.5, d=100, t1=2, t2=4, filter=get_order_filter(0.01, 0.02, 0.03))


# T1 = 0.01
# T2 = 0.02
# T3 = 0.03

# filtering(a=5, b=0, c=5, d=100, t1=2, t2=4, filter=get_order_filter(0.01, 0.02, 0.03))

# T1 = 0.01
# T2 = 0.02
# T3 = 0.03

filtering(a=5, b=0, c=0.1, d=100, t1=2, t2=4, filter=get_order_filter(0.01, 0.02, 0.03))





plt.show()