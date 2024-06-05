import numpy as np
import matplotlib.pyplot as plt
import datetime

pi = np.pi

# Vẽ hàm số 
def plot_func(X, func, color, title, legend=['Source function'], labels=['t', 'f(t)']):
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
def cmp_func(X, funcs, title, legend=['Source function', 'Restored function'], labels=['t', 'f(t)']):
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
        plt.ylabel(labels[1])
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
    plt.ylabel('f(\u03C9)')
    plt.legend(['Real', 'Imag'], loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


# Tích vô hướng      
def dotProduct(X, f, g):
    dx = X[1] - X[0]
    return np.dot(f, g) * dx


get_fourier_image = lambda X, V, func: np.array([1 / (np.sqrt(2 * pi)) * dotProduct(X, func, (lambda t: np.e ** (-1j * image_clip * t))(X)) for image_clip in V])
get_fourier_function = lambda X, V, image: np.array([1 / (np.sqrt(2 * pi)) * dotProduct(V, image, (lambda t: np.e ** (1j * x * t))(V)) for x in X])

# get_func = lambda a, t1, t2: np.vectorize(lambda t: a if t1 <= t <= t2 else 0, otypes=[complex])

noised = lambda X, func, a: func + a * (np.random.rand(X.size) - 0.5) 
# найти численную производную от зашумлённого сигнала
numerical_derivative = lambda X, func: np.array([0 if i == 0 else (func[i] - func[i - 1]) / (X[i] - X[i - 1]) for i in range(len(func))])



# get_first_order_filter = lambda T: tf2zpk([0, 1], [T, 1]) 
# get_second_order_filter = lambda T1, T2, T3: tf2zpk([T1 ** 2, 2 * T1, 1], [T2 * T3, T2 + T3, 1])


def spectral_diff(T):
    t = np.linspace(-100, 100, 10000) # array with arguments for functions

    sin = np.sin(t)
    noised_sin = noised(t, sin, 0.2)

    plot_func(t, noised_sin, color='black', title='Зашумлённый сигнал', legend=['f(t) = sin(t)'])

    # найти численную производную от зашумлённого сигнала
    noised_sin_derivative = numerical_derivative(t, noised_sin)
    plot_func(t, noised_sin_derivative, color='black', title='Численная производная от зашумлённого сигнала', legend=['f \'(t)'])

    noised_sin_image = get_fourier_image(t, t, noised_sin)
    plot_image(t, noised_sin_image, title='Фурье-образ зашумленной сигнала')

    # Образ производной функции (также известное как умножение на iw)
    noised_sin_derivative_image = get_fourier_image(t, t, noised_sin) * 1j * t
    plot_image(t, noised_sin_derivative_image, title='Фурье-образ производной зашумленной сигнала')

    # восстановить функцию из образа 
    restored_noised_sin_derivative = get_fourier_function(t, t, noised_sin_derivative_image)
    plot_func(t, restored_noised_sin_derivative, color='red', title='Восстановленная функция из производного образа зашумленной сигнала', legend=['Восстановленная функция'])

    # сравнить восстановленную из образа производной с исходной производной и с функцией cos
    cos = np.cos(t)
    cmp_func(t, funcs=[noised_sin_derivative, restored_noised_sin_derivative, cos], title='Сравнение истинной производной cos(t) с численной и спектральной производной.', legend=['численная производная', 'Спектральная производная', 'cos(t)'])
    

## FIRST TASK 

spectral_diff(T=10)
# spectral_diff(T=20)

plt.show()