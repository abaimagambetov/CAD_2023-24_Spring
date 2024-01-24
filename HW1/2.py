import matplotlib.pyplot as plt
import numpy as np

def plot_functions(start, end, step_size, functions, labels=None):
    x_values = np.arange(start, end + step_size, step_size)

    plt.figure(figsize=(8, 6))

    for i, func in enumerate(functions):
        y_values = func(x_values)
        label = labels[i] if labels else f'Function {i + 1}'
        plt.plot(x_values, y_values, label=label)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Functions')
    plt.legend()
    plt.grid(True)
    plt.show()


# Problem 2
def function(x):
    return np.sin(x + 1)

def derivative_of_function(x):
    return np.cos(x + 1)

def limit_derivative_of_function(x):
    for i in range(20):
        return (np.sin((i + 1)*x + 1) - np.sin(i*x + 1)) / x

def error_function(x):
    for i in range(20):
        return np.cos(x + 1) - (np.sin((i + 1)*x + 1) - np.sin(i*x + 1)) / x


functions_to_plot = [function, derivative_of_function, limit_derivative_of_function, error_function]
function_labels = ['Function', 'Derivative of the Function', 'Limit Derivative of the Function', 'Error of Function']

step = float(input("Choose your step size: "))

plot_functions(-10, 10, step, functions_to_plot, function_labels)