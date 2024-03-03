import numpy as np
from scipy.optimize import fsolve

def equation_to_solve(x):
    return np.exp(-(x**2) / (1.96 * 10**(-14))) + 0.014 * x


initial_guess = 0.1  # Adjustable

# Solve the equation
solution = fsolve(equation_to_solve, initial_guess)

print("The solution is:", solution[0])


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
# def equation_to_solve(x):
#     return np.exp(-(x**2) / (1.96 * 10**(-14))) + 0.014 * x
# x_values = np.linspace(-1, 1, 1000)
# y_values = equation_to_solve(x_values)
# plt.plot(x_values, y_values)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Plot of the function')
# plt.grid(True)
# plt.show()
# sign_changes = np.where(np.diff(np.sign(y_values)))[0]
# initial = [x_values[idx] for idx in sign_changes]
# # Solve the equation for each initial guess
# solutions = []
# for initial_guess in initial:
#     solution = fsolve(equation_to_solve, initial)
#     solutions.append(solution[0])
# print("Initial guesses based on sign change:", initial)
# print("Solutions:", solutions)



