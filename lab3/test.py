import numpy as np
import matplotlib.pyplot as plt

dx = 2
dt = 0.1
x = np.arange(0, 10 + dx, dx)
t = np.arange(0, 10 + dt, dt)
boundcond = [0, 10]
initcond = dx * x

n = len(x)
m = len(t)
T = np.zeros((n, m))

T[0, :] = boundcond[0]
T[-1, :] = boundcond[1]
T[:, 0] = initcond

factor = dt / dx**2
A = np.diag([1 + 2 * factor] * (n - 2), 0) + np.diag([-factor] * (n - 3), -1) + np.diag([-factor] * (n - 3), 1)

for j in range(1, m):
    B = T[1:-1, j - 1].copy()
    B[0] = B[0] + factor * T[0, j]
    B[-1] = B[-1] + factor * T[-1, j]
    sol = np.linalg.solve(A, B)
    print(sol)

r = np.linspace(1, 0, m)
b = np.linspace(0, 1, m)
g = 0

for j in range(m):
    plt.plot(x, T[:, j], color=[r[j], g, b[j]])

plt.xlabel("temperature in celcius")
plt.ylabel("time in seconds")
plt.legend([f't = {value} s' for value in t])
plt.show()