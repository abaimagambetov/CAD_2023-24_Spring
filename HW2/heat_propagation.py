import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constant parameters
k = 1000  # heat transfer coefficient
c = 1500  # capacity
p = 2000  # density

l_x = 1.0       # Length of the rod in x-direction
l_y = 1.0       # Length of the rod in y-direction
l_t = 1.0        # Total time

# number of grids
N_x = 100
N_y = 100
N_t = 100

# Discretization
dx = l_x / (N_x - 1)  # Spatial step in x
dy = l_y / (N_y - 1)  # Spatial step in y
dt = l_t / N_t  # Spatial step in t

# Initialize grid
x_vals = np.linspace(0, l_x, N_x)
y_vals = np.linspace(0, l_y, N_y)

# Creation of a 3D array to show temperature distribution
vals = np.zeros((N_x, N_y, N_t + 1))

# Initialize with initial condition
for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        vals[i, j, 0] = 230.0

# Boundary conditions
def boundary_conditions(u, t):
    # Fixed temperature at all boundaries
    u[:, 0, t] = 225.0
    u[:, -1, t] = 0
    u[0, :, t] = 225.0
    u[-1, :, t] = 0


# Main loop
for n in range(1, N_t + 1):
    # Apply boundary conditions
    boundary_conditions(vals, n)

    # Update the solution using finite difference scheme
    for i in range(1, N_x - 1):
        for j in range(1, N_y - 1):
            # The equation below is the expansion around the (T_i,j)^n from the manual solution
            vals[i, j, n] = -1 * vals[i, j, n - 1]
            - k / (c * p) * (dt / dy**2 * (vals[i, j + 1, n - 1] - 2 * vals[i, j, n - 1] - vals[i, j - 1, n - 1])
            + dt / dx**2 * (vals[i + 1, j, n - 1] - 2 * vals[i, j, n - 1] + vals[i - 1, j, n - 1]))

# Plot the final solution
X, Y = np.meshgrid(x_vals, y_vals)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, vals[:, :, -1].T, cmap='viridis')

ax.set_title('Heat Propagation with Discretization and Periodic Boundary Conditions')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Temperature')

plt.show()