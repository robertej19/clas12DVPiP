from scipy.integrate import quad
import numpy as np

def f(x):
    return np.sin(x)  # Replace with your function

c,d = 0,1

a, b = 0, np.pi/2  # x-range of the rectangle
result, error = quad(f, a, b)

print(result)


result1, error1 = quad(f, a, c)  # from the left boundary to the intersection
result2, error2 = quad(f, d, b)  # from the intersection to the right boundary

area = (c - a) + result1 + result2

print(area)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define your function
def f(x):
    return np.sin(x)  # replace with your function

# Rectangle boundaries
a, b = 0, np.pi/2  # x-range
c, d = 0, 1  # y-range

# Intersection points
x_int1 = 0.5  # replace with your intersection point
x_int2 = np.pi/4  # replace with your intersection point

# Plot function and rectangle
x = np.linspace(a, b, 400)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, 'r', label='y=f(x)')  # plot the function
ax.fill_between(x, y, color='gray', alpha=0.5)  # shade under the curve
ax.fill_betweenx([c, d], a, b, color='blue', alpha=0.2)  # plot the rectangle

# Shade the intersection
mask = (x > x_int1) & (x < x_int2)
ax.fill_between(x[mask], y[mask], c, color='green', alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
