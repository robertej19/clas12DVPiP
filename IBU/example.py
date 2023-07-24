#From CHATGPT

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# True distribution
true_values = np.random.normal(loc=0, scale=1, size=10000)

# Detector response smearing
smearing = np.random.normal(loc=0, scale=0.2, size=10000)
measured_values = true_values + smearing

# Histogram parameters
bins = np.linspace(-3, 3, 100)

# Histogram of true and measured values
true_hist, _ = np.histogram(true_values, bins=bins)
measured_hist, _ = np.histogram(measured_values, bins=bins)

# Response matrix: estimated by dividing smeared distribution by the true one
# Add small constant to avoid division by zero
response_matrix = np.divide(measured_hist + 1e-9, true_hist + 1e-9)

# Bayesian unfolding
unfolded_hist = np.divide(measured_hist, response_matrix)

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.bar(bins[:-1], measured_hist-true_hist, width=np.diff(bins), align="edge", alpha=0.7, label='True')
#ax1.bar(bins[:-1], measured_hist, width=np.diff(bins), align="edge", alpha=0.7, label='Measured')
ax1.set_title('True and Measured Distributions')
ax1.legend()

ax2.plot(bins[:-1], response_matrix)
ax2.set_title('Response Matrix')

ax3.bar(bins[:-1], unfolded_hist-true_hist, width=np.diff(bins), align="edge", alpha=0.7, label='Unfolded')
#ax3.bar(bins[:-1], true_hist, width=np.diff(bins), align="edge", alpha=0.7, label='True')
ax3.set_title('Unfolded and True Distributions')
ax3.legend()

plt.tight_layout()
plt.show()
