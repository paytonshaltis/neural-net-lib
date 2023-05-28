from matplotlib import pyplot as plt
import numpy as np

# Read in the data from errors.txt.
errors = []
with open('errors1.txt', 'r') as f:
  for line in f:
    errors.append(float(line))

# Plot epochs vs. success rate.
for i in range(len(errors)):
  errors[i] = 1 - errors[i]
plt.plot(np.arange(len(errors)), errors)

# Success threshold.
plt.plot(np.arange(len(errors)), [0.995] * len(errors))

# Mark where the success threshold was first reached.
for i in range(len(errors)):
  if errors[i] >= 0.995:
    plt.plot([i], [errors[i]], marker='o', markersize=3, color="red")
    break

# Mark the epoch where the success threshold was first reached.
plt.plot([i, i], [0, errors[i]], linestyle='--', color="red")

plt.xlabel('Epoch')
plt.ylabel('Success Rate')
plt.xlim(0, len(errors))
plt.ylim(0.9 - errors[0], 1)
plt.title('Error vs. Epoch')
plt.show()