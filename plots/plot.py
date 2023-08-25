from matplotlib import pyplot as plt
import numpy as np

# # Read in the data from errors.txt.
# errors = []
# with open('errors1.txt', 'r') as f:
#   for line in f:
#     errors.append(float(line))

# Read in the data from errors.json
import json
with open('errors.json', 'r') as f:
  errors = json.load(f)

# Plot epochs vs. success rate.
# for i in range(len(errors)):
#   errors[i] = 1 - errors[i]
plt.plot(np.arange(len(errors)), errors)

# Success threshold.
plt.plot(np.arange(len(errors)), [25] * len(errors))

# Mark where the success threshold was first reached.
for i in range(len(errors)):
  if errors[i] <= 25:
    plt.plot([i], [errors[i]], marker='o', markersize=3, color="red")
    break

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.xlim(0, len(errors))
plt.ylim(0, 500)
plt.title('Error vs. Epoch')
plt.show()