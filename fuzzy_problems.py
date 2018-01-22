import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_ac = np.arange(0, 22990, 1)
x_wa = np.arange(0, 14537, 1)
x_dif = np.arange(0,3,1)

dif_lo = fuzz.trimf(x_dif, [0,0,1])
dif_med = fuzz.trimf(x_dif, [0,1,2])
dif_hi = fuzz.trimf(x_dif, [1,2,2])

fig, ax = plt.subplots(nrows=1)
ax.plot(x_dif, dif_lo, 'b', linewidth=1.5, label='Easy')
ax.plot(x_dif, dif_med, 'g', linewidth=1.5, label='Medium')
ax.plot(x_dif, dif_hi, 'r', linewidth=1.5, label='Hard')
ax.set_xlabel('Difficulty class')
ax.set_ylabel('Degree of membership')
ax.legend()
ax.set_title('Problem Difficulty')
plt.show()
