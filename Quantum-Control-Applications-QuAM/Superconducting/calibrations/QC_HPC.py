import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("TKAgg")
# Set global font size
plt.rcParams.update({'font.size': 15.7})

x_data = [1, 3, 10, 30, 100]
q_data = [0.1, 6.1, 11.2, 25.8, 75.9] 
c_data = [1.3, 2.4, 6.3, 17.5, 58.8]

plt.figure()
plt.suptitle("qubit frequencies")
plt.xlabel("qubit.name")
plt.ylabel("qubit.xy.frequency (GHz)")
plt.plot(x_data, q_data, marker='o', color='blue')
plt.plot(x_data, c_data, marker='o', color='red')

plt.show()