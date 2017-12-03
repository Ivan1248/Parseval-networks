#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()

#%%
import sys
sys.version

#%%
x = np.linspace(0, 20, 100)
plt.plot(x, np.exp(x))
plt.show()

#%%
trues = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]).astype(np.bool)
probs = np.array([[0.2, 0.2, 0.5], [1, 0, 0], [0.1, 0.8, 0.1], [0, 0.1, 0.9]])
trues.sum(0)
np.diag(probs)
probs.trace()
