import numpy as np
data=np.load('0004.npy')
print(data[0])

import matplotlib.pyplot as plt
plt.imshow(data[0],interpolation='nearest')
plt.show()