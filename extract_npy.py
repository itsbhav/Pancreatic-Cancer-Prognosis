import numpy as np
data=np.load('0004.npy')
print(np.info(data[0]))

# import matplotlib.pyplot as plt
# for i in data:
#     plt.imshow(i,interpolation='nearest')
#     plt.show()