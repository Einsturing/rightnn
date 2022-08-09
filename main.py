import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import math
import torch
from scatterFig import figsave

generate = np.zeros((1024, 1))

for k in range(4):
    if k == 0:
        for t in range(24):
            data = np.ones((32, 32))
            data[0:8, t:t + 8] = -data[0:8, t:t + 8]
            data = data.reshape(1024, 1)
            generate = np.concatenate((generate, data), axis=1)
            print('\n')
    elif k == 1:
        for t in range(24):
            data = np.ones((32, 32))
            data[t:t + 8, 24:32] = -data[t:t + 8, 24:32]
            data = data.reshape(1024, 1)
            generate = np.concatenate((generate, data), axis=1)
    elif k == 2:
        for t in range(24):
            data = np.ones((32, 32))
            data[24:32, 24 - t:32 - t] = -data[24:32, 24 - t:32 - t]
            data = data.reshape(1024, 1)
            generate = np.concatenate((generate, data), axis=1)
    else:
        for t in range(24):
            data = np.ones((32, 32))
            data[24 - t:32 - t, 0:8] = -data[24 - t:32 - t, 0:8]
            data = data.reshape(1024, 1)
            generate = np.concatenate((generate, data), axis=1)


df = pd.DataFrame(generate)
X = pd.DataFrame(torch.Tensor([[j for j in range(32)] for i in range(32)]).view(1024).numpy())
Y = pd.DataFrame(torch.Tensor([[i for j in range(32)] for i in range(32)]).view(1024).numpy())
df.insert(0, 'y', Y)
df.insert(0, 'x', X)
for i in range(3, 99):
    print(df.columns[i])
    figsave(df, df.columns[2], df.columns[i])

df.to_csv('move_full.csv')
print('done')
