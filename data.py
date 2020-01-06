import numpy as np

def loadcsv(filename, n):
    data = np.loadtxt(filename, usecols=(1,))
    data = data[::-1]
    x = np.zeros([data.shape[0] - n - 1, n])
    y = np.zeros([data.shape[0] - n - 1, 1])
    for i in range(n):
        x[:, i] = data[i:data.shape[0] - n - 1 + i].copy()
    y = data[n + 1:].copy()
    x -= data.mean()
    y -= data.mean()
    return x, y, data.mean()

if __name__ == "__main__":
    x, y, m = loadcsv('TRD_Exchange.csv', 5)
    from matplotlib import pyplot as plt
    plt.plot(y+m)
    plt.xlabel('time points')
    plt.ylabel('Yuan')
    plt.show()
