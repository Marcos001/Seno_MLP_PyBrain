"""
requesito: Dados de entrada-saida conhecidos
procedimento: projeto da camada de entrada
projeto da camada escondida
projeto da camada de saida

input: linear
hiden layer: glaucinana
ouput: backpropagation

algoritmo KNN*, normalization*, 
"""


import numpy as np
from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim)]#for i in xrange(numCenters)]
        self.beta = 20
        self.W = random.random((self.numCenters, self.outdim))
        #print(self.W)

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)

    def _calcAct(self, X):
        # Ativation das RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y): #X: matriz de dimention n x indim #y: vetor coluna de dimention n x 1
        # escolher vetores de centro aleatorio do conjunto de treinamento
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        print(rnd_idx)
        self.centers = [X[i,:] for i in rnd_idx]
        #print(self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print(G)
        # matriz de pesos (pseudo-inversa)
        self.W = dot(pinv(G), Y)
        #print(self.W)

    def test(self, X): #X: matriz de dimensao n x indim
        G = self._calcAct(X)
        #print(G)
        Y = dot(G, self.W)
        #print(Y)
        return Y

if __name__ == '__main__':
    n = 100
    x = mgrid[0:2*np.pi:complex(0,n)].reshape(n, 1)
    y = np.sin(x)
    # regressao RBF
    rbf = RBF(1, 50, 1)
    rbf.train(x, y)
    z = rbf.test(x)
    # funcao seno
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'k-')
    # modelo aprendido
    plt.plot(x, z, 'ro', linewidth=2)
    for c in rbf.centers:
        # predicao RBF
        cx = arange(c-0.7, c+0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='white', linewidth=0.2)
    plt.xlim(0, 6.3)
    plt.show()