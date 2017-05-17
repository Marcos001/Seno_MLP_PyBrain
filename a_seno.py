'''
Implementation de uma rede neural MLP para
calcular a fuction da function seno to 0 to 2pi - usando a API- pybrains
#Marcos Vinicius dos Santos Ferreira - 
'''

import matplotlib.pyplot as plt
import numpy as np
import math

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer    
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.validation import ModuleValidator



class MLP_SENO():

    def __init__(self):
        ''''''



    def ver_taxa_erro(self,lista):
        '''
        funcao para ver em grafico a taxa de erro
        :param lista: valores da taca de erro
        :return: None
        '''
        x = 10 * np.array(range(len(lista)))

        # plt.plot(x, data1, 'go')  # green bolinha
        # plt.plot(x, data1, 'k:', color='orange')  # linha pontilha orange

        plt.plot(x, lista, 'r^')  # red triangulo
        plt.plot(x, lista, 'k--', color='blue')  # linha tracejada azul

        plt.axis([-10, 60, 0, 11])
        plt.title("Taxa de Erro")

        plt.grid(True)
        plt.xlabel("eixo horizontal")
        plt.ylabel("Vertical")
        plt.show()


    def constroi_MLP(self):
        '''
        define o modelo da rede criada
        :return: lista de erro por numeros de acesso por epoca
        '''

        # Definicao da classe de rede neural
        dimensaoDaEntrada = 1
        dimensaoDaCamadaEscondida = 5
        dimensaoDaSaida = 1
        lista_erro = []

        print 'criando a rede neural > '
        rede_neural_mlp = buildNetwork(dimensaoDaEntrada,
                                       dimensaoDaCamadaEscondida,
                                       dimensaoDaSaida, bias=True,
                                       hiddenclass=TanhLayer)

        # Criacao dos dados
        tamanhoDaAmostra = 100

        dados = SupervisedDataSet(dimensaoDaEntrada,
                                  dimensaoDaSaida) #Initialize an empty supervised dataset

        print 'dados = ', dados, '\n tamanho de dados > ', len(dados), ' tipo ', type(dados)

        comRuido = True

        contador = 0
        print 'gerando dados > '
        for i in range(tamanhoDaAmostra):
            if (comRuido):
                x = np.random.uniform(0, 2 * math.pi, 1)
                print 'amostra [',str(contador), '] > valor de x = ', x , ' other = ', math.sin(x) + np.random.normal(0, 0.1, 1)
                dados.addSample((x), (math.sin(x) + np.random.normal(0, 0.1, 1),))
                """Add a new sample consisting of `input` and `target`."""
            else:
                x = np.random.uniform(0, 2 * math.pi, 1)
                dados.addSample((x), (math.sin(x),))
            contador += 1
        print 'done.'

        print 'criando o aprendizado supervisionado com o backpropagation > '
        treinadorSupervisionado = BackpropTrainer(rede_neural_mlp, dados)
        '''Create a BackpropTrainer to train the specified `module` on the
        specified `dataset'''
        print 'done.'

        maior = 0
        menor = 999999999999
        print 'calculando maior e menor'
        for i in dados:
            val = float(i[0])
            print 'i <> ', val
            if val > maior:
                maior = val
            if val < menor:
                menor = val

        print 'maior['+str(maior)+'] menor['+str(menor)+']'

        numeroDeAcessos = 12
        numeroDeEpocasPorAcesso = 50  # 500 epocas


        #
        fig1 = plt.figure("figura 1 seno - valor ideal[azul] e aprendizado[vermelho]")
        ax1 = fig1.add_subplot(111)
        ax1.axis([0, 2 * math.pi, -1.5, 1.5])
        ax1.legend()
        ax1.grid(True)
        fig1.hold()

        #
        fig2 = plt.figure("Figura 2 seno - erro por numeros de acesso por epoca > ")
        ax2 = fig2.add_subplot(111)
        ax2.axis([-50, numeroDeAcessos * numeroDeEpocasPorAcesso + 50, 0.00001, 4])
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        fig2.hold()

        meansq = ModuleValidator()
        erro2 = meansq.MSE(treinadorSupervisionado.module, dados)
        lista_erro.append(erro2)

        print ('taxa de erro inicial = ', erro2)
        ax2.plot([0], [erro2], 'bo')

        tempoPausa = 1

        for i in range(numeroDeAcessos):
            treinadorSupervisionado.trainEpochs(numeroDeEpocasPorAcesso)
            meansq = ModuleValidator()
            erro2 = meansq.MSE(treinadorSupervisionado.module, dados)
            lista_erro.append(erro2)

            print 'erro2 no ', i, ' acesso = ', erro2

            ax1.plot(dados['input'], dados['target'], 'bo', markersize=7, markeredgewidth=0)
            ax1.plot(dados['input'], np.array([rede_neural_mlp.activate(datax) for datax, _ in dados]), 'ro',
                     markersize=7,
                     markeredgewidth=0)
            ax2.plot([numeroDeEpocasPorAcesso * (i + 1)], [erro2], 'bo')
            plt.pause(tempoPausa)
            ax1.plot(dados['input'], np.array([rede_neural_mlp.activate(datax) for datax, _ in dados]), 'wo',
                     markersize=9,
                     markeredgewidth=0)

            # NetworkWriter.writeToFile(rede_neural_mlp, 'MLP.xml') -> salvando o formato da rede

        return lista_erro



if __name__ == '__main__':
    ''''''
    seno = MLP_SENO()
    lista_erro = seno.constroi_MLP()

    from mv_arquivo import mv_file

    a = mv_file()

    for i in lista_erro:
        dados = str(str(i)+'\n')
        a.escrever(dados=dados, nome_file='erro.txt', modo='a')
        print 'erro = ', str(i)
    print 'view graphics of loss > '
    seno.ver_taxa_erro(lista_erro)

