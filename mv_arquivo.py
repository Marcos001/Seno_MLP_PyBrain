
class mv_file:

    def __init__(self):
        pass

    def escrever(self, dados, nome_file, modo):
        '''escrever dados em um arquivo'''
        arquivo = open(nome_file, modo)
        arquivo.write(dados)
        arquivo.close()

    def ler(self, nome_file):
        '''ler dados em um arquivo e retona os dados lidos'''
        arquivo = open(file=nome_file, mode='r')
        dados_lidos = arquivo.read()
        arquivo.close()
        return dados_lidos