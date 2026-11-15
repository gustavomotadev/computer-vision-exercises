import cv2
import numpy as np
import matplotlib.pyplot as plt

# caminho da imagem de teste
caminho_imagem = 'original.jpg'
caminho_hist_original = 'histograma_original__.jpg'
caminho_hist_normalizado = 'histograma_normalizado__.jpg'
caminho_hist_combinado = 'histograma_combinado__.jpg'
caminho_hist_limiar1 = 'histograma_limiar1__.jpg'
caminho_limiar1 = 'limiar1__.jpg'
caminho_combinado_limiar1 = 'original+limiar1__.jpg'
caminho_hist_3limiares = 'histograma_3limiares__.jpg'
caminho_limiar2 = 'limiar2__.jpg'
caminho_limiar3 = 'limiar3__.jpg'
caminho_combinado_3limiares = 'limiar1+limiar2+limiar3__.jpg'
caminho_otsu = 'otsu__.jpg'
caminho_combinado_otsu = 'limiar2+otsu__.jpg'

formato_salvamento = [int(cv2.IMWRITE_JPEG_QUALITY), 98]

def desenhar_histograma(histograma, titulo, nome_arquivo, limiar1=None, limiar2=None, limiar3=None):

    imax = histograma.max()

    plt.figure(figsize=(1024, 512, 'px'), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.bar(range(255+1), histograma.ravel(), width=1.0, color='black', edgecolor='black')

    plt.title(titulo, color='black', fontsize=14)
    plt.xlabel('Intensidade', color='black', fontsize=14)
    plt.ylabel('Frequência', color='black', fontsize=14)
    plt.xlim([0-1, 255+1])
    plt.ylim([0, imax])
    plt.xticks(range(0, 255+1, 15))

    ax.tick_params(colors='black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # linhas dos limiares
    if limiar1 is not None:
        plt.axvline(x=limiar1, color='red', linestyle='--', linewidth=1.0, label=f'Limiar 1 = {limiar1}')
    if limiar2 is not None:
        plt.axvline(x=limiar2, color='green', linestyle='-.', linewidth=1.0, label=f'Limiar 2 = {limiar2}')
    if limiar3 is not None:
        plt.axvline(x=limiar3, color='blue', linestyle=':', linewidth=1.0, label=f'Limiar 3 = {limiar3}')

    # legenda dos limiares
    if limiar1 is not None or limiar2 is not None or limiar3 is not None:
        plt.legend(loc='upper right', facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.savefig(nome_arquivo, pil_kwargs={'quality': 98})

def combinacao_vertical(caminho_cima, caminho_baixo, caminho_salvar):

    cima = cv2.imread(caminho_hist_original, cv2.IMREAD_GRAYSCALE)
    if cima is None:
        print(f'Erro ao abrir imagem do histograma 1: "{caminho_cima}".')
        exit(1)

    baixo = cv2.imread(caminho_hist_normalizado, cv2.IMREAD_GRAYSCALE)
    if baixo is None:
        print(f'Erro ao abrir imagem do histograma 1: "{caminho_baixo}".')
        exit(1)

    cima_baixo = cv2.vconcat([cima, baixo])

    cv2.imwrite(caminho_salvar, cima_baixo, formato_salvamento)

def legendar_imagem(imagem, legenda, x, y, cor):
    nova = imagem.copy()
    cv2.putText(
        nova, 
        text=legenda, 
        org=(x, y), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.5, 
        color=cor, 
        thickness=1, 
        lineType=cv2.LINE_AA
    )
    return nova

if __name__ == '__main__':

    # leitura da imagem
    imagem_original = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if imagem_original is None:
        print(f'Erro ao abrir a imagem "{caminho_imagem}".')
        exit(1)

    # extração de dimensões e medidas
    altura, largura = imagem_original.shape
    num_pixels = altura*largura
    menor_intensidade = imagem_original.min()
    maior_intensidade = imagem_original.max()
    media_intensidade = imagem_original.mean()
    # print(altura, largura, num_pixels, menor_intensidade, maior_intensidade, media_intensidade)

    print(f'Medidas da Imagem:')
    print(f'Largura = {largura}; Altura = {altura};  Número de Pixels = {num_pixels};')
    print(f'Menor Intensidade = {menor_intensidade}; Maior Intensidade = {maior_intensidade}; Intensidade Média = {media_intensidade:.1f}')

    # cálculo do histograma
    # imagem, canal 0, sem máscara, 256 níveis, range 0 a 255
    histograma_original = cv2.calcHist([imagem_original], [0], None, [256], [0, 255+1])
    # print(histograma_original)
    desenhar_histograma(histograma_original, 'Histograma da Imagem de Teste', caminho_hist_original)

    # normalização do histograma
    histograma_normalizado = histograma_original / num_pixels
    desenhar_histograma(histograma_normalizado, 'Histograma Normalizado', caminho_hist_normalizado)

    # somatorio do histograma normalizado
    somatorio_histograma = histograma_normalizado.sum()
    print(f'\nSomatório do Histograma Normalizado = {somatorio_histograma}.') 

    # combinar histogramas
    combinacao_vertical(caminho_hist_original, caminho_hist_normalizado, caminho_hist_combinado)

    # escolha do primeiro limiar
    limiar1 = 202

    # desenhar limiar sobre histograma
    desenhar_histograma(histograma_normalizado, 'Posição do Primeiro Limiar', caminho_hist_limiar1, limiar1=limiar1)

    # limiarização
    _, imagem_limiar1 = cv2.threshold(imagem_original, limiar1, 255, cv2.THRESH_BINARY)
    legendada1 = legendar_imagem(imagem_limiar1, f'Limiar 1 = {limiar1}', 64, 64, 255)
    cv2.imwrite(caminho_limiar1, legendada1, formato_salvamento)

    # combinar imagem original e limiar 1
    legendada0 = legendar_imagem(imagem_original, 'Original', 64, 64, 255)
    combinado_limiar1 = cv2.hconcat([legendada0, legendada1])
    cv2.imwrite(caminho_combinado_limiar1, combinado_limiar1, formato_salvamento)

    # escolha do segundo e terceiro limiares
    limiar2 = 127
    limiar3 = 56

    # desenhar o histograma com os 3 limiares
    desenhar_histograma(histograma_normalizado, 'Posição dos 3 Limiares', caminho_hist_3limiares, 
        limiar1=limiar1, limiar2=limiar2, limiar3=limiar3)

    # novas limiarizações
    _, imagem_limiar2 = cv2.threshold(imagem_original, limiar2, 255, cv2.THRESH_BINARY)
    legendada2 = legendar_imagem(imagem_limiar2, f'Limiar 2 = {limiar2}', 64, 64, 255)
    cv2.imwrite(caminho_limiar2, legendada2, formato_salvamento)
    _, imagem_limiar3 = cv2.threshold(imagem_original, limiar3, 255, cv2.THRESH_BINARY)
    legendada3 = legendar_imagem(imagem_limiar3, f'Limiar 3 = {limiar3}', 64, 64, 255)
    cv2.imwrite(caminho_limiar3, legendada3, formato_salvamento)

    # combinar as limiarizações em uma única imagem
    combinado_3limiares = cv2.hconcat([legendada1, legendada2, legendada3])
    cv2.imwrite(caminho_combinado_3limiares, combinado_3limiares, formato_salvamento)

    # limiarização pelo método de Otsu
    limiar_otsu, imagem_otsu = cv2.threshold(imagem_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f'\nLimiar obtido pelo método de Otsu = {limiar_otsu}.')
    legendada_otsu = legendar_imagem(imagem_otsu, f'Limiar Otsu = {int(limiar_otsu)}', 64, 64, 255)
    cv2.imwrite(caminho_otsu, legendada_otsu, formato_salvamento)

    # combinando imagem otsu com o limiar 2, que foi o mais similar
    combinado_otsu = cv2.hconcat([legendada2, legendada_otsu])
    cv2.imwrite(caminho_combinado_otsu, combinado_otsu, formato_salvamento)

    # contabilização de pixels
    print()
    for imagem, id in [(imagem_limiar1, '1'), (imagem_limiar2, '2'), (imagem_otsu, 'Otsu'), (imagem_limiar3, '3')]:
        n0 = np.sum(imagem == 0)
        n1 = np.sum(imagem == 255)
        p0 = n0/num_pixels
        p1 = n1/num_pixels
        p_total = p0+p1
        print(f'[Limiar {id:4}] N0= {n0:7}; N1= {n1:7}; P0= {p0:.4f}; P1= {p1:.4f}; P0+P1= {p_total:.4f}')
