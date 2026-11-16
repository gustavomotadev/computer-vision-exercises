import cv2
import numpy as np
import matplotlib.pyplot as plt

# caminho da imagem de teste
caminho_imagem = 'original.jpg'
caminho_magnitude_uv = 'magnitude_Iu_Iv__.jpg'
caminho_sinal_uv = 'sinal_Iu_Iv__.jpg'
caminho_gradiente = 'gradiente__.jpg'
caminho_combinado4 = 'combinado4__.jpg'
caminho_histograma = 'histograma__.jpg'
caminho_limiar_manual = 'limiar_manual__.jpg'
caminho_limiar_otsu = 'limiar_otsu__.jpg'
caminho_combinado2 = 'combinado2__.jpg'
caminho_combinado3 = 'combinado3__.jpg'

formato_salvamento = [int(cv2.IMWRITE_JPEG_QUALITY), 98]

def representacao_sinal(imagem):

    maximo_absoluto = max(abs(imagem.min()), abs(imagem.max()))

    normalizada = imagem / maximo_absoluto

    altura, largura = imagem.shape
    nova_imagem = np.zeros((altura, largura, 3), dtype=np.uint8)

    mascara_positiva = normalizada > 0
    mascara_negativa = normalizada < 0

    canal_R = 2
    canal_G = 1
    canal_B = 0
    nova_imagem[mascara_positiva, canal_R] = (normalizada[mascara_positiva] * 255).astype(np.uint8)
    nova_imagem[mascara_positiva, canal_G] = (normalizada[mascara_positiva] * 255).astype(np.uint8)
    nova_imagem[mascara_negativa, canal_B] = (np.abs(normalizada[mascara_negativa]) * 255).astype(np.uint8)
    nova_imagem[mascara_negativa, canal_G] = (np.abs(normalizada[mascara_negativa]) * 255).astype(np.uint8)
    
    return nova_imagem

def representacao_magnitude(imagem):

    magnitude = np.abs(imagem)

    nova_imagem = cv2.normalize(
        magnitude, 
        None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )

    return nova_imagem

def legendar_imagem(imagem, legenda, x, y, cor):

    nova_imagem = imagem.copy()

    cv2.putText(
        nova_imagem, 
        text=legenda, 
        org=(x, y), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.5, 
        color=cor, 
        thickness=1, 
        lineType=cv2.LINE_AA
    )

    return nova_imagem

def magnitude_gradiente(imagem_u, imagem_v):

    magnitude = cv2.magnitude(imagem_u, imagem_v)

    nova_imagem = cv2.normalize(
        magnitude, 
        None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )

    return nova_imagem

def desenhar_histograma(histograma, titulo, nome_arquivo, limiar_manual=None, limiar2=None, limiar_otsu=None):

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
    if limiar_manual is not None:
        plt.axvline(x=limiar_manual, color='red', linestyle='--', linewidth=1.0, label=f'Limiar Manual = {limiar_manual}')
    if limiar_otsu is not None:
        plt.axvline(x=limiar_otsu, color='blue', linestyle=':', linewidth=1.0, label=f'Limiar Otsu = {limiar_otsu}')

    # legenda dos limiares
    if limiar_manual is not None or limiar2 is not None or limiar_otsu is not None:
        plt.legend(loc='upper right', facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.savefig(nome_arquivo, pil_kwargs={'quality': 98})

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

    # definicao do kernel de sobel Kv
    kernel_v = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)

    # obtenção do kernel Ku a partir da transposta de Kv
    kernel_u = kernel_v.T

    # impressao nos kernels
    print('\nKernel Kv:\n', str(kernel_v)[1:-1])
    print('\nKernel Ku:\n', str(kernel_u)[1:-1])

    # aplicação de convolução
    # o uso de ddepth=cv2.CV_32F fará a imagem resultante ser ponto flutuante, preservando valores negativos
    imagem_u = cv2.filter2D(imagem_original, ddepth=cv2.CV_32F, kernel=kernel_u)
    imagem_v = cv2.filter2D(imagem_original, ddepth=cv2.CV_32F, kernel=kernel_v)

    # representação gráfica da magnitude das componentes Iu e Iv do gradiente
    magnitude_u = representacao_magnitude(imagem_u)
    magnitude_v = representacao_magnitude(imagem_v)
    magnitude_u_leg = legendar_imagem(magnitude_u, 'Magnitude de Iu', 64, 64, (255, 255, 255))
    magnitude_v_leg = legendar_imagem(magnitude_v, 'Magnitude de Iv', 64, 64, (255, 255, 255))
    barra_vertical = np.full((altura, 4, 1), 255, dtype=np.uint8)
    magnitude_uv = cv2.hconcat([magnitude_u_leg, barra_vertical, magnitude_v_leg])
    cv2.imwrite(caminho_magnitude_uv, magnitude_uv, formato_salvamento)

    # representação gráfica do sinal das componentes Iu e Iv do gradiente
    # positivo é amarelo, negativo é ciano
    sinal_u = representacao_sinal(imagem_u)
    sinal_v = representacao_sinal(imagem_v)
    sinal_u_leg = legendar_imagem(sinal_u, 'Sinal de Iu', 64, 64, (255, 255, 255))
    sinal_v_leg = legendar_imagem(sinal_v, 'Sinal de Iv', 64, 64, (255, 255, 255))
    barra_vertical_rgb = np.full((altura, 4, 3), (255, 255, 255), dtype=np.uint8)
    sinal_uv = cv2.hconcat([sinal_u_leg, barra_vertical_rgb, sinal_v_leg])
    cv2.imwrite(caminho_sinal_uv, sinal_uv, formato_salvamento)

    # magnitude do gradiente obtida de sqrt(Iu*Iu + Iv*Iv)
    imagem_gradiente = magnitude_gradiente(imagem_u, imagem_v) 
    gradiente_leg = legendar_imagem(imagem_gradiente, 'Magnitude do Gradiente', 64, 64, 255)
    cv2.imwrite(caminho_gradiente, gradiente_leg, formato_salvamento)

    # combinacao de 4 imagens
    original_leg = legendar_imagem(imagem_original, 'Original', 64, 64, 255)
    linha1 = cv2.hconcat([original_leg, barra_vertical, magnitude_u_leg])
    linha2 = cv2.hconcat([magnitude_v_leg, barra_vertical, gradiente_leg])
    barra_horizontal_g = np.full((4, linha1.shape[1], 1), 255, dtype=np.uint8)
    combinado4 = cv2.vconcat([linha1, barra_horizontal_g, linha2])
    cv2.imwrite(caminho_combinado4, combinado4, formato_salvamento)

    # limiarização com limiar escolhido manualmente
    limiar_manual = 75
    _, imagem_limiar_manual = cv2.threshold(imagem_gradiente, limiar_manual, 255, cv2.THRESH_BINARY)
    legendada_manual = legendar_imagem(imagem_limiar_manual, f'Limiar Manual = {limiar_manual}', 64, 64, 255)
    cv2.imwrite(caminho_limiar_manual, legendada_manual, formato_salvamento)

    # limiarização automática com método de otsu
    limiar_otsu, imagem_otsu = cv2.threshold(imagem_gradiente, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f'\nLimiar obtido pelo método de Otsu = {limiar_otsu}.')
    legendada_otsu = legendar_imagem(imagem_otsu, f'Limiar Otsu = {int(limiar_otsu)}', 64, 64, 255)
    cv2.imwrite(caminho_limiar_otsu, legendada_otsu, formato_salvamento)

    # histograma normalizado do gradiente
    histograma_gradiente = cv2.calcHist([imagem_gradiente], [0], None, [256], [0, 255+1])
    histograma_normalizado = histograma_gradiente / num_pixels
    desenhar_histograma(histograma_normalizado, 'Histograma Normalizado do Gradiente', caminho_histograma, limiar_manual=limiar_manual, limiar_otsu=limiar_otsu)

    # combinação de 3 imagens
    combinado2 = cv2.hconcat([legendada_manual, barra_vertical, legendada_otsu])
    combinado3 = cv2.hconcat([gradiente_leg, barra_vertical, legendada_manual, barra_vertical, legendada_otsu])
    cv2.imwrite(caminho_combinado2, combinado2, formato_salvamento)
    cv2.imwrite(caminho_combinado3, combinado3, formato_salvamento)

