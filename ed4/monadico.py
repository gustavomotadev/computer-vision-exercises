import cv2
import numpy as np
import matplotlib.pyplot as plt

caminho_imagem = 'original.jpg'
caminho_brilho = 'brilho__.jpg'
caminho_contraste = 'contraste__.jpg'
caminho_negativo = 'negativo__.jpg'
caminho_posterizada4 = 'posterizada_4__.jpg'
caminho_posterizada4_8_16 = 'posterizada_4_8_16__.jpg'
caminho_combinado5 = 'combinado5__.jpg'
caminho_comparacao = 'comparacao__.jpg'

formato_salvamento = [int(cv2.IMWRITE_JPEG_QUALITY), 98]

def carregar_imagem(caminho):

    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        print(f'Erro ao abrir a imagem "{caminho}".')
        exit(1)

    return imagem

def alterar_contraste_brilho(imagem, contraste, brilho, inteiro):

    lim_inf = 0
    lim_sup = 1

    if inteiro:
        lim_sup = 255
        imagem = imagem.astype(np.float64)
    
    alterada = np.clip(imagem * contraste + brilho, lim_inf, lim_sup)

    if inteiro:
        return alterada.astype(np.uint8)

    return alterada

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

def uint8_para_double(imagem):

    return imagem.astype(np.float64) / 255.0

def double_para_uint8(imagem):

    return np.round(np.clip(imagem, 0.0, 1.0) * 255).astype(np.uint8)

def negativo(imagem, inteiro):

    lim_sup = 255 if inteiro else 1

    return lim_sup - imagem

def posterizar(imagem, niveis):

    delta = 256 / niveis

    return np.clip(np.floor(imagem.astype(np.float64) / delta) * delta, 0, 255).astype(np.uint8)

def min_max_media(imagem):

    img_min = np.min(imagem)
    img_max = np.max(imagem)
    img_media = np.mean(imagem)

    return img_min, img_max, img_media

def print_cabecalho():

    print(f'Tipo de Dado | Intensidade Mínima | Intensidade Máxima | Intensidade Média')

def print_medidas(imagem):

    img_min, img_max, img_media = min_max_media(imagem)

    if imagem.dtype.name == 'uint8':
        print(f'{'uint8':12} | {img_min:<18} | {img_max:<18} | {img_media:<17.1f}')
    else:
        print(f'{'double':12} | {img_min:<18.3f} | {img_max:<18.3f} | {img_media:<17.3f}')

if __name__ == '__main__':

    # leitura da imagem
    imagem_uint8 = carregar_imagem(caminho_imagem)

    # conversão para double
    imagem_double = uint8_para_double(imagem_uint8)

    # cálculo das medidas:
    altura, largura = imagem_uint8.shape
    uint8_min, uint8_max, uint8_media = min_max_media(imagem_uint8)
    double_min, double_max, double_media = min_max_media(imagem_double)

    # apresentação das medidas
    print('Imagem Original:')
    print_cabecalho()
    print_medidas(imagem_uint8)
    print_medidas(imagem_double)

    # alteração do brilho
    uint8_brilho = alterar_contraste_brilho(imagem_uint8, 1, 64, True)
    double_brilho = alterar_contraste_brilho(imagem_double, 1, 0.25, False)
    
    # apresentação das medidas
    print('\nAlteração de Brilho:')
    print_cabecalho()
    print_medidas(uint8_brilho)
    print_medidas(double_brilho)

    # criação da imagem combinada de brilho
    original_legenda = legendar_imagem(imagem_uint8, 'Original', 64, 64, 0)
    brilho_uint8_legenda = legendar_imagem(uint8_brilho, '[uint8] Brilho +64', 64, 64, 0)
    brilho_double_legenda = legendar_imagem(double_para_uint8(double_brilho), '[double] Brilho 0.25', 64, 64, 0)
    barra_vertical = np.full((altura, 4, 1), 255, dtype=np.uint8)
    combinado_brilho = cv2.hconcat([original_legenda, barra_vertical, brilho_uint8_legenda, barra_vertical, brilho_double_legenda])
    cv2.imwrite(caminho_brilho, combinado_brilho, formato_salvamento)

    # alteração de contraste e brilho
    uint8_contraste = alterar_contraste_brilho(imagem_uint8, 2, -127, True)
    double_contraste = alterar_contraste_brilho(imagem_double, 2, -0.5, False)

    # apresentação das medidas
    print('\nAlteração de Contraste e Brilho:')
    print_cabecalho()
    print_medidas(uint8_contraste)
    print_medidas(double_contraste)

    # criação da imagem combinada de contraste
    contraste_uint8_legenda = legendar_imagem(uint8_contraste, '[uint8] Contraste x2; Brilho -127', 64, 64, 0)
    contraste_double_legenda = legendar_imagem(double_para_uint8(double_contraste), '[double] Contraste x2; Brilho -0.5', 64, 64, 0)
    combinado_contraste = cv2.hconcat([original_legenda, barra_vertical, contraste_uint8_legenda, barra_vertical, contraste_double_legenda])
    cv2.imwrite(caminho_contraste, combinado_contraste, formato_salvamento)

    # negativo
    uint8_negativo = negativo(imagem_uint8, True)
    double_negativo = negativo(imagem_double, False)

    # apresentação das medidas
    print('\nImagem Negativa:')
    print_cabecalho()
    print_medidas(uint8_negativo)
    print_medidas(double_negativo)

    # criação da imagem combinada de negativos
    negativo_uint8_legenda = legendar_imagem(uint8_negativo, '[uint8] Negativo', 64, 64, 255)
    negativo_double_legenda = legendar_imagem(double_para_uint8(double_negativo), '[double] Negativo', 64, 64, 255)
    combinado_negativo = cv2.hconcat([original_legenda, barra_vertical, negativo_uint8_legenda, barra_vertical, negativo_double_legenda])
    cv2.imwrite(caminho_negativo, combinado_negativo, formato_salvamento)

    # posterização L = 4
    uint8_posterizada4 = posterizar(imagem_uint8, 4)
    print(f'\nNíveis presentes na imagem posterizada com L = 4: {np.unique(uint8_posterizada4)}.')

    # criação da imagem combinada posterizada L = 4
    posterizada4_legenda = legendar_imagem(uint8_posterizada4, '[uint8] Posterizado L=4', 64, 64, 0)
    combinado_posterizada4 = cv2.hconcat([original_legenda, barra_vertical, posterizada4_legenda])
    cv2.imwrite(caminho_posterizada4, combinado_posterizada4, formato_salvamento)

    # posterização L = 8 e L = 16
    uint8_posterizada8 = posterizar(imagem_uint8, 8)
    print(f'\nNíveis presentes na imagem posterizada com L = 8: {np.unique(uint8_posterizada8)}.')
    uint8_posterizada16 = posterizar(imagem_uint8, 16)
    print(f'\nNíveis presentes na imagem posterizada com L = 16: {np.unique(uint8_posterizada16)}.')

    # criação da imagem combinada posterizada 4, 8 e 16
    posterizada8_legenda = legendar_imagem(uint8_posterizada8, '[uint8] Posterizado L=8', 64, 64, 0)
    posterizada16_legenda = legendar_imagem(uint8_posterizada16, '[uint8] Posterizado L=16', 64, 64, 0)
    combinado_posterizada4_8_16 = cv2.hconcat([posterizada4_legenda, barra_vertical, posterizada8_legenda, barra_vertical, posterizada16_legenda])
    cv2.imwrite(caminho_posterizada4_8_16, combinado_posterizada4_8_16, formato_salvamento)

    # combinado de 5 imagens
    linha1 = cv2.hconcat([original_legenda, barra_vertical, brilho_uint8_legenda, barra_vertical, contraste_uint8_legenda])
    barra_horizontal_g = np.full((4, linha1.shape[1], 1), 255, dtype=np.uint8)
    linha2 = cv2.hconcat([barra_vertical, negativo_uint8_legenda, barra_vertical, posterizada4_legenda])
    linha2 = cv2.copyMakeBorder(linha2, 0, 0, linha1.shape[1]-linha2.shape[1], 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    comparacao = cv2.vconcat([linha1, barra_horizontal_g, linha2])
    cv2.imwrite(caminho_combinado5, comparacao, formato_salvamento)

    # comparação de uint8 e double
    mascara_eq = (double_contraste == uint8_para_double(uint8_contraste)).astype(np.uint8)
    mascara_diff = (double_contraste != uint8_para_double(uint8_contraste)).astype(np.uint8)
    aplicada_eq = uint8_contraste * mascara_eq
    aplicada_diff = uint8_contraste * mascara_diff
    valores_iguais = np.unique(aplicada_eq)
    print(f'\nValores em que a imagem com alteração de contraste é igual para uint8 e double: {valores_iguais}')

    # criação da imagem de comparação
    temp1 = legendar_imagem(mascara_eq * 255, 'Máscara de Igualdade', 64, 64, 127)
    temp2 = legendar_imagem(aplicada_diff, 'Pixels Diferentes', 64, 64, 127)
    comparacao = cv2.hconcat([contraste_uint8_legenda, barra_vertical, temp1, barra_vertical, temp2])
    cv2.imwrite(caminho_comparacao, comparacao, formato_salvamento)

    # comparação de um pixel
    print(f'\nValor do pixel [0][1023]: uint8 = {uint8_contraste[0][1023]}; float = {round(double_contraste[0][1023] * 255)}')

