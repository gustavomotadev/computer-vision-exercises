import cv2
import numpy as np

# caminho da imagem de teste
caminho_imagem = 'original.jpg'
caminho_4pontos = '4pontos__.jpg'
caminho_retificada = 'retificada__.jpg'
caminho_comparacao = 'comparacao__.jpg'

def desenhar_pontos(imagem, pontos, cor, nomes):

    for i in range(len(pontos)):

        cv2.drawMarker(imagem, pontos[i], cor, cv2.MARKER_CROSS, 20, 2)

        cv2.putText(imagem, nomes[i], (pontos[i][0]+10,pontos[i][1]+30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, cor)

def imprimir_matriz(matriz, titulo):

    print(f'{titulo}: \n')
    print(' ' + np.array2string(matriz, precision=4, 
        suppress_small=True, sign='+', separator='\t'
        ).replace('[', '|'
        ).replace(']', '|')[1:-1] + ' ')
    print()

if __name__ == '__main__':

    # leitura da imagem
    imagem_original = cv2.imread(caminho_imagem)
    if imagem_original is None:
        print(f'Erro ao abrir a imagem "{caminho_imagem}".')
        exit(1)

    # pontos da imagem original
    pontos_fonte = [ [224, 283], [581, 185], [849, 574], [388, 752] ]

    # ponto de verificacao
    ponto_verificacao = [481, 373]

    # desenhar 4 pontos fonte
    imagem_copia = imagem_original.copy()
    desenhar_pontos(imagem_copia, pontos_fonte, (0, 0, 255), ['P1', 'P2', 'P3', 'P4'])
    cv2.imwrite(caminho_4pontos, imagem_copia, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    # desenhar ponto de verificacao
    imagem_copia = imagem_original.copy()
    desenhar_pontos(imagem_copia, [ponto_verificacao], (255, 0, 0), ['P5'])

    # pontos de destino
    largura_destino = 480
    altura_destino = 640
    pontos_destino = [ [0, 0], [largura_destino, 0], [largura_destino, altura_destino], [0, altura_destino]]

    # matriz de homografia
    matriz_homografia = cv2.getPerspectiveTransform(np.float32(pontos_fonte), np.float32(pontos_destino))
    imprimir_matriz(matriz_homografia, 'Matriz de Homografia')

    # aplicação da transformação na imagem
    imagem_retificada = cv2.warpPerspective(imagem_original, matriz_homografia, (largura_destino, altura_destino))
    cv2.imwrite(caminho_retificada, imagem_retificada, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    # aplicação da transformação no ponto P5
    ponto_transformado = cv2.perspectiveTransform(np.float32([[ponto_verificacao]]), matriz_homografia)[0][0]
    ponto_transformado = [round(ponto_transformado[0]), round(ponto_transformado[1])]

    # desenhar ponto transformado
    desenhar_pontos(imagem_retificada, [ponto_transformado], (255, 0, 0), ['P5'])

    # combinar duas imagens para comparacao
    altura_fonte = imagem_original.shape[0]
    retificada_copia = cv2.copyMakeBorder(imagem_retificada, altura_fonte-altura_destino, 0, 0, 0, 
        cv2.BORDER_CONSTANT, value=(0,0,0))
    imagem_comparacao = cv2.hconcat([imagem_copia, retificada_copia])
    cv2.imwrite(caminho_comparacao, imagem_comparacao, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

