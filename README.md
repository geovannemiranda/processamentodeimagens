# processamentodeimagens
pacote de processamento de imagens - DIO

from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Leitura e Exibição de Imagens
def abrir_imagem(caminho):
    try:
        img = Image.open(caminho)
        img.show()
        return img
    except Exception as e:
        print(f"Erro ao abrir a imagem: {e}")
        return None

# 2. Transformações Básicas
def rotacionar_imagem(img, angulo):
    """Rotaciona a imagem em um determinado ângulo"""
    return img.rotate(angulo)

def redimensionar_imagem(img, largura, altura):
    """Redimensiona a imagem para a largura e altura especificadas"""
    return img.resize((largura, altura))

# 3. Filtros
def aplicar_suavizacao(img):
    """Aplica suavização à imagem"""
    return img.filter(ImageFilter.SMOOTH)

def aplicar_nitidez(img):
    """Aplica um filtro de nitidez à imagem"""
    return img.filter(ImageFilter.SHARPEN)

# 4. Detecção de Bordas usando OpenCV
def detectar_bordas(caminho):
    """Detecta bordas em uma imagem usando o algoritmo Canny"""
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    bordas = cv2.Canny(img, 100, 200)
    cv2.imshow('Bordas', bordas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 5. Histograma
def gerar_histograma(caminho):
    """Gera o histograma de uma imagem em escala de cinza"""
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Histograma')
    plt.show()

# 6. Efeitos Especiais
def aplicar_sepia(img):
    """Aplica o efeito sepia na imagem"""
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = np.dot(np.array(img), sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_img)

def aplicar_negativo(img):
    """Aplica o efeito negativo na imagem"""
    inverted_img = np.invert(np.array(img))
    return Image.fromarray(inverted_img)

# Função principal para testar o pacote
def main():
    # Defina o caminho para sua imagem
    caminho_imagem = "sua_imagem_aqui.jpg"

    # 1. Abrir e exibir a imagem
    img = abrir_imagem(caminho_imagem)
    if img is None:
        return
    
    # 2. Aplicar transformações básicas
    img_rotacionada = rotacionar_imagem(img, 45)
    img_rotacionada.show()

    img_redimensionada = redimensionar_imagem(img, 300, 300)
    img_redimensionada.show()

    # 3. Aplicar filtros
    img_suave = aplicar_suavizacao(img)
    img_suave.show()

    img_nitida = aplicar_nitidez(img)
    img_nitida.show()

    # 4. Detectar bordas
    detectar_bordas(caminho_imagem)

    # 5. Gerar histograma
    gerar_histograma(caminho_imagem)

    # 6. Aplicar efeitos especiais
    img_sepia = aplicar_sepia(img)
    img_sepia.show()

    img_negativo = aplicar_negativo(img)
    img_negativo.show()

if __name__ == "__main__":
    main()
