import numpy as np
import cv2
import matplotlib.pyplot as plt
from .text_tools import *

# funckja obliczająca korelację fazową dla dwóch obrazów
def phase_correlation_score(img1, img2):
    f1 = np.fft.fft2(img1) # fft 2d
    f2 = np.fft.fft2(img2)
    cross_power_spectrum = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)# cross-power spectrum
    correlation_surface = np.fft.ifft2(cross_power_spectrum) # IFFT
    score = np.max(np.real(correlation_surface)) # szukamy peaku 
    return score

def recognize_character_fft(char_img, templates, show_match=False):
    priority_order = "sgw89650m2pabdeqk34hoy7xzcnu?v1tfji!rl,. "
    if char_img.size == 0: return '~', 0.0
    normalized_char = normalize_character(char_img)
    best_match = '~'
    best_score = 0.0 
    scores = {}
    
    for char in priority_order:
        if char in templates:
            normalized_template = normalize_character(templates[char])
            score = phase_correlation_score(normalized_char, normalized_template) # score opart na FFT
            scores[char] = score
            if score > best_score:
                best_score = score
                best_match = char

    if best_score < 0.1: best_match = '~'  # bardzo mało ale tyle trzeba

    if show_match:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 6, 1)
        plt.imshow(char_img, cmap='gray')
        plt.title('Rozpoznawany\nznak')
        plt.axis('off')
        for i, (char, score) in enumerate(sorted_scores):
            plt.subplot(1, 6, i+2)
            template = normalize_character(templates[char])
            plt.imshow(template, cmap='gray')
            plt.title(f'{char}\n{score:.3f}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        print(f"Najlepsze dopasowania: {sorted_scores}")
    return best_match, best_score   

def preprocess_image_fft(image, show_steps=False):
    if len(image.shape) > 2: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Używamy progu adaptacyjnego dla lepszych wyników na nierównym oświetleniu
    binary_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary = (binary_inv / 255).astype(np.uint8) # Normalizacja do 0 i 1
    
    if show_steps:
        plt.imshow(binary, cmap='gray')
        plt.title("Obraz po preprocessingu (zbinoryzowany)")
        plt.axis('off')
        plt.show()
    return binary