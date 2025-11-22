import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters


# funckja do generowania obrazów testowych
def generate_test_image(text="hello world!", font_size=40, image_size=(400, 200), font=cv2.FONT_HERSHEY_SIMPLEX):
    img = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255
    
    # Parametry czcionki
    #font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 60
    target_size = (48, 48)
    font_scale = font_size / 30.0
    thickness = max(1, int(font_size / 20))
    
    # Rysowanie tekstu linia po linii
    lines = text.split('\n')
    y_offset = 50
    line_height = int(font_size * 1.5)
    
    for line in lines:
        cv2.putText(img, line, (20, y_offset), font, font_scale, 0, thickness, cv2.LINE_AA)
        y_offset += line_height
    
    return img

# funckja do preprocesingu
def preprocess_image(image, show_steps=True):
    if len(image.shape) == 3:  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else: gray = image.copy()
    
    # filtrowanie szumów 
    denoised = gaussian_filter(gray.astype(float), sigma=0.5)
    
    # binaryzacja
    try:
        threshold = filters.threshold_otsu(denoised)
        binary = (denoised < threshold).astype(np.uint8)
    except:
        binary = (denoised < 128).astype(np.uint8)
    
    if show_steps:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Oryginalny obraz')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(denoised, cmap='gray')
        plt.title('Po odszumianiu')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(binary, cmap='gray')
        plt.title('Po binaryzacji')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return binary

def generate_rotated_test_image(text="hello world!", font_size=60, rotation_angle=5, image_size=(600, 300)):
    img = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size / 30.0
    thickness = max(1, int(font_size / 20))
    
    lines = text.split('\n')
    y_offset = 80
    line_height = int(font_size * 1.5)
    
    for line in lines:
        cv2.putText(img, line, (50, y_offset), font, font_scale, 0, thickness, cv2.LINE_AA)
        y_offset += line_height
    
    if rotation_angle != 0:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), 
                           borderValue=255, flags=cv2.INTER_LINEAR)
    
    print(f"Wygenerowano obraz z obrotem: {rotation_angle}°")
    return img

def detect_text_angle_v2(binary_image, show_steps=True):
    # metoda Hougha
    edges = cv2.Canny(binary_image * 255, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is None:
        print("Nie wykryto linii - kąt = 0°")
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Normalizuj kąt do zakresu [-45°, 45°]
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
            
        if abs(angle) < 30:  angles.append(angle)
    
    if not angles:
        print("Nie wykryto linii poziomych - kąt = 0°")
        return 0.0

    detected_angle = np.median(angles)
    
    if show_steps:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binarny obraz')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Wykryte krawędzie')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        img_with_lines = cv2.cvtColor(binary_image * 255, cv2.COLOR_GRAY2RGB)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.imshow(img_with_lines)
        plt.title(f'Wykryte linie ({len(lines)} linii)')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        if len(angles) > 0:
            plt.hist(angles, bins=min(len(angles), 20), alpha=0.7, edgecolor='black')
            plt.axvline(detected_angle, color='red', linestyle='--', linewidth=2, 
                       label=f'Mediana: {detected_angle:.2f}°')
            plt.legend()
        plt.title(f'Rozkład kątów ({len(angles)} kątów)')
        plt.xlabel('Kąt (stopnie)')
        plt.ylabel('Częstość')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, 5)

        h, w = binary_image.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -detected_angle, 1.0)
        rotated_preview = cv2.warpAffine(binary_image * 255, rotation_matrix, (w, h), 
                                       borderValue=255)
        plt.imshow(rotated_preview, cmap='gray')
        plt.title(f'Podgląd korekcji\n(obrót o {-detected_angle:.2f}°)')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, f'Wykryte linie: {len(lines) if lines is not None else 0}', fontsize=12)
        plt.text(0.1, 0.7, f'Linie poziome: {len(angles)}', fontsize=12)
        plt.text(0.1, 0.6, f'Wykryty kąt: {detected_angle:.2f}°', fontsize=12, weight='bold')
        plt.text(0.1, 0.5, f'Korekcja: {-detected_angle:.2f}°', fontsize=12, weight='bold', color='red')
        plt.text(0.1, 0.3, f'Min kąt: {min(angles):.2f}°' if angles else 'Brak', fontsize=10)
        plt.text(0.1, 0.2, f'Max kąt: {max(angles):.2f}°' if angles else 'Brak', fontsize=10)
        plt.text(0.1, 0.1, f'Std dev: {np.std(angles):.2f}°' if angles else 'Brak', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Statystyki')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print(f"Wykryto {len(lines) if lines is not None else 0} linii")
    print(f"Linie poziome (tekst): {len(angles)}")
    print(f"Wykryty kąt nachylenia tekstu: {detected_angle:.2f}°")
    print(f"Korekcja zostanie wykonana o: {-detected_angle:.2f}°")
    
    return detected_angle

def rotate_image(image, angle, show_steps=True):
    if abs(angle) < 0.1:  
        print("Kąt za mały - pomijam obrót")
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))
    
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                            borderValue=255 if len(image.shape) == 2 else (255, 255, 255),
                            flags=cv2.INTER_LINEAR)
    
    if show_steps:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Przed obrotem')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(rotated, cmap='gray')
        plt.title(f'Po obrocie o {angle:.2f}°')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print(f"Obraz obrócony o {angle:.2f}°")
    return rotated


