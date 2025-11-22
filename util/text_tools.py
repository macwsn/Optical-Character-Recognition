import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def crop_to_content(img):
    ys, xs = np.where(img > 0)
    if len(ys) == 0 or len(xs) == 0:
        return img
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

# generowanie wzorców czcionki
def generate_character_templates(font_size=40, save_dir="templates", font=cv2.FONT_HERSHEY_SIMPLEX):
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    chars = "abcdefghijklmnopqrstuvwxyz0123456789!? "
    templates = {}

    #font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size / 30.0
    thickness = max(1, int(font_size / 20))
    kernel = np.ones((2, 2), np.uint8)  # do dylatacji

    for char in chars:
        if char == " ":
            img = np.ones((40, 40), dtype=np.uint8) * 255
            templates[char] = img
            cv2.imwrite(os.path.join(save_dir, f"space.png"), img)
            continue

        (w, h), baseline = cv2.getTextSize(char, font, font_scale, thickness)
        img_size = (w + 20, h + baseline + 20)
        img = np.ones(img_size[::-1], dtype=np.uint8) * 255
        cv2.putText(img, char, (10, h + 10), font, font_scale, 0, thickness, cv2.LINE_AA)

        binary = (img < 128).astype(np.uint8)
        cropped = crop_to_content(binary)
        cropped = cv2.dilate(cropped, kernel, iterations=1)
        templates[char] = cropped
        cv2.imwrite(os.path.join(save_dir, f"{char}.png"), cropped * 255)

    print(f"Wygenerowano {len(templates)} szablonów znaków")
    return templates

# funkcja do segmentacji tekstu
def segment_text_lines(binary_image, show_steps=True):
    horizontal_proj = np.sum(binary_image, axis=1)
    lines = []
    in_text = False
    start_row = 0
    min_line_height = 5
    
    for i, proj_val in enumerate(horizontal_proj):
        if proj_val > 0 and not in_text:
            in_text = True
            start_row = i
        elif proj_val == 0 and in_text:
            in_text = False
            if i - start_row >= min_line_height:
                lines.append((start_row, i))
    
    if in_text and len(horizontal_proj) - start_row >= min_line_height:  lines.append((start_row, len(horizontal_proj)))
    
    if show_steps:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(binary_image, cmap='gray')
        for start, end in lines:
            plt.axhline(y=start, color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=end, color='red', linestyle='--', alpha=0.7)
        plt.title(f'Segmentacja linii ({len(lines)} linii)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.plot(horizontal_proj, range(len(horizontal_proj)))
        plt.title('Projekcja pozioma')
        plt.xlabel('Suma pikseli')
        plt.ylabel('Wiersz')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print(f"Znaleziono {len(lines)} linii tekstu")
    return lines

# funckja do segmentacji znaków
def segment_characters(line_image, show_steps=True):
    vertical_proj = np.sum(line_image, axis=0)

    chars = []
    in_char = False
    start_col = 0
    min_char_width = 2
    prev_end = None
    space_threshold = int(0.5 * line_image.shape[0])
    
    for i, proj_val in enumerate(vertical_proj):
        if proj_val > 0 and not in_char:
            in_char = True
            start_col = i
            if prev_end is not None and (start_col - prev_end) > space_threshold: chars.append('space')
        elif proj_val == 0 and in_char:
            in_char = False
            if i - start_col >= min_char_width:
                chars.append((start_col, i))
                prev_end = i
    
    if in_char and len(vertical_proj) - start_col >= min_char_width: chars.append((start_col, len(vertical_proj)))
    
    if show_steps and len(chars) > 0:
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(line_image, cmap='gray')

        for c in chars:
            if isinstance(c, tuple):
                start, end = c
                plt.axvline(x=start, color='red', linestyle='--', alpha=0.7)
                plt.axvline(x=end, color='red', linestyle='--', alpha=0.7)
        plt.title(f'Segmentacja znaków ({len(chars)} znaków)')
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.plot(vertical_proj)
        plt.title('Projekcja pionowa')
        plt.xlabel('Kolumna')
        plt.ylabel('Suma pikseli')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return chars

# normalizacja znaków
def normalize_character(char_img, target_size=(48,48)):
    char_img = crop_to_content(char_img)
    if char_img.size == 0: return np.zeros(target_size)
    
    kernel = np.ones((2, 2), np.uint8)
    char_img = cv2.erode(char_img, kernel, iterations=1)
    
    h, w = char_img.shape
    if h == 0 or w == 0: return np.zeros(target_size)

    th, tw = target_size
    scale = min(th/h, tw/w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    if new_h == 0 or new_w == 0: return np.zeros(target_size)
    resized = cv2.resize(char_img.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_AREA)
    result = np.zeros(target_size, dtype=np.uint8)
    start_h = (th - new_h) // 2
    start_w = (tw - new_w) // 2
    result[start_h:start_h+new_h, start_w:start_w+new_w] = resized
    
    return result

# korelacja jako IOU
def simple_correlation(img1, img2):
    #IOU
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    return intersection / union if union > 0 else 0

# funkcja do rozpoznawania znaków
def recognize_character(char_img, templates, show_match=False):
    priority_order = "gw89650m2pabdeqk34hosy7xzcnu?v1tfji!rl,. "
    if char_img.size == 0: return '~', 0.0
    normalized_char = normalize_character(char_img)
    
    best_match = '~'
    best_score = 0.0
    scores = {}
    
    for char in priority_order:
        if char in templates:
            normalized_template = normalize_character(templates[char])
            score = simple_correlation(normalized_char, normalized_template)
            scores[char] = score
            if score > best_score:
                best_score = score
                best_match = char

    if best_score < 0.44: best_match = '~' # rezerwujemy ~ jako znak specjalny
    
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
