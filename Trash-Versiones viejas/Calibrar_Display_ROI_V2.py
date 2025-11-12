import cv2
import numpy as np

# --- 1. CONFIGURACIÓN ---
# (Asegúrate de que esta ruta apunte al video ORIGINAL con el voltímetro)
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoCompleto.mp4' # <--- CONFIRMA ESTA RUTA

ROI_DISPLAY = [606, 270, 782, 347] 
SCALE_FACTOR = 3 
window_name = "PASO 1: Calibrar Plantilla (Frame 0)"
UMBRAL = 32 # El valor que encontramos en el diagnóstico

SEGMENT_NAMES = [
    "a (horizontal-superior)",
    "b (vertical-superior-derecha)",
    "c (vertical-inferior-derecha)",
    "d (horizontal-inferior)",
    "e (vertical-inferior-izquierda)",
    "f (vertical-superior-izquierda)",
    "g (horizontal-medio)"
]
# -----------------------------

# Variables globales
segment_centers = []
current_segment_index = 0
display_img_scaled = None
display_img_clean = None
cap = None
total_frames = 0

def click_segment(event, x, y, flags, param):
    global current_segment_index, segment_centers, display_img_scaled
    if display_img_clean is None: return
    
    if event == cv2.EVENT_LBUTTONDOWN and current_segment_index < 7:
        segment_centers.append((x, y))
        print(f"Segmento '{SEGMENT_NAMES[current_segment_index].split(' ')[0]}' guardado en: ({x}, {y})")
        current_segment_index += 1
        update_display()

def update_display():
    global display_img_scaled
    if display_img_clean is None: return
        
    display_img_scaled = display_img_clean.copy()

    for i, (x_c, y_c) in enumerate(segment_centers):
        seg_letter = SEGMENT_NAMES[i].split(' ')[0]
        cv2.circle(display_img_scaled, (x_c, y_c), 5, (0, 255, 0), -1)
        cv2.putText(display_img_scaled, seg_letter, (x_c + 10, y_c + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if current_segment_index < 7:
        help_text = f"CLIC EN: {SEGMENT_NAMES[current_segment_index]}"
        cv2.putText(display_img_scaled, help_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(display_img_scaled, "¡PLANTILLA LISTA!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img_scaled, "Pulsa 'c' para confirmar.", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_img_scaled, "(Pulsa 'r' para resetear)", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name, display_img_scaled)

def on_trackbar(val):
    global display_img_clean, segment_centers, current_segment_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if not ret: return

    x1, y1, x2, y2 = ROI_DISPLAY
    display_img = frame[y1:y2, x1:x2]
    gray_display = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    
    # Usamos el umbral (32) que sabemos que funciona
    _, binary_display = cv2.threshold(gray_display, UMBRAL, 255, cv2.THRESH_BINARY)
    
    binary_display_bgr = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR)
    new_width = int(binary_display_bgr.shape[1] * SCALE_FACTOR)
    new_height = int(binary_display_bgr.shape[0] * SCALE_FACTOR)
    display_img_clean = cv2.resize(binary_display_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Resetea los clics si se mueve el slider
    segment_centers = []
    current_segment_index = 0
    update_display()

# --- Main ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en {video_path}")
    exit()
    
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_segment)
cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)

print("--- INSTRUCCIONES (PASO 1 - Frame 0) ---")
print("1. El script se abrirá en el Frame 0 (deberías ver '6.94' o '5.94').")
print("2. Haz clic PRECISAMENTE EN EL CENTRO de los 7 segmentos (a-g) del PRIMER DÍGITO.")
print("   (Para el segmento 'b' [sup-der] que está APAGADO, haz clic donde DEBERÍA estar).")
print("3. Si te equivocas, presiona 'r' para resetear.")
print("4. Pulsa 'c' para confirmar.")
print("------------------------------")

# Cargar el frame 0 por defecto
default_frame = 0
cv2.setTrackbarPos("Frame", window_name, default_frame)
on_trackbar(default_frame)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'): # Tecla de reseteo
        print("--- RESETEANDO CLICS ---")
        segment_centers = []
        current_segment_index = 0
        update_display()
    elif key == ord('c'):
        if current_segment_index == 7:
            segment_map = {}
            for i, name in enumerate(SEGMENT_NAMES):
                seg_letter = name.split(' ')[0]
                segment_map[seg_letter] = segment_centers[i]
            
            print("\n" + "="*40)
            print("--- ¡PLANTILLA (Frame 0) GUARDADA! ---")
            print("Este es tu NUEVO ABS_SEGMENT_MAP. Pégamelo para el Paso 2:")
            print("\n")
            print(f"ABS_SEGMENT_MAP = {segment_map}")
            print("\n")
            print("="*40)
            break
        else:
            print(f"Error: Aún te faltan {7 - current_segment_index} segmentos por marcar.")

cap.release()
cv2.destroyAllWindows()