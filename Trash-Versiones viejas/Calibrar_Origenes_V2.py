import cv2
import numpy as np

# --- 1. DATOS DE CALIBRACIÓN (¡Actualizados!) ---
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoCompleto.mp4' # Tu ruta guardada
ROI_DISPLAY = [606, 270, 782, 347] 
SCALE_FACTOR = 3 
UMBRAL = 32 # El valor que encontramos
window_name = "PASO 2: Calibrar Origenes (Frame 0)"

# ¡TU NUEVA PLANTILLA (basada en el Frame 0)!
ABS_SEGMENT_MAP = {'a': (208, 49), 'b': (231, 77), 'c': (224, 133), 
                   'd': (192, 153), 'e': (172, 125), 'f': (179, 70), 
                   'g': (200, 99)}
# ---------------------------------------------------------------

# --- 2. CÁLCULO DE LA PLANTILLA (Automático) ---
min_x = min(val[0] for val in ABS_SEGMENT_MAP.values())
min_y = min(val[1] for val in ABS_SEGMENT_MAP.values())
REL_SEGMENT_MAP = {
    key: (val[0] - min_x, val[1] - min_y) 
    for key, val in ABS_SEGMENT_MAP.items()
}
# ---------------------------------------------------------------

# Variables globales
digit_origins = []
current_digit_index = 0
display_img_scaled = None
display_img_clean = None
cap = None
total_frames = 0

def draw_template(image, origin, template, color):
    """ Dibuja la plantilla de 7 segmentos en una ubicación """
    if image is None: return
    ox, oy = origin
    for seg_name, (rx, ry) in template.items():
        abs_x, abs_y = ox + rx, oy + ry
        cv2.circle(image, (abs_x, abs_y), 3, color, -1)

def click_origin(event, x, y, flags, param):
    global current_digit_index, digit_origins, display_img_scaled
    if display_img_clean is None: return

    if event == cv2.EVENT_LBUTTONDOWN and current_digit_index < 3:
        digit_origins.append((x, y))
        print(f"Origen del Dígito {current_digit_index + 1} guardado en: ({x}, {y})")
        current_digit_index += 1
        update_display()

def update_display():
    global display_img_scaled
    if display_img_clean is None: return
        
    display_img_scaled = display_img_clean.copy()

    for i, origin in enumerate(digit_origins):
        draw_template(display_img_scaled, origin, REL_SEGMENT_MAP, (0, 255, 0)) 
        cv2.putText(display_img_scaled, f"Digito {i+1}", (origin[0], origin[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if current_digit_index < 3:
        help_text = f"CLIC en esquina SUPERIOR-IZQUIERDA del DIGITO {current_digit_index + 1}"
        cv2.putText(display_img_scaled, help_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(display_img_scaled, "¡CALIBRACION COMPLETA!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img_scaled, "Pulsa 'c' para confirmar.", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow(window_name, display_img_scaled)

def on_trackbar(val):
    global display_img_clean
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if not ret: return

    x1, y1, x2, y2 = ROI_DISPLAY
    display_img = frame[y1:y2, x1:x2]
    gray_display = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    _, binary_display = cv2.threshold(gray_display, UMBRAL, 255, cv2.THRESH_BINARY)
    binary_display_bgr = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR)
    new_width = int(binary_display_bgr.shape[1] * SCALE_FACTOR)
    new_height = int(binary_display_bgr.shape[0] * SCALE_FACTOR)
    display_img_clean = cv2.resize(binary_display_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    update_display()

# --- Main ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en {video_path}")
    exit()
    
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_origin)
# Creamos el slider, pero no lo necesitamos, así que lo dejamos en 0
cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)

print("--- INSTRUCCIONES (PASO 2 - Frame 0) ---")
print("1. El script se abrirá en el Frame 0 ('6.94').")
print("2. Clic en la esquina SUPERIOR IZQUIERDA del PRIMER dígito (el '6').")
print("3. Clic en la esquina SUPERIOR IZQUIERDA del SEGUNDO dígito (el '9').")
print("4. Clic en la esquina SUPERIOR IZQUIERDA del TERCER dígito (el '4').")
print("   (La plantilla verde debería 'encajar' visualmente)")
print("------------------------------")
print(" 'c': Confirmar (tras los 3 clics).")
print(" 'q': Salir.")
print("------------------------------")

# Cargar el frame 0 por defecto
default_frame = 0
cv2.setTrackbarPos("Frame", window_name, default_frame)
on_trackbar(default_frame)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        if current_digit_index == 3:
            print("\n" + "="*40)
            print("--- ¡CALIBRACIÓN DE ORÍGENES GUARDADA! ---")
            print("¡Esta es la línea final! Pégala en el script 'Analizar_Voltimetro_V2.py':")
            print("\n")
            print(f"DIGIT_ORIGINS = {digit_origins}")
            print("\n")
            print("="*40)
            break
        else:
            print(f"Error: Aún te faltan {3 - current_digit_index} orígenes por marcar.")

cap.release()
cv2.destroyAllWindows()