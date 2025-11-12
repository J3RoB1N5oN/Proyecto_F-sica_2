import cv2
import numpy as np

# --- 1. CONFIGURACIÓN (Pega tus datos aquí) ---
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoCompleto.mp4'
FRAME_DE_PRUEBA = 648 # El frame que muestra "4.04"

# Tus calibraciones
ROI_DISPLAY = [606, 270, 782, 347] 
SCALE_FACTOR = 3 
ABS_SEGMENT_MAP = {'a': (210, 50), 'b': (234, 74), 'c': (224, 132), 
                   'd': (188, 160), 'e': (168, 126), 'f': (181, 68), 
                   'g': (201, 103)}
DIGIT_ORIGINS = [(176, 45), (291, 56), (403, 69)]
# -------------------------------------------------

# --- 2. Procesamiento de plantilla (Automático) ---
min_x = min(val[0] for val in ABS_SEGMENT_MAP.values())
min_y = min(val[1] for val in ABS_SEGMENT_MAP.values())
DIGIT_TEMPLATE_MAP = {
    key: (val[0] - min_x, val[1] - min_y) 
    for key, val in ABS_SEGMENT_MAP.items()
}
# -------------------------------------------------

# Variables globales
gray_display = None
window_name = "Diagnostico de Umbral (Threshold)"

def draw_calibration_points(image, origins, template, color):
    """ Dibuja los 21 puntos de calibración """
    for (ox, oy) in origins:
        for seg_name, (rx, ry) in template.items():
            abs_x, abs_y = ox + rx, oy + ry
            # Dibuja un círculo pequeño y un punto central
            cv2.circle(image, (abs_x, abs_y), 3, color, 1)
            cv2.drawMarker(image, (abs_x, abs_y), color, cv2.MARKER_CROSS, 5, 1)

def on_thresh_change(val):
    global gray_display
    
    # 1. Aplicar el NUEVO umbral
    # val es el valor del slider (0-255)
    _, binary_display = cv2.threshold(gray_display, val, 255, cv2.THRESH_BINARY)
    
    # 2. Escalar y convertir a color para dibujar
    new_width = int(binary_display.shape[1] * SCALE_FACTOR)
    new_height = int(binary_display.shape[0] * SCALE_FACTOR)
    scaled_display_bgr = cv2.cvtColor(
        cv2.resize(binary_display, (new_width, new_height), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR
    )
    
    # 3. Dibujar los puntos de calibración
    draw_calibration_points(scaled_display_bgr, DIGIT_ORIGINS, DIGIT_TEMPLATE_MAP, (0, 0, 255)) # Puntos en ROJO
    
    # 4. Mostrar
    cv2.putText(scaled_display_bgr, f"Threshold: {val}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(window_name, scaled_display_bgr)

# --- Main ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: no se pudo abrir el video en {video_path}")
    exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_DE_PRUEBA)
ret, frame = cap.read()
if not ret:
    print(f"Error al leer el frame {FRAME_DE_PRUEBA}")
    cap.release()
    exit()
cap.release()

# Recortar y convertir a GRIS (sólo una vez)
x1, y1, x2, y2 = ROI_DISPLAY
display_img = frame[y1:y2, x1:x2]
gray_display = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY) # Guardado globalmente

# Crear ventana y slider
cv2.namedWindow(window_name)
# El slider 'Threshold' llamará a 'on_thresh_change'
cv2.createTrackbar("Threshold", window_name, 30, 255, on_thresh_change)

print("--- INSTRUCCIONES (Diagnóstico) ---")
print("1. Ha aparecido una ventana con un slider 'Threshold'.")
print("2. Mueve el slider (empezará en 30, que es el valor que falló).")
print("3. Busca el número que haga que los dígitos '4.04' se vean perfectamente BLANCOS y el fondo perfectamente NEGRO.")
print("4. Los 21 puntos rojos deben caer SOBRE los píxeles blancos.")
print("\n¡DIME QUÉ NÚMERO FUNCIONA MEJOR!")
print("(Pulsa 'q' para salir)")

# Cargar la primera imagen con el valor de 30
on_thresh_change(30)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()