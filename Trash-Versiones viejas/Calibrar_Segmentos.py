import cv2
import numpy as np

# --- Configuración de Fase 2 ---

# ¡DATO DE FASE 1! Esta es la ROI que me pasaste.
ROI_DISPLAY = [606, 270, 782, 347] 

# Ruta del video
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoCompleto.mp4'

# Cuánto vamos a ampliar la imagen de la ROI para poder hacer clic
SCALE_FACTOR = 3 

# El orden en que calibraremos
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
display_img_clean = None # Para resetear

# --- Callback del Ratón ---
def click_segment(event, x, y, flags, param):
    global current_segment_index, segment_centers, display_img_scaled

    if event == cv2.EVENT_LBUTTONDOWN and current_segment_index < 7:
        # Guardamos la coordenada del clic
        segment_centers.append((x, y))
        print(f"Segmento '{SEGMENT_NAMES[current_segment_index].split(' ')[0]}' guardado en: ({x}, {y})")
        
        current_segment_index += 1
        
        # Actualizar la imagen con los puntos y el texto
        update_display()

def update_display():
    """ Dibuja los puntos y el texto de ayuda sobre la imagen """
    global display_img_scaled
    
    # Reseteamos al último estado limpio
    display_img_scaled = display_img_clean.copy()

    # Dibujar todos los puntos ya clickeados
    for i, (x_c, y_c) in enumerate(segment_centers):
        # La letra del segmento (a, b, c...)
        seg_letter = SEGMENT_NAMES[i].split(' ')[0]
        cv2.circle(display_img_scaled, (x_c, y_c), 5, (0, 255, 0), -1)
        cv2.putText(display_img_scaled, seg_letter, (x_c + 10, y_c + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Preparar el texto de ayuda
    if current_segment_index < 7:
        help_text = f"CLIC EN: {SEGMENT_NAMES[current_segment_index]}"
        cv2.putText(display_img_scaled, help_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # ¡Terminado!
        help_text_1 = "¡CALIBRACION COMPLETA!"
        help_text_2 = "Pulsa 'c' para confirmar y ver el resultado."
        help_text_3 = "Pulsa 'r' para resetear."
        cv2.putText(display_img_scaled, help_text_1, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img_scaled, help_text_2, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_img_scaled, help_text_3, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Calibrar Segmentos", display_img_scaled)

# --- Main ---
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Error al leer el video.")
    cap.release()
    exit()
cap.release()

# 1. Recortar la ROI del display
x1, y1, x2, y2 = ROI_DISPLAY
display_img = frame[y1:y2, x1:x2]

# 2. Pre-procesar (Gris + Umbral + Zoom)
gray_display = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
# Binarizar: todo lo que no es negro (0) se vuelve blanco (255)
# Esto limpia la imagen y deja solo los segmentos encendidos
_, binary_display = cv2.threshold(gray_display, 30, 255, cv2.THRESH_BINARY)
# Volver a BGR para dibujar en color
binary_display_bgr = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR)

# 3. Escalar para que sea fácil hacer clic
new_width = int(binary_display_bgr.shape[1] * SCALE_FACTOR)
new_height = int(binary_display_bgr.shape[0] * SCALE_FACTOR)
display_img_scaled = cv2.resize(binary_display_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
display_img_clean = display_img_scaled.copy() # Guardamos copia limpia

# 4. Configurar ventana y callback
cv2.namedWindow("Calibrar Segmentos")
cv2.setMouseCallback("Calibrar Segmentos", click_segment)

print("--- INSTRUCCIONES (Fase 2) ---")
print("Ha aparecido una ventana ampliada del display.")
print("Debes hacer clic en el CENTRO de cada uno de los 7 segmentos.")
print("Sigue el orden que te pide la ventana (a, b, c, d, e, f, g).")
print(f"Referencia: {SEGMENT_NAMES}")
print("---")
print(" 'r': Resetear todos los puntos y empezar de nuevo.")
print(" 'c': Confirmar (SOLO cuando hayas marcado los 7 puntos).")
print(" 'q': Salir.")
print("------------------------------")

# 5. Iniciar la primera actualización de pantalla
update_display()

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):
        print("--- RESETEANDO ---")
        segment_centers = []
        current_segment_index = 0
        update_display()
        print("Puntos reseteados. Empieza de nuevo desde 'a'.")
    
    elif key == ord('c'):
        if current_segment_index == 7:
            # ¡Éxito! Imprimir el diccionario de Python
            
            # Mapeamos los nombres (a,b,c...) a las coordenadas (x,y)
            segment_map = {}
            for i, name in enumerate(SEGMENT_NAMES):
                seg_letter = name.split(' ')[0] # Quedarnos solo con 'a', 'b', etc.
                segment_map[seg_letter] = segment_centers[i]
            
            print("\n" + "="*40)
            print("--- ¡CALIBRACIÓN DE SEGMENTOS GUARDADA! ---")
            print("Pégame este diccionario en el chat para la Fase 3:")
            print("\n")
            print(f"SCALE_FACTOR = {SCALE_FACTOR}")
            print(f"SEGMENT_MAP = {segment_map}")
            print("\n")
            print("="*40)
            break
        else:
            print(f"Error: Aún te faltan {7 - current_segment_index} segmentos por marcar.")

cv2.destroyAllWindows()