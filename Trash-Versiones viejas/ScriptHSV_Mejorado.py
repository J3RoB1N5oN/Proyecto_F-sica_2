import cv2
import numpy as np

# --- CONFIGURACIÓN ---
video_path = 'g:/Otros ordenadores/Asus Laptop/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
window_name = "Segmentador Interactivo (Flood Fill)"

initial_tolerance = 40

# Cargar el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: no se pudo abrir el video en {video_path}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# --- Función para el clic del mouse (MODIFICADA CON INFO DE COLOR) ---
def on_click_floodfill(event, x, y, flags, param):
    """
    Esta función se activa al hacer clic.
    1. Imprime la información de color (BGR y HSV) del píxel 'semilla'.
    2. Usa floodFill para segmentar la región conectada al clic.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param['frame']
        
        # --- ¡NUEVO! OBTENER Y MOSTRAR INFO DEL PÍXEL ---
        current_frame_num = cv2.getTrackbarPos("Frame", window_name)
        
        # 1. Obtener BGR del píxel exacto
        bgr_color = frame[y, x]
        # Convertimos a int para una impresión más limpia
        b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
        
        # 2. Convertir ese píxel a HSV
        #    (Necesita un array de 3 dimensiones para cvtColor)
        hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])

        print("====================================")
        print(f"** Píxel Semilla Seleccionado **")
        print(f"Frame:    {current_frame_num}")
        print(f"Coords:   (x={x}, y={y})")
        print(f"Color BGR: [B={b}, G={g}, R={r}]")
        print(f"Color HSV: [H={h}, S={s}, V={v}]  <-- ¡ESTOS SON TUS VALORES!")
        print("------------------------------------")
        
        # --- LÓGICA DE FLOOD FILL (igual que antes) ---
        
        frame_copy = frame.copy()
        
        try:
            tolerance = cv2.getTrackbarPos("Tolerancia", window_name)
        except:
            tolerance = initial_tolerance

        diff = (tolerance, tolerance, tolerance)
        mask = np.zeros((height + 2, width + 2), np.uint8)

        print(f"Ejecutando Flood Fill con Tolerancia={tolerance}...")
        
        cv2.floodFill(frame_copy, mask, (x, y), (0, 255, 0), diff, diff, cv2.FLOODFILL_FIXED_RANGE)

        cv2.imshow(window_name, frame_copy)
        print("¡Segmentación completada! Mueve el frame ('a'/'d') para resetear.")
        print("====================================")


# --- Función para la barra de desplazamiento "Frame" ---
def on_trackbar(val):
    """
    Se ejecuta al mover la barra 'Frame' o presionar 'a'/'d'.
    Muestra el frame original y lo prepara para el clic.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, on_click_floodfill, {'frame': frame})

# --- Función placeholder para la barra "Tolerancia" ---
def on_tolerance_trackbar(val):
    pass

# --- Configuración de la ventana y ejecución ---
cv2.namedWindow(window_name)

cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)
cv2.createTrackbar("Tolerancia", window_name, initial_tolerance, 255, on_tolerance_trackbar)

print("\n--- INSTRUCCIONES (Versión Flood Fill + Info) ---")
print("1. Usa 'a' (atrás) y 'd' (adelante) para navegar los frames.")
print("2. Ajusta la barra 'Tolerancia'.")
print("3. Haz clic en la mancha blanca.")
print("4. El script rellenará la mancha Y TE MOSTRARÁ EN CONSOLA los valores HSV del píxel que clickeaste.")
print("5. Anota los valores HSV y la Tolerancia que mejor funcionen.")
print("6. Presiona 'q' para salir.")

# Mostrar el primer frame
on_trackbar(0)

# Bucle principal
while True:
    current_frame = cv2.getTrackbarPos("Frame", window_name)
    
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    
    elif key == ord('a'): # Retroceder
        new_frame = max(current_frame - 1, 0)
        cv2.setTrackbarPos("Frame", window_name, new_frame)
        on_trackbar(new_frame)
        
    elif key == ord('d'): # Avanzar
        new_frame = min(current_frame + 1, total_frames - 1)
        cv2.setTrackbarPos("Frame", window_name, new_frame)
        on_trackbar(new_frame)

cap.release()
cv2.destroyAllWindows()