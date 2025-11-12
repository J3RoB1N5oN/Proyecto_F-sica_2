import cv2
import numpy as np

# --- Configuración ---
# Asegúrate de que la ruta del video sea correcta
video_path = 'WhatsApp Video 2025-09-29 at 20.17.52.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total de frames: {total_frames}")

current_frame = 0

# El diccionario ahora guardará un solo punto por frame
clicked_points = {}

# --- Función callback para el click del mouse (MODIFICADA) ---
def mouse_callback(event, x, y, flags, param):
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_key = current_frame
        
        # --- CAMBIO PRINCIPAL ---
        # En lugar de agregar a una lista, ahora sobrescribimos el valor para este frame.
        # Esto asegura que solo el último clic sea guardado.
        clicked_points[frame_key] = (x, y)
        print(f"Marca actualizada en el frame {frame_key} en la posición: ({x}, {y})")

        # VOLVEMOS a posicionar el lector en el frame actual para redibujarlo
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            # Mostramos el MISMO frame pero ahora con la nueva marca incluida
            show_frame_with_markers(frame, current_frame)

# --- Mostrar frame con las marcas (MODIFICADA) ---
def show_frame_with_markers(frame, frame_number):
    # Verificamos si existe una marca para el frame actual
    if frame_number in clicked_points:
        point = clicked_points[frame_number]
        cv2.drawMarker(frame, point, (0, 0, 255), 
                       markerType=cv2.MARKER_CROSS, 
                       markerSize=15, thickness=2) 

    info_text = f"Frame: {frame_number}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Visor de video", frame)

# --- Función callback para la trackbar ---
def on_trackbar(val):
    global current_frame
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        show_frame_with_markers(frame, current_frame)

# --- Ventana, trackbar y callback del mouse ---
cv2.namedWindow("Visor de video")
cv2.createTrackbar("Frame", "Visor de video", 0, total_frames - 1, on_trackbar)
cv2.setMouseCallback("Visor de video", mouse_callback)

# Mostrar primer frame
on_trackbar(0)

# --- Bucle principal ---
while True:
    # Se cambia waitKey(0) a waitKey(1) para un bucle más responsivo
    # y se mantiene la lógica de teclas. Presiona y mantén 'd' o 'a' para avanzar/retroceder.
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('d'): # Avanzar frame
        current_frame = min(current_frame + 1, total_frames - 1)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame) # Llama a la función para actualizar la imagen
    elif key == ord('a'): # Retroceder frame
        current_frame = max(current_frame - 1, 0)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame) # Llama a la función para actualizar la imagen
    elif key == ord('r'): # Reiniciar al frame 0
        current_frame = 0
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)

# Opcional: Imprimir todos los puntos guardados al final
print("\n--- Puntos guardados ---")
for frame, point in sorted(clicked_points.items()):
    print(f"Frame {frame}: {point}")
    
    cap.release()
    cv2.destroyAllWindows()