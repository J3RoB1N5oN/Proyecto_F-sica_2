import cv2
import numpy as np
import math

# --- Configuración ---
video_path = 'WhatsApp Video 2025-09-29 at 20.17.52.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

# --- CORRECCIÓN: Obtenemos todos los datos del video aquí, al principio ---
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) # <-- MOVIMOS ESTA LÍNEA AQUÍ
# --------------------------------------------------------------------

print(f"Total de frames: {total_frames}")
print(f"FPS del video: {fps}")

current_frame = 0
clicked_points = {}

# --- Función callback para el click del mouse ---
def mouse_callback(event, x, y, flags, param):
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_key = current_frame
        clicked_points[frame_key] = (x, y)
        print(f"Marca actualizada en el frame {frame_key} en la posición: ({x}, {y})")
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            show_frame_with_markers(frame, current_frame)

# --- Mostrar frame con las marcas ---
def show_frame_with_markers(frame, frame_number):
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
on_trackbar(0)

# --- Bucle principal ---
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        current_frame = min(current_frame + 1, total_frames - 1)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)
    elif key == ord('a'):
        current_frame = max(current_frame - 1, 0)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)
    elif key == ord('r'):
        current_frame = 0
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)

# --- Cierre de OpenCV ---
cap.release() # Ahora sí podemos cerrar el video, ya tenemos los FPS
cv2.destroyAllWindows()


# --- Sección de Cálculo (Función de la respuesta anterior) ---
def calcular_velocidad_angular(puntos_guardados, fps, centro_rotacion):
    if len(puntos_guardados) < 2:
        print("Se necesitan al menos 2 puntos para calcular la velocidad.")
        return None
    frames_ordenados = sorted(puntos_guardados.keys())
    velocidades_angulares = []
    cx, cy = centro_rotacion
    for i in range(len(frames_ordenados) - 1):
        frame_actual = frames_ordenados[i]
        frame_siguiente = frames_ordenados[i+1]
        x1, y1 = puntos_guardados[frame_actual]
        x2, y2 = puntos_guardados[frame_siguiente]
        angulo1 = math.atan2(y1 - cy, x1 - cx)
        angulo2 = math.atan2(y2 - cy, x2 - cx)
        delta_angulo = angulo2 - angulo1
        if delta_angulo > math.pi:
            delta_angulo -= 2 * math.pi
        elif delta_angulo < -math.pi:
            delta_angulo += 2 * math.pi
        delta_frames = frame_siguiente - frame_actual
        delta_tiempo = delta_frames / fps
        if delta_tiempo > 0:
            velocidad_rad_s = delta_angulo / delta_tiempo
            velocidades_angulares.append(velocidad_rad_s)
    if not velocidades_angulares:
        return None
    velocidad_promedio_rad_s = sum(velocidades_angulares) / len(velocidades_angulares)
    velocidad_promedio_rpm = velocidad_promedio_rad_s * (60 / (2 * math.pi))
    return velocidad_promedio_rpm

# --- Llamada a la función de cálculo ---
print("\n--- Puntos guardados ---")
# Ordenar el diccionario por frame para una visualización limpia
puntos_ordenados = sorted(clicked_points.items())
for frame, point in puntos_ordenados:
    print(f"Frame {frame}: {point}")

# **IMPORTANTE**: Define el centro de rotación del disco.
# Por ejemplo, si el centro está en las coordenadas (400, 300)
CENTRO_DEL_DISCO = (443, 173) 

velocidad_rpm = calcular_velocidad_angular(clicked_points, fps, CENTRO_DEL_DISCO)

if velocidad_rpm is not None:
    print(f"\nLa velocidad de giro del disco es: {abs(velocidad_rpm):.2f} RPM")