import cv2
import numpy as np
import math

# --- Configuración ---
video_path = 'WhatsApp Video 2025-09-29 at 20.17.52.mp4' # Asegúrate que la ruta del video sea correcta
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

# --- Obtención de datos del video ---
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video cargado correctamente.")
print(f"Total de frames: {total_frames}")
print(f"FPS del video: {fps:.2f}")

# --- Variables Globales ---
current_frame = 0
clicked_points = {}         # Almacena TODOS los puntos para el cálculo final
puntos_ciclo_actual = {}    # Almacena solo los puntos del ciclo visual actual
punto_marcado_en_frame_actual = False
# --- CONFIGURACIÓN INICIAL ---
# <<-- CAMBIO: El centro empieza como 'None' y el estado inicial es definirlo
centro_rotacion = (440,170)
estado_actual = "MARCAR_PUNTOSq"
# ---------------------------------

# --- Función para calcular la velocidad angular (SIN CAMBIOS) ---
def calcular_velocidades(puntos_guardados, fps_video, centro):
    if len(puntos_guardados) < 2:
        print("\nError: Se necesitan al menos 2 puntos en total para calcular la velocidad.")
        return None, None
        
    frames_ordenados = sorted(puntos_guardados.keys())
    velocidades_angulares_rad_s = []
    cx, cy = centro

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
        delta_tiempo = delta_frames / fps_video

        if delta_tiempo > 0:
            velocidad_rad_s = delta_angulo / delta_tiempo
            velocidades_angulares_rad_s.append(velocidad_rad_s)

    if not velocidades_angulares_rad_s: 
        return None, None

    velocidad_promedio_rad_s = sum(velocidades_angulares_rad_s) / len(velocidades_angulares_rad_s)
    velocidad_promedio_rpm = velocidad_promedio_rad_s * (60 / (2 * math.pi))
    return velocidad_promedio_rad_s, velocidad_promedio_rpm

# --- Función para dibujar todo en el frame (SIN CAMBIOS) ---
def show_frame_with_markers(frame, frame_number):
    # Dibuja el centro de rotación si ya fue definido
    if centro_rotacion:
        cv2.circle(frame, centro_rotacion, 7, (0, 255, 0), -1) # Círculo verde en el centro
        cv2.putText(frame, 'CENTRO', (centro_rotacion[0] + 10, centro_rotacion[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Dibuja los puntos marcados en el ciclo actual y las líneas hacia el centro
    for fn, point in puntos_ciclo_actual.items():
        if centro_rotacion:
            cv2.line(frame, centro_rotacion, point, (255, 255, 0), 1) # Línea del radio
        cv2.drawMarker(frame, point, (0, 0, 255), 
                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2) # Cruz en el punto
    
    # Muestra un texto de ayuda según el estado actual
    if estado_actual == "DEFINIR_CENTRO":
        cv2.putText(frame, "MODO: Definir Centro. Haga clic para seleccionar el centro.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Visor de video", frame)


# --- Función callback para el click del mouse (MODIFICADA) ---
def mouse_callback(event, x, y, flags, param):
    global current_frame, clicked_points, puntos_ciclo_actual, punto_marcado_en_frame_actual
    global estado_actual, centro_rotacion
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # <<-- CAMBIO: La acción del clic depende del estado actual
        if estado_actual == "DEFINIR_CENTRO":
            centro_rotacion = (x, y)
            estado_actual = "MARCAR_PUNTOS" # Vuelve al modo normal
            print(f"\n*** Centro de rotación definido en: {centro_rotacion} ***")
            print("--- Modo 'Marcar Puntos' activado. Ya puede marcar objetos. ---")
        
        elif estado_actual == "MARCAR_PUNTOS":
            if not centro_rotacion:
                print("Error: Primero debe definir un centro de rotación. Presione 'r' y haga clic.")
                return

            frame_key = current_frame
            clicked_points[frame_key] = (x, y)
            puntos_ciclo_actual[frame_key] = (x, y)
            punto_marcado_en_frame_actual = True
            print(f"Marca añadida en frame {frame_key}: ({x}, {y}). Total guardado: {len(clicked_points)}")
                
        # Refrescamos la imagen para ver el cambio al instante
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            show_frame_with_markers(frame, current_frame)

# --- Función callback para la trackbar (MODIFICADA para refrescar el texto de ayuda) ---
def on_trackbar(val):
    global current_frame, punto_marcado_en_frame_actual
    current_frame = val
    # Cada vez que cambiamos de frame, reiniciamos el flag
    punto_marcado_en_frame_actual = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        show_frame_with_markers(frame, current_frame)

# --- Creación de la ventana y controles ---
cv2.namedWindow("Visor de video")
cv2.createTrackbar("Frame", "Visor de video", 0, total_frames - 1, on_trackbar)
cv2.setMouseCallback("Visor de video", mouse_callback)
on_trackbar(0) # Muestra el primer frame al iniciar

# --- Bucle principal de Interacción (Controles actualizados) ---
print("\n--- Controles ---")
print(" 'r': Reconfigurar el centro de rotación")
print(" 'a' / 'd': Navegar frames")
print(" Click izquierdo: Marcar un punto en el objeto")
print(" 'd' (sin marcar antes): Finaliza y limpia el ciclo visual actual")
print(" 'c': Calcular velocidad angular con TODOS los puntos guardados")
print(" 'l': Limpiar (reiniciar) todos los puntos marcados")
print(" 'q': Salir")
print("-----------------\n")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): 
        break
    
    elif key == ord('d'):
        # Si NO hemos marcado un punto en este frame Y hay puntos visuales que limpiar...
        if not punto_marcado_en_frame_actual and len(puntos_ciclo_actual) > 0:
            puntos_ciclo_actual = {} # Limpiamos los puntos visuales
            print("--- Fin del ciclo de marcado. Puntos visuales limpiados. ---")
            print("--- Puede iniciar el marcado de un nuevo ciclo. ---")
        
        # Avanza al siguiente frame
        current_frame = min(current_frame + 1, total_frames - 1)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)

    elif key == ord('a'):
        # Retrocede al frame anterior
        current_frame = max(current_frame - 1, 0)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)
        
    elif key == ord('c'):
        # Llama a la función de cálculo
        if centro_rotacion:
            rad_s, rpm = calcular_velocidades(clicked_points, fps, centro_rotacion)
            if rad_s is not None and rpm is not None:
                print("\n--- RESULTADO DEL CALCULO FINAL ---")
                print(f"Puntos totales utilizados: {len(clicked_points)}")
                print(f"Velocidad Angular Promedio: {abs(rad_s):.2f} rad/s")
                print(f"Velocidad de Giro Promedio: {abs(rpm):.2f} RPM")
                print("---------------------------------\n")
        else:
            print("\nError: No se ha definido un centro de rotación. Presione 'r' y haga clic.")
            
    # <<-- CAMBIO: Nueva lógica para la tecla 'r'
    elif key == ord('r'):
        estado_actual = "DEFINIR_CENTRO"
        print("\n--- Modo 'Definir Centro' activado. ---")
        print("--- Haga clic en la imagen para seleccionar el nuevo centro de rotación. ---")
        # Refrescamos el frame para mostrar el texto de ayuda
        on_trackbar(current_frame)
    
    # <<-- CAMBIO: Nueva tecla 'l' para la funcionalidad de limpiar
    elif key == ord('l'):
        clicked_points = {}
        puntos_ciclo_actual = {}
        on_trackbar(current_frame) # Redibuja el frame limpio
        print("\n¡Puntos reiniciados! Puede comenzar a marcar de nuevo.")


# --- Cierre de OpenCV ---
cap.release()
cv2.destroyAllWindows()