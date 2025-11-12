import cv2
import numpy as np
import math

# --- Configuración ---
video_path = 'WhatsApp Video 2025-09-29 at 20.17.52.mp4' 
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
puntos_ciclo_actual = {}    # Almacena los puntos para visualización
punto_marcado_en_frame_actual = False
centro_rotacion = (440,170)
estado_actual = "MARCAR_PUNTOS"

# --- Función de Seguimiento Automático (VERSIÓN FINAL: PUNTO DERECHO) ---
def seguimiento_automatico():
    """
    Recorre el video, detecta la estela, y específicamente guarda las coordenadas
    del punto MÁS A LA DERECHA de esa estela, siempre que esté dentro del semicírculo.
    """
    global clicked_points, puntos_ciclo_actual
    
    clicked_points = {}
    puntos_ciclo_actual = {}
    print("\n--- INICIANDO SEGUIMIENTO AUTOMÁTICO (DETECTANDO BORDE DERECHO) ---")

    # Mantenemos el rango de color flexible que funcionó mejor
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([255, 60, 255])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"No se pudo leer el frame {frame_num}. Finalizando.")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        
        # Mantenemos el procesamiento suave de la máscara
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        
        cv2.imshow("Mascara de Deteccion (DEBUG)", mask)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 30:
                
                # --- ### INICIO DEL CAMBIO CLAVE ### ---
                # ANTES: Calculábamos el centro de la estela con momentos.
                # M = cv2.moments(largest_contour)
                # if M["m00"] != 0:
                #     cx = int(M["m10"] / M["m00"])
                #     cy = int(M["m01"] / M["m00"])

                # AHORA: Encontramos el punto del contorno con el valor 'x' máximo.
                # Esto nos da el borde derecho de la estela.
                rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
                cx, cy = rightmost_point
                # --- ### FIN DEL CAMBIO CLAVE ### ---

                # El resto de la lógica de filtrado sigue igual, pero ahora
                # se aplica al punto correcto (el borde derecho).
                distancia_al_centro = math.dist(centro_rotacion, (cx, cy))
                
                if distancia_al_centro <= radio_promedio * 1.2 and cy <= centro_rotacion[1] + 10:
                    clicked_points[frame_num] = (cx, cy)
                    puntos_ciclo_actual[frame_num] = (cx, cy)

        frame_display = frame.copy()
        show_frame_with_markers(frame_display, frame_num)
        cv2.putText(frame_display, f"Procesando: {frame_num}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Visor de video", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("--- Seguimiento automático cancelado por el usuario. ---")
            break

    cv2.destroyWindow("Mascara de Deteccion (DEBUG)")

    print(f"--- SEGUIMIENTO FINALIZADO. Se marcaron {len(clicked_points)} puntos válidos del borde derecho. ---")
    if len(clicked_points) == 0:
        print("ADVERTENCIA: No se detectó ningún punto. Revisa la ventana de depuración para ajustar los colores HSV.")
    print("--- Ahora puedes presionar 'c' para calcular la velocidad. ---")
    on_trackbar(0)

# --- Función para calcular la velocidad angular ---
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
        
        # Asegurarse de que ambos frames existen en el diccionario
        if frame_actual in puntos_guardados and frame_siguiente in puntos_guardados:
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

# --- Función para dibujar todo en el frame ---
def show_frame_with_markers(frame, frame_number):
    if centro_rotacion:
        cv2.circle(frame, centro_rotacion, 7, (0, 255, 0), -1)
        cv2.putText(frame, 'CENTRO', (centro_rotacion[0] + 10, centro_rotacion[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Dibuja el punto correspondiente al frame actual si existe
    if frame_number in puntos_ciclo_actual:
        point = puntos_ciclo_actual[frame_number]
        if centro_rotacion:
            cv2.line(frame, centro_rotacion, point, (255, 255, 0), 1)
        cv2.drawMarker(frame, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    if estado_actual == "DEFINIR_CENTRO":
        cv2.putText(frame, "MODO: Definir Centro. Haga clic para seleccionar.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Visor de video", frame)

# --- Función callback para el click del mouse ---
def mouse_callback(event, x, y, flags, param):
    global current_frame, clicked_points, puntos_ciclo_actual, punto_marcado_en_frame_actual
    global estado_actual, centro_rotacion
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if estado_actual == "DEFINIR_CENTRO":
            centro_rotacion = (x, y)
            estado_actual = "MARCAR_PUNTOS"
            print(f"\n*** Centro de rotación definido en: {centro_rotacion} ***")
            print("--- Modo 'Marcar Puntos' activado. Ya puede marcar objetos o iniciar el modo automático. ---")
        
        elif estado_actual == "MARCAR_PUNTOS":
            if not centro_rotacion:
                print("Error: Primero debe definir un centro. Presione 'r' y haga clic.")
                return

            frame_key = current_frame
            clicked_points[frame_key] = (x, y)
            puntos_ciclo_actual[frame_key] = (x, y)
            punto_marcado_en_frame_actual = True
            print(f"Marca manual añadida en frame {frame_key}: ({x}, {y}).")
                
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            show_frame_with_markers(frame, current_frame)

# --- Función callback para la trackbar ---
def on_trackbar(val):
    global current_frame, punto_marcado_en_frame_actual
    current_frame = val
    punto_marcado_en_frame_actual = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        show_frame_with_markers(frame, current_frame)

# --- Creación de la ventana y controles ---
cv2.namedWindow("Visor de video")
cv2.createTrackbar("Frame", "Visor de video", 0, total_frames - 1, on_trackbar)
cv2.setMouseCallback("Visor de video", mouse_callback)
on_trackbar(0)

# --- Bucle principal de Interacción ---
print("\n--- Controles ---")
print(" 'e': Empezar seguimiento AUTOMÁTICO del punto blanco.")
print("-----------------")
print(" 'a' / 'd': Navegar frames (modo manual).")
print(" Click izquierdo: Marcar un punto (modo manual).")
print(" 'r': Reconfigurar el centro de rotación.")
print(" 'c': Calcular velocidad angular con TODOS los puntos guardados.")
print(" 'l': Limpiar (reiniciar) todos los puntos marcados.")
print(" 'q': Salir.")
print("-----------------\n")

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
        
    elif key == ord('c'):
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
            
    elif key == ord('r'):
        estado_actual = "DEFINIR_CENTRO"
        print("\n--- Modo 'Definir Centro' activado. ---")
        on_trackbar(current_frame)
    
    elif key == ord('l'):
        clicked_points = {}
        puntos_ciclo_actual = {}
        on_trackbar(current_frame)
        print("\n¡Puntos reiniciados! Puede comenzar a marcar de nuevo.")
        
    elif key == ord('e'):
        if centro_rotacion:
            seguimiento_automatico()
        else:
            print("\nError: Primero debe definir un centro de rotación. Presione 'r' y haga clic.")

# --- Cierre de OpenCV ---
cap.release()
cv2.destroyAllWindows()