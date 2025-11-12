import cv2
import numpy as np
import math

# --- Configuración ---
# !! IMPORTANTE: Asegúrate de que la ruta del video sea correcta !!
video_path = 'VideoRecortado.mp4' 
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
clicked_points = {}
puntos_ciclo_actual = {}
punto_marcado_en_frame_actual = False
centro_rotacion = None      # Se definirá con un clic
estado_actual = "DEFINIR_CENTRO" # El primer paso siempre será definir el centro
# Variables para el semicirculo
posicion_radio_izquierda = None
posicion_radio_derecha = None
posicion_radio_superior = None
radio_promedio = 0

# --- Función para calcular y mostrar los radios ---
def calcular_y_mostrar_radios():
    """
    Calcula los radios a partir de los puntos definidos y el centro.
    Usa la distancia euclidiana para mayor precisión.
    """
    global radio_promedio

    if not all([centro_rotacion, posicion_radio_izquierda, posicion_radio_derecha, posicion_radio_superior]):
        print("Advertencia: Faltan puntos para calcular los radios.")
        return

    # Usamos math.dist para calcular la distancia euclidiana (requiere Python 3.8+)
    radio_izquierdo = math.dist(centro_rotacion, posicion_radio_izquierda)
    radio_derecho = math.dist(centro_rotacion, posicion_radio_derecha)
    radio_superior = math.dist(centro_rotacion, posicion_radio_superior)
    
    radio_promedio = (radio_izquierdo + radio_derecho + radio_superior) / 3

    print("\n--- CÁLCULO DE RADIOS ---")
    print(f"Radio Izquierdo: {radio_izquierdo:.2f} pixeles")
    print(f"Radio Derecho:   {radio_derecho:.2f} pixeles")
    print(f"Radio Superior:  {radio_superior:.2f} pixeles")
    print(f"--> Radio Promedio: {radio_promedio:.2f} pixeles")
    print("--------------------------")
    
# --- Función de Seguimiento Automático (VERSIÓN 5: BUSCAR EL CONTORNO MÁS LEJANO) ---
def seguimiento_automatico():
    """
    Recorre el video, busca todos los contornos en el semicírculo superior,
    identifica el contorno MÁS LEJANO del centro (la estela)
    y guarda el punto más a la derecha de ESE contorno.
    """
    global clicked_points, puntos_ciclo_actual
    
    clicked_points = {}
    puntos_ciclo_actual = {}
    print("\n--- INICIANDO SEGUIMIENTO AUTOMÁTICO (BUSCANDO CONTORNO MÁS LEJANO) ---")

    lower_white = np.array([100, 25, 150])
    upper_white = np.array([120, 55, 255])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Filtro geométrico general (bastante permisivo)
    # Lo usamos solo para descartar ruido obvio
    radio_maximo_permitido = radio_promedio * 1.5 
    limite_vertical_permitido = centro_rotacion[1] + 25 

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            
            contornos_candidatos = []
            distancias_de_candidatos = []

            # --- NUEVA LÓGICA V5 ---
            # 1. Pre-filtrar todos los contornos
            for c in contours:
                if cv2.contourArea(c) < 2: # Ignorar ruido
                    continue
                
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                distancia = math.dist(centro_rotacion, (cX, cY))
                
                # Filtro generoso: SÓLO queremos cosas en el semicírculo superior
                # y que no estén absurdamente lejos.
                if distancia <= radio_maximo_permitido and cY <= limite_vertical_permitido:
                    contornos_candidatos.append(c)
                    distancias_de_candidatos.append(distancia)
            
            # 2. Si tenemos contornos candidatos...
            if contornos_candidatos:
                
                # 3. Encontrar el contorno MÁS LEJANO del centro.
                # Esto debe ser la estela, no el reflejo del eje.
                # np.argmax() nos da el índice del valor más alto en la lista de distancias
                indice_mas_lejano = np.argmax(distancias_de_candidatos)
                contorno_estela = contornos_candidatos[indice_mas_lejano]
                
                # 4. Encontrar el punto más a la derecha DE ESE CONTORNO ÚNICAMENTE.
                # Ya no usamos np.concatenate
                rightmost_point = tuple(contorno_estela[contorno_estela[:, :, 0].argmax()][0])
                cx, cy = rightmost_point
                
                # 5. Guardar el punto
                clicked_points[frame_num] = (cx, cy)
                puntos_ciclo_actual[frame_num] = (cx, cy)
                
        # --- FIN DE LA MODIFICACIÓN ---

        frame_display = frame.copy()
        show_frame_with_markers(frame_display, frame_num)
        cv2.putText(frame_display, f"Procesando: {frame_num}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Visor de video", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("--- Seguimiento automático cancelado por el usuario. ---")
            break

    print(f"--- SEGUIMIENTO FINALIZADO. Se marcaron {len(clicked_points)} puntos válidos del borde derecho. ---")
    if len(clicked_points) < 2:
        print("ADVERTENCIA: No se detectaron suficientes puntos. Revisa los filtros o la definición de bordes.")
    else:
        print("--- Ahora puedes presionar 'c' para calcular la velocidad. ---")
    on_trackbar(0)

# --- Función para calcular la velocidad angular (CORREGIDA) ---
def calcular_velocidades(puntos_guardados, fps_video, centro):
    if len(puntos_guardados) < 2:
        print("\nError: Se necesitan al menos 2 puntos en total para calcular la velocidad.")
        return None, None
        
    frames_ordenados = sorted(puntos_guardados.keys())
    
    # --- LÓGICA MEJORADA ---
    # En lugar de promediar las velocidades (sensible al ruido),
    # acumularemos el desplazamiento angular total.
    
    angulo_total_acumulado = 0.0
    cx, cy = centro

    # 1. Iterar por todos los pares de puntos consecutivos
    for i in range(len(frames_ordenados) - 1):
        frame_actual = frames_ordenados[i]
        frame_siguiente = frames_ordenados[i+1]
        
        if frame_actual in puntos_guardados and frame_siguiente in puntos_guardados:
            x1, y1 = puntos_guardados[frame_actual]
            x2, y2 = puntos_guardados[frame_siguiente]
            
            # Calcular ángulo de cada punto
            angulo1 = math.atan2(y1 - cy, x1 - cx)
            angulo2 = math.atan2(y2 - cy, x2 - cx)
            
            # Calcular el delta y corregir el "salto" (unwrapping)
            delta_angulo = angulo2 - angulo1
            
            # Esta es la corrección del "salto" de ángulo (unwrapping)
            if delta_angulo > math.pi:  
                delta_angulo -= 2 * math.pi
            elif delta_angulo < -math.pi: 
                delta_angulo += 2 * math.pi
            
            # Acumular el desplazamiento angular total
            angulo_total_acumulado += delta_angulo
            
            # NOTA: Ya no calculamos la velocidad instantánea aquí

    # 2. Calcular el tiempo total transcurrido
    frame_inicial = frames_ordenados[0]
    frame_final = frames_ordenados[-1] # El último frame en la lista
    delta_frames_total = frame_final - frame_inicial
    
    if delta_frames_total == 0:
        print("\nError: Los puntos están en el mismo frame.")
        return None, None
        
    tiempo_total_segundos = delta_frames_total / fps_video

    if tiempo_total_segundos == 0:
         print(f"\nError: El tiempo total ({tiempo_total_segundos}s) es cero. No se puede calcular la velocidad.")
         return None, None

    # 3. Calcular la velocidad promedio (Desplazamiento Total / Tiempo Total)
    velocidad_promedio_rad_s = angulo_total_acumulado / tiempo_total_segundos
    velocidad_promedio_rpm = velocidad_promedio_rad_s * (60 / (2 * math.pi))
    
    print(f"Info Cálculo: Ángulo total acumulado: {angulo_total_acumulado:.2f} rad")
    print(f"Info Cálculo: Tiempo total: {tiempo_total_segundos:.2f} s")

    return velocidad_promedio_rad_s, velocidad_promedio_rpm

# --- Función para dibujar todo en el frame ---
def show_frame_with_markers(frame, frame_number):
    # Dibuja el centro
    if centro_rotacion:
        cv2.circle(frame, centro_rotacion, 7, (0, 255, 0), -1)
        cv2.putText(frame, 'CENTRO', (centro_rotacion[0] + 10, centro_rotacion[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # --- DIBUJO DEL SEMICÍRCULO Y RADIOS ---
    if posicion_radio_izquierda: cv2.circle(frame, posicion_radio_izquierda, 5, (255, 0, 255), -1)
    if posicion_radio_derecha: cv2.circle(frame, posicion_radio_derecha, 5, (255, 0, 255), -1)
    if posicion_radio_superior: cv2.circle(frame, posicion_radio_superior, 5, (255, 0, 255), -1)
    
    # Si todos los puntos están definidos, dibuja los radios y el semicírculo
    if radio_promedio > 0 and centro_rotacion:
        cv2.line(frame, centro_rotacion, posicion_radio_izquierda, (255, 100, 255), 1)
        cv2.line(frame, centro_rotacion, posicion_radio_derecha, (255, 100, 255), 1)
        cv2.line(frame, centro_rotacion, posicion_radio_superior, (255, 100, 255), 1)
        # Dibuja el semicírculo superior
        cv2.ellipse(frame, centro_rotacion, (int(radio_promedio), int(radio_promedio)), 0, 180, 360, (0, 255, 255), 2)

    # Dibuja el punto de seguimiento del frame actual
    if frame_number in puntos_ciclo_actual:
        point = puntos_ciclo_actual[frame_number]
        if centro_rotacion:
            cv2.line(frame, centro_rotacion, point, (255, 255, 0), 1)
        cv2.drawMarker(frame, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    # --- MENSAJES DE ESTADO ---
    mensaje = ""
    if estado_actual == "DEFINIR_CENTRO":
        mensaje = "PASO 1: Haga clic para definir el CENTRO de rotacion."
    elif estado_actual == "DEFINIR_RADIO_IZQ":
        mensaje = "PASO 2: Haga clic en el borde IZQUIERDO del disco."
    elif estado_actual == "DEFINIR_RADIO_DER":
        mensaje = "PASO 3: Haga clic en el borde DERECHO del disco."
    elif estado_actual == "DEFINIR_RADIO_SUP":
        mensaje = "PASO 4: Haga clic en el borde SUPERIOR del disco."
    else: # MARCAR_PUNTOS
        mensaje = f"Frame: {frame_number}"
    cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Visor de video", frame)

# --- Función callback para el click del mouse ---
def mouse_callback(event, x, y, flags, param):
    global current_frame, clicked_points, puntos_ciclo_actual, punto_marcado_en_frame_actual
    global estado_actual, centro_rotacion, posicion_radio_izquierda, posicion_radio_derecha, posicion_radio_superior
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if estado_actual == "DEFINIR_CENTRO":
            centro_rotacion = (x, y)
            print(f"\n*** Centro de rotación definido en: {centro_rotacion} ***")
            print("--- Ahora presione 'b' para empezar a definir los bordes del disco. ---")
            # No cambiamos de estado aquí, esperamos a que el usuario presione 'b'
        
        elif estado_actual == "DEFINIR_RADIO_IZQ":
            posicion_radio_izquierda = (x, y)
            print(f"Punto izquierdo definido en: {posicion_radio_izquierda}")
            estado_actual = "DEFINIR_RADIO_DER"
        
        elif estado_actual == "DEFINIR_RADIO_DER":
            posicion_radio_derecha = (x, y)
            print(f"Punto derecho definido en: {posicion_radio_derecha}")
            estado_actual = "DEFINIR_RADIO_SUP"
            
        elif estado_actual == "DEFINIR_RADIO_SUP":
            posicion_radio_superior = (x, y)
            print(f"Punto superior definido en: {posicion_radio_superior}")
            calcular_y_mostrar_radios()
            estado_actual = "MARCAR_PUNTOS"
            print("\n--- ¡Listo! Modo 'Marcar Puntos' activado. ---")
            print("Puede marcar puntos manualmente con el clic o iniciar el seguimiento automático con 'e'.")
        
        elif estado_actual == "MARCAR_PUNTOS":
            frame_key = current_frame
            clicked_points[frame_key] = (x, y)
            puntos_ciclo_actual[frame_key] = (x, y)
            punto_marcado_en_frame_actual = True
            print(f"Marca manual añadida en frame {frame_key}: ({x}, {y}).")
    
    # Actualizar la vista después de cualquier clic
    on_trackbar(current_frame)

# --- Función callback para la trackbar ---
def on_trackbar(val):
    global current_frame, punto_marcado_en_frame_actual
    current_frame = val
    punto_marcado_en_frame_actual = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        show_frame_with_markers(frame.copy(), current_frame) # Usar una copia para no alterar el original

# --- Creación de la ventana y controles ---
cv2.namedWindow("Visor de video")
cv2.createTrackbar("Frame", "Visor de video", 0, total_frames - 1, on_trackbar)
cv2.setMouseCallback("Visor de video", mouse_callback)
on_trackbar(0)

# --- Bucle principal de Interacción ---
print("\n--- Controles ---")
print(" 'r': Reconfigurar el centro de rotación.")
print(" 'b': Definir/redefinir los bordes del disco (después de fijar el centro).")
print("-----------------")
print(" 'e': Empezar seguimiento AUTOMÁTICO del punto blanco.")
print(" Click izquierdo: Marcar un punto (en modo de marcado).")
print(" 'a' / 'd': Navegar frames.")
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
        if centro_rotacion and clicked_points:
            rad_s, rpm = calcular_velocidades(clicked_points, fps, centro_rotacion)
            if rad_s is not None and rpm is not None:
                print("\n--- RESULTADO DEL CALCULO FINAL ---")
                print(f"Puntos totales utilizados: {len(clicked_points)}")
                print(f"Velocidad Angular Promedio: {abs(rad_s):.2f} rad/s")
                print(f"Velocidad de Giro Promedio: {abs(rpm):.2f} RPM")
                print("---------------------------------\n")
        else:
            print("\nError: Debe definir un centro y marcar al menos dos puntos antes de calcular.")
            
    elif key == ord('r'):
        estado_actual = "DEFINIR_CENTRO"
        centro_rotacion = None
        print("\n--- Modo 'Definir Centro' activado. ---")
        on_trackbar(current_frame)
    
    elif key == ord('b'):
        if not centro_rotacion:
            print("\nError: Primero debe definir un centro. Haga clic en el video.")
        else:
            posicion_radio_izquierda = None
            posicion_radio_derecha = None
            posicion_radio_superior = None
            radio_promedio = 0
            estado_actual = "DEFINIR_RADIO_IZQ"
            print("\n--- Modo 'Definir Bordes' activado. ---")
            on_trackbar(current_frame)

    elif key == ord('l'):
        clicked_points = {}
        puntos_ciclo_actual = {}
        on_trackbar(current_frame)
        print("\n¡Puntos de seguimiento reiniciados! Puede comenzar a marcar de nuevo.")
        
    elif key == ord('e'):
        if centro_rotacion and radio_promedio > 0:
            seguimiento_automatico()
        else:
            print("\nError: Primero debe definir un centro ('r') y los bordes del disco ('b').")

# --- Cierre de OpenCV ---
cap.release()
cv2.destroyAllWindows()