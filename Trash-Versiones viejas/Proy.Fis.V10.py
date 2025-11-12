import cv2
import numpy as np
import math

# --- Configuración ---
# !! IMPORTANTE: Asegúrate de que la ruta del video sea correcta !!
#Local:
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
#Drive:
# video_path = 'g:/Otros ordenadores/Asus Laptop/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
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
current_frame_image = None
clicked_points = {}
puntos_ciclo_actual = {}
punto_marcado_en_frame_actual = False
punto_previo_seguimiento = None

# --- CONFIGURACIÓN GEOMÉTRICA ---

# 1. Definir el estado INICIAL (por defecto, si la precarga falla)
centro_rotacion = None
posicion_radio_izquierda = None
posicion_radio_derecha = None
posicion_radio_superior = None
radio_promedio = 0
# Estado por defecto si la precarga falla:
estado_actual = "DEFINIR_CENTRO" 

# 2. Valores de PRECARGA (Valores fijos de tu script)
centro_precargado = (285, 285)
radio_izq_precargado = (75, 285)
radio_der_precargado = (480, 285)
radio_sup_precargado = (285, 55)

# 3. Intentar aplicar la precarga
try:
    # Calcular el radio promedio DE LOS VALORES PRECARGADOS
    radio_izq = math.dist(centro_precargado, radio_izq_precargado)
    radio_der = math.dist(centro_precargado, radio_der_precargado)
    radio_sup = math.dist(centro_precargado, radio_sup_precargado)
    radio_prom_calculado = (radio_izq + radio_der + radio_sup) / 3

    # 4. Si el cálculo es válido, SOBREESCRIBIR las variables globales
    if radio_prom_calculado > 0:
        centro_rotacion = centro_precargado
        posicion_radio_izquierda = radio_izq_precargado
        posicion_radio_derecha = radio_der_precargado
        posicion_radio_superior = radio_sup_precargado
        radio_promedio = radio_prom_calculado
        
        # --- ¡CAMBIO CLAVE! ---
        # El estado inicial ahora permite definir el color directamente.
        estado_actual = "DEFINIR_COLOR" 

        print("\n--- CONFIGURACIÓN GEOMÉTRICA PRECARGADA ---")
        print(f"Centro: {centro_rotacion}")
        print(f"Radio Promedio: {radio_promedio:.2f} pixeles")
        print("------------------------------------------")
        print("\n¡Geometría lista! Presiona 'k' para definir el color.")
        print("O presiona 'r' para redefinir la geometría manualmente.")
    
    else:
        # El cálculo dio <= 0
        print("Advertencia: Los valores precargados no son válidos. Defina el centro.")
        
except Exception as e:
    # Error en el cálculo (p.ej. uno de los valores era None si se borraron)
    print(f"Valores precargados no encontrados o inválidos ({e}).")
    print("Iniciando en modo manual: 'DEFINIR_CENTRO'.")

# --- FIN DE CONFIGURACIÓN ---

# --- Variables para el Selector de Color ---
lower_white = None
upper_white = None
initial_tolerance = 35  # Tolerancia por defecto
HUE_MARGIN = 10         # Margen para el Matiz (H)

# --- Función para calcular y mostrar los radios ---
def calcular_y_mostrar_radios():
    global radio_promedio
    if not all([centro_rotacion, posicion_radio_izquierda, posicion_radio_derecha, posicion_radio_superior]):
        print("Advertencia: Faltan puntos para calcular los radios.")
        return

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
    
# --- Función de Seguimiento Automático (VERSIÓN V10-PX-Strict: Proximidad + Limpieza + Anti-Salto) ---
def seguimiento_automatico():
    """
    (Versión V10-PX-Strict)
    
    Igual que V10-PX-Clean, pero añade una lógica "estricta"
    para prevenir el "salto" de seguimiento al otro punto
    cuando el punto actual es ocluido.
    """
    global clicked_points, puntos_ciclo_actual, lower_white, upper_white
    global punto_previo_seguimiento 
    
    if lower_white is None or upper_white is None:
        print("\n¡Error! Primero debe definir el rango de color ('k').")
        on_trackbar(current_frame) 
        return
    
    print(f"\n--- INICIANDO SEGUIMIENTO (V10-PX-Strict) con rango: {lower_white} a {upper_white} ---")
    print("   (Presione 'p' para pausar, 'q' para cancelar)")
    
    # --- RESETEO ---
    clicked_points = {}
    puntos_ciclo_actual = {}
    punto_previo_seguimiento = None 
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    delay_ms = int(500 / fps)
    if delay_ms <= 0:
        delay_ms = 1

    # Definición de límites geométricos
    radio_minimo_permitido = radio_promedio * 0.7 
    radio_maximo_permitido = radio_promedio * 1.10
    limite_vertical_permitido = centro_rotacion[1] + 25 
    area_minima_ruido = 10 
    
    # Umbral de Salto:
    # Si un punto está más lejos que esto (en píxeles) del punto previo,
    # se asume que es el OTRO punto y se ignora.
    # Un radio promedio es ~200-250px, así que 150px es un salto inter-frame
    # razonablemente grande pero menor que el salto al otro lado (~400-500px).
    MAX_JUMP_DISTANCIA = 150 

    # Kernel de limpieza (sin cambios)
    kernel_limpieza = np.ones((3,3), np.uint8)
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        
        # Limpieza morfológica (sin cambios)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_limpieza, iterations=2)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- SECCIÓN DE LÓGICA (V10-PX-Strict, MODIFICADA) ---
        
        centroides_candidatos = []
        
        if contours:
            # 1. Iterar TODOS los contornos y convertirlos en centroides válidos (sin cambios)
            for c in contours:
                if cv2.contourArea(c) < area_minima_ruido:
                    continue
                
                x, y, w, h = cv2.boundingRect(c)
                cY = y + h / 2
                if cY > limite_vertical_permitido: 
                    continue

                M = cv2.moments(c)
                if M["m00"] <= 0:
                    continue
                    
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                distancia_al_centro = math.dist(centro_rotacion, (cx, cy))
                if (distancia_al_centro < radio_minimo_permitido or
                    distancia_al_centro > radio_maximo_permitido):
                    continue
                    
                centroides_candidatos.append((cx, cy))

        # 2. Lógica de Decisión Estricta (con memoria)
        punto_elegido = None
        
        if not centroides_candidatos:
            # No se detectó nada, no hacer nada
            pass
            
        elif len(centroides_candidatos) == 1:
            # Solo se detectó un punto
            punto_candidato = centroides_candidatos[0]
            
            if punto_previo_seguimiento is None:
                # Es el primer punto detectado en el video, aceptarlo
                punto_elegido = punto_candidato
            else:
                # Hay memoria. Comprobar si este punto está "cerca" del anterior
                dist_salto = math.dist(punto_candidato, punto_previo_seguimiento)
                
                if dist_salto > MAX_JUMP_DISTANCIA:
                    # Al no hacer nada, 'punto_elegido' sigue siendo None
                    # y 'punto_previo_seguimiento' NO se actualiza.
                    pass 
                else:
                    # Es un salto pequeño, es el mismo punto. Aceptarlo.
                    punto_elegido = punto_candidato
                    
        else: # len > 1 (Múltiples puntos detectados)
        
            if punto_previo_seguimiento is None:
                # Primera vez, elegir el más izquierdo por defecto
                punto_elegido = min(centroides_candidatos, key=lambda p: p[0])
            else:
                # Tenemos memoria. Encontrar el punto más cercano AL PUNTO PREVIO
                # PERO que también esté dentro del umbral de salto.
                
                candidatos_validos = []
                for p_cand in centroides_candidatos:
                    dist = math.dist(p_cand, punto_previo_seguimiento)
                    if dist <= MAX_JUMP_DISTANCIA:
                        candidatos_validos.append( (dist, p_cand) ) # Tupla (distancia, punto)
                
                if candidatos_validos:
                    # Elegir el más cercano (distancia mínima) de los válidos
                    punto_elegido = min(candidatos_validos, key=lambda t: t[0])[1]
                else:
                    # Raro: hay múltiples puntos, pero ninguno está "cerca"
                    # Se perdió el rastro. No elegir ninguno.
                    pass

        # 3. Guardar el punto y actualizar la memoria
        # !MODIFICADO! Solo se actualiza la memoria si se ELIGIÓ un punto.
        if punto_elegido:
            clicked_points[frame_num] = punto_elegido
            puntos_ciclo_actual[frame_num] = punto_elegido
            # Actualizar la memoria SOLO con el último punto VÁLIDO
            punto_previo_seguimiento = punto_elegido 
        
        # --- Fin de la sección de lógica ---

        # (El resto de la función: display, 'q', 'p', etc. sigue igual)
        frame_display = frame.copy()
        show_frame_with_markers(frame_display, frame_num)
        
        cv2.putText(frame_display, f"Procesando: {frame_num}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_display, "Presione 'p' para PAUSA", (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Visor de video", frame_display)

        key = cv2.waitKey(delay_ms) & 0xFF
        
        if key == ord('q'): 
            print("--- Seguimiento automático cancelado. ---")
            break
        elif key == ord('p'):
            print(f"\n--- PAUSA en frame {frame_num}. Presione cualquier tecla para continuar... ---")
            while True:
                key_pause = cv2.waitKey(0) & 0xFF
                if key_pause:
                    print("--- Reanudando... ---")
                    break 

    print(f"--- SEGUIMIENTO FINALIZADO. Se marcaron {len(clicked_points)} puntos. ---")
    on_trackbar(0)

# --- Función de Cálculo (V5 - Polyfit con unwrapping y diagnóstico) ---
def calcular_velocidades_polyfit(puntos_a_calcular, fps_video):
    """
    (Versión V5, corregida para asegurar la conversión de unidades)
    Calcula la velocidad angular (w) y RPM usando regresión lineal (Polyfit)
    sobre los ángulos unwrapped.
    """
    
    if not puntos_a_calcular:
        print("\nError Cálculo: No hay puntos para analizar.")
        return None, None
        
    # Convertir el diccionario {frame: (x, y)} a dos listas separadas
    frames_num, coords = zip(*sorted(puntos_a_calcular.items()))

    # --- 1. CONVERSIÓN A COORDENADAS POLARES ---
    # Convertir (x, y) a ángulos. 'tiempos' está en segundos.
    tiempos_s = np.array(frames_num) / fps_video
    
    angles_rad = np.array([
        math.atan2(-(y - centro_rotacion[1]), x - centro_rotacion[0]) 
        for (x, y) in coords
    ])
    
    # 'np.unwrap' es CRÍTICO. Corrige los saltos de 2*pi (ej. de 359° a 1°).
    angles_rad_unwrapped = np.unwrap(angles_rad)

    # --- 2. DIAGNÓSTICO ---
    # (Mantenemos el diagnóstico original)
    angles_std_dev = np.std(angles_rad)
    if angles_std_dev < 0.1: 
        print("\n--- ¡DIAGNÓSTICO DE ERROR! ---")
        print(f"Error: La desviación estándar de los ángulos es casi cero ({angles_std_dev:.4f}).")
        print("Esto casi con seguridad significa que el tracking está 'clavado' en el centro de rotación.")
        print("O todos los puntos están en la misma línea recta desde el centro.")
        print("El cálculo de RPM no es posible. Intente un nuevo tracking ('e').")
        print("---------------------------------")
        return None, None
    else:
        print(f"Info Cálculo: La desviación estándar de los ángulos es {angles_std_dev:.4f} (OK)")

    # --- 3. REGRESIÓN LINEAL (Cálculo de Pendiente) ---
    # Ajustamos la línea a: angulo_unwrapped = f(tiempo_en_segundos)
    # model[0] (slope) será la velocidad angular DIRECTAMENTE.
    # Usamos 'tiempos_s' (en segundos) en el eje X.
    # NO 'frames_num' (en frames).
    model = np.polyfit(tiempos_s, angles_rad_unwrapped, 1)
    
    # La pendiente (model[0]) AHORA SÍ es 'rad/s'
    omega_promedio_rad_s = model[0]
    
    # (Opcional) Si quieres ver la pendiente por frame (como antes):
    # model_frames = np.polyfit(np.array(frames_num), angles_rad_unwrapped, 1)
    # slope_rad_por_frame = model_frames[0]
    # print(f"  -> Slope (w) calculado: {slope_rad_por_frame:.4f} rad/frame")

    print(f"  -> Slope (w) calculado: {omega_promedio_rad_s:.4f} rad/s (basado en {len(tiempos_s)} puntos)")

    # --- 4. CALCULAR RESULTADOS FINALES ---
    # T = 2*pi / |w|
    if abs(omega_promedio_rad_s) < 1e-6: # Evitar división por cero
        T_estimado = float('inf')
        rpm_promedio = 0.0
    else:
        T_estimado = (2 * np.pi) / abs(omega_promedio_rad_s)
        # RPM = w * (60 / 2*pi)
        rpm_promedio = omega_promedio_rad_s * (60 / (2 * np.pi))

    print(f"Info Cálculo: Período completo (T) estimado: {T_estimado:.4f} s")

    return abs(omega_promedio_rad_s), abs(rpm_promedio)

# --- Función para dibujar todo en el frame (Con nuevos estados) ---
def show_frame_with_markers(frame, frame_number):
    # Dibuja el centro
    if centro_rotacion:
        cv2.circle(frame, centro_rotacion, 7, (0, 255, 0), -1)
        cv2.putText(frame, 'CENTRO', (centro_rotacion[0] + 10, centro_rotacion[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(frame, (0, centro_rotacion[1]), (frame.shape[1], centro_rotacion[1]), (255, 0, 255), 1)
        cv2.line(frame, (centro_rotacion[0], 0), (centro_rotacion[0], frame.shape[0]), (255, 0, 255), 1)
    
    # Dibuja radios
    if posicion_radio_izquierda: cv2.circle(frame, posicion_radio_izquierda, 5, (255, 0, 255), -1)
    if posicion_radio_derecha: cv2.circle(frame, posicion_radio_derecha, 5, (255, 0, 255), -1)
    if posicion_radio_superior: cv2.circle(frame, posicion_radio_superior, 5, (255, 0, 255), -1)
    
    # Dibuja semicírculo
    if radio_promedio > 0 and centro_rotacion:
        cv2.line(frame, centro_rotacion, posicion_radio_izquierda, (255, 100, 255), 1)
        cv2.line(frame, centro_rotacion, posicion_radio_derecha, (255, 100, 255), 1)
        cv2.line(frame, centro_rotacion, posicion_radio_superior, (255, 100, 255), 1)
        radio_dibujo = int(radio_promedio)
        cv2.ellipse(frame, centro_rotacion, (radio_dibujo, radio_dibujo), 0, 180, 360, (0, 255, 255), 2)

    # Dibuja punto de seguimiento
    if frame_number in puntos_ciclo_actual:
        point = puntos_ciclo_actual[frame_number]
        if centro_rotacion:
            cv2.line(frame, centro_rotacion, point, (255, 255, 0), 1)
        cv2.drawMarker(frame, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    # Mensajes de estado
    mensaje = ""
    if estado_actual == "DEFINIR_CENTRO":
        mensaje = "PASO 1: Haga clic para definir el CENTRO de rotacion."
    elif estado_actual == "DEFINIR_RADIO_IZQ":
        mensaje = "PASO 2: Haga clic en el borde IZQUIERDO del disco."
    elif estado_actual == "DEFINIR_RADIO_DER":
        mensaje = "PASO 3: Haga clic en el borde DERECHO del disco."
    elif estado_actual == "DEFINIR_RADIO_SUP":
        mensaje = "PASO 4: Haga clic en el borde SUPERIOR del disco."
    elif estado_actual == "DEFINIR_COLOR":
        mensaje = "PASO 5: Ajuste 'Tolerancia' y HAGA CLIC en la mancha blanca."
    else: # MARCAR_PUNTOS
        mensaje = f"Modo Marcado. Frame: {frame_number}. (Use 'k' para definir color)"
        
    cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Visor de video", frame)

# --- Función callback para el click del mouse (Con estado DEFINIR_COLOR) ---
def mouse_callback(event, x, y, flags, param):
    global current_frame, clicked_points, puntos_ciclo_actual, punto_marcado_en_frame_actual
    global estado_actual, centro_rotacion, posicion_radio_izquierda, posicion_radio_derecha, posicion_radio_superior
    global current_frame_image, lower_white, upper_white
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if estado_actual == "DEFINIR_CENTRO":
            centro_rotacion = (x, y)
            print(f"\n*** Centro de rotación definido en: {centro_rotacion} ***")
            print("--- Ahora presione 'b' para empezar a definir los bordes. ---")
            
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
            estado_actual = "MARCAR_PUNTOS" # Pasa a marcar puntos por defecto
            print("\n--- ¡Radios listos! ---")
            print("PASO SIGUIENTE: Presione 'k' para definir el color de la mancha.")
        
        elif estado_actual == "DEFINIR_COLOR":
            if current_frame_image is None:
                return
            
            frame = current_frame_image.copy()
            height, width, _ = frame.shape
            
            # 1. Info del Píxel Semilla
            bgr_color = frame[y, x]
            b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
            hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h_seed, s_seed, v_seed = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])

            print("====================================")
            print(f"** Píxel Semilla Seleccionado (Frame {current_frame}) **")
            print(f"Color HSV: [H={h_seed}, S={s_seed}, V={v_seed}]")

            # 2. Lógica de Flood Fill
            frame_copy = frame.copy()
            tolerance = cv2.getTrackbarPos("Tolerancia", "Visor de video")
            diff = (tolerance, tolerance, tolerance)
            mask = np.zeros((height + 2, width + 2), np.uint8)

            print(f"Ejecutando Flood Fill con Tolerancia={tolerance}...")
            num_pixels_filled, _, _, _ = cv2.floodFill(frame_copy, mask, (x, y), (0, 255, 0), diff, diff, cv2.FLOODFILL_FIXED_RANGE)

            cv2.imshow("Visor de video", frame_copy)
            print("¡Segmentación completada!")

            # 3. Analizar la Mancha y Guardar Rangos Globales
            if num_pixels_filled > 0:
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_area = mask[1:-1, 1:-1]
                hsv_pixels_in_stain = hsv_frame[mask_area == 1]
                
                min_h = max(0, h_seed - HUE_MARGIN)
                max_h = min(179, h_seed + HUE_MARGIN)
                min_s = int(np.min(hsv_pixels_in_stain[:, 1]))
                max_s = int(np.max(hsv_pixels_in_stain[:, 1]))
                min_v = int(np.min(hsv_pixels_in_stain[:, 2]))
                max_v = int(np.max(hsv_pixels_in_stain[:, 2]))

                # Guardamos los rangos globalmente
                lower_white = np.array([min_h, min_s, min_v])
                upper_white = np.array([max_h, max_s, max_v])

                print("\n--- ¡RANGO DE COLOR DEFINIDO Y GUARDADO! ---")
                print(f"lower_white = {lower_white}")
                print(f"upper_white = {upper_white}")
                print("------------------------------------")
                print("¡Listo! Ahora puede presionar 'e' para el seguimiento automático.")
                estado_actual = "MARCAR_PUNTOS" # Volver al modo normal
            else:
                print("No se encontraron píxeles. Pruebe una tolerancia mayor o un píxel diferente.")
            print("====================================")

        elif estado_actual == "MARCAR_PUNTOS":
            frame_key = current_frame
            clicked_points[frame_key] = (x, y)
            puntos_ciclo_actual[frame_key] = (x, y)
            punto_marcado_en_frame_actual = True
            print(f"Marca manual añadida en frame {frame_key}: ({x}, {y}).")
    
    if estado_actual != "DEFINIR_COLOR":
        on_trackbar(current_frame)

# --- Función callback para la trackbar "Frame" ---
def on_trackbar(val):
    global current_frame, punto_marcado_en_frame_actual, current_frame_image
    current_frame = val
    punto_marcado_en_frame_actual = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        current_frame_image = frame.copy() # Guarda el frame limpio
        show_frame_with_markers(frame.copy(), current_frame)

# --- Callback para la trackbar "Tolerancia" ---
def on_tolerance_trackbar(val):
    # No necesita hacer nada, el valor se lee al hacer clic
    pass

# --- Creación de la ventana y controles ---
cv2.namedWindow("Visor de video")
cv2.createTrackbar("Frame", "Visor de video", 0, total_frames - 1, on_trackbar)
cv2.createTrackbar("Tolerancia", "Visor de video", initial_tolerance, 255, on_tolerance_trackbar)
cv2.setMouseCallback("Visor de video", mouse_callback)
on_trackbar(0)

# --- Bucle principal de Interacción ---
print("\n--- Controles ---")
print(" 'r': (PASO 1) Reconfigurar el centro de rotación.")
print(" 'b': (PASO 2) Definir/redefinir los bordes del disco.")
print(" 'k': (PASO 3) Definir el color de la mancha (MODO FLOOD FILL).")
print("-----------------")
print(" 'e': Empezar seguimiento AUTOMÁTICO (después de 'r', 'b', 'k').")
print(" Click izquierdo: Marcar un punto (en modo de marcado).")
print(" 'a' / 'd': Navegar frames.")
print(" 'c': Calcular velocidad angular con TODOS los puntos guardados.")
print(" 'l': Limpiar (reiniciar) todos los puntos marcados.")
print(" 'q': Salir.")
print("-----------------\n")

while True:
    key = cv2.waitKey(10) & 0xFF

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
            rad_s, rpm = calcular_velocidades_polyfit(clicked_points, fps)
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

    elif key == ord('k'):
        if not centro_rotacion or radio_promedio == 0:
            print("\nError: Primero debe definir un centro ('r') y los bordes ('b').")
        else:
            estado_actual = "DEFINIR_COLOR"
            print("\n--- Modo 'Definir Color' activado. ---")
            print("Ajuste la barra 'Tolerancia' y haga clic en la mancha blanca.")
            on_trackbar(current_frame) # Para actualizar el mensaje de estado

    elif key == ord('l'):
        clicked_points = {}
        puntos_ciclo_actual = {}
        lower_white = None
        upper_white = None
        on_trackbar(current_frame)
        print("\n¡Puntos y Rango de Color reiniciados!")
        
    elif key == ord('e'):
        if centro_rotacion and radio_promedio > 0:
            seguimiento_automatico()
        else:
            print("\nError: Primero debe definir un centro ('r') y los bordes del disco ('b').")
# --- Cierre de OpenCV ---
cap.release()
cv2.destroyAllWindows()