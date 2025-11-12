import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


# --- Configuración ---
# !! IMPORTANTE: Asegúrate de que la ruta del video sea correcta !!
#Local:
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
#Drive:
# video_path = 'g:/Otros ordenadores/Asus Laptop/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'

csv_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/voltaje_resultados.csv'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

try:
    df_voltaje = pd.read_csv(csv_path)
    print(f"Datos de voltaje cargados desde '{csv_path}' ({len(df_voltaje)} filas).")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo CSV en {csv_path}")
    print("El gráfico de Voltaje vs RPM no estará disponible.")
    df_voltaje = None # Si no se encuentra, lo ponemos a None
except Exception as e:
    print(f"Error al cargar el CSV: {e}")
    df_voltaje = None

# --- Obtención de datos del video ---
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"Video cargado correctamente.")
print(f"Total de frames: {total_frames}")
print(f"FPS del video: {fps:.2f}")
print(f"Resolución: {width}x{height}")

# --- Variables Globales ---

# -- Estado del Video --
current_frame = 0
current_frame_image = None
clicked_points = {}
puntos_ciclo_actual = {}

# -- Geométria del Video --
centro_rotacion = None
roi_seleccionada = None
roi_puntos_temp = []
dibujando_roi = False

# -- Selector de Color --
lower_white = None
upper_white = None
initial_tolerance = 35
HUE_MARGIN = 10

# -- Lógica de estados empezando por marcar el centro
estado_actual = "DEFINIR_CENTRO"

# -- Precarga de datos para agilizar los pasos
centro_precargado = (285, 295)

roi_precargada = (220, 220, 370, 370) #recalcularloq

try:
    # 1. Cargar el CENTRO predeterminado
    if centro_precargado and len(centro_precargado) == 2:
        centro_rotacion = centro_precargado
        estado_actual = "DEFINIR_ROI" # Siguiente paso de Estado
        print("\n--- CONFIGURACIÓN GEOMÉTRICA PRECARGADA ---")
        print(f"Centro: {centro_rotacion} (OK)")

        # 2. Intentar cargar la ROI (SOLO si el centro se cargó)
        if roi_precargada and len(roi_precargada) == 4:
            x1 = min(roi_precargada[0], roi_precargada[2])
            y1 = min(roi_precargada[1], roi_precargada[3])
            x2 = max(roi_precargada[0], roi_precargada[2])
            y2 = max(roi_precargada[1], roi_precargada[3])
            roi_seleccionada = (x1, y1, x2, y2)
            
            estado_actual = "DEFINIR_COLOR" # Siguiente paso
            print(f"ROI:    {roi_seleccionada} (OK)")
            print("------------------------------------------")
            print("\n¡Geometría lista! Presiona 'k' para definir el color.")
            print("O presiona 'c' (centro) o 'r' (ROI) para redefinir.")
        else:
            print("------------------------------------------")
            print("\n¡Centro listo! Presiona 'c' para redefinir.")
            print("PASO SIGUIENTE: Defina la ROI (Región de Interés).")
    
    else:
        print("Iniciando en modo manual: 'DEFINIR_CENTRO'.")
        
except Exception as e:
    print(f"Error en precarga ({e}). Iniciando en modo manual: 'DEFINIR_CENTRO'.")

print("--- Fin de la inicialización ---")

# --- Creación de la Ventana, Trackbars y Bucle Principal ---

# --- Función callback para la trackbar "Frame" ---
def on_trackbar(val):
    """
    Se activa cuando se mueve el trackbar 'Frame'.
    Carga el frame 'val', lo guarda globalmente y lo muestra.
    """
    global current_frame, current_frame_image
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        current_frame_image = frame.copy() # Guarda el frame limpio
        
        # --- (LÓGICA DE DIBUJADO ACTUALIZADA) ---
        # Llama a la nueva función de dibujado centralizada
        show_frame_with_markers(frame.copy(), current_frame)
        # ----------------------------------------
        
    else:
        print(f"Error al leer el frame {current_frame}")

# --- Callback para la trackbar "Tolerancia" ---
def on_tolerance_trackbar(val):
    """
    Callback vacío. El valor de la tolerancia se leerá
    directamente cuando el usuario haga clic.
    """
    pass

# --- Funciones de Análisis y Física ---

def analyze_white_cluster_in_fixed_roi(frame_to_analyze, roi_coords, final_lower_hsv, final_upper_hsv):
    """
    (Portado de PFis2.py)
    Analiza una mancha dentro de una ROI fija en un solo fotograma,
    usando los rangos HSV calculados por el usuario.
    Retorna el centroide, área, contorno y la imagen segmentada.
    """
    if frame_to_analyze is None or roi_coords is None or final_lower_hsv is None or final_upper_hsv is None:
        return None, None, None, np.zeros(frame_to_analyze.shape[:2], dtype=np.uint8)

    (x1, y1, x2, y2) = roi_coords
    h_frame, w_frame = frame_to_analyze.shape[:2]

    # Limitar coordenadas de la ROI
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    if x2 <= x1 or y2 <= y1: # ROI inválida
        return None, None, None, np.zeros(frame_to_analyze.shape[:2], dtype=np.uint8)

    roi_image = frame_to_analyze[y1:y2, x1:x2]

    if roi_image.size == 0: # ROI vacía
        return None, None, None, np.zeros(frame_to_analyze.shape[:2], dtype=np.uint8)

    # --- Convertir la ROI a HSV y aplicar los rangos dinámicos ---
    hsv_roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi_image, final_lower_hsv, final_upper_hsv)

    # Integrar máscara en una imagen del tamaño del frame original
    full_segmented_image = np.zeros((h_frame, w_frame), dtype=np.uint8)
    full_segmented_image[y1:y2, x1:x2] = mask

    # Buscar contornos
    contours, _ = cv2.findContours(full_segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroid = None
    area = None
    largest_contour = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # --- ¡IMPORTANTE! Umbral de área ---
        # Evita que se detecte "ruido" como un punto válido.
        # Ajusta este valor si es necesario.
        if area > 40:  
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
            
    return centroid, area, largest_contour, full_segmented_image


def calculate_angle(center, point):
    """ 
    (Portado de PFis2.py)
    Calcula el ángulo en radianes entre el centro y un punto dado. 
    """
    if center is None or point is None:
        return None
    # OJO: Se invierte 'dy' (y - center[1]) porque en OpenCV 
    # el eje Y crece hacia abajo.
    dx = point[0] - center[0]
    dy = -(point[1] - center[1]) 
    return math.atan2(dy, dx) 

def calculate_angular_velocity(angles, times):
    """ 
    (Portado de PFis2.py)
    Calcula la velocidad angular (rad/s) entre los dos últimos puntos.
    """
    if len(angles) < 2 or len(times) < 2:
        return 0.0

    delta_angle = angles[-1] - angles[-2]
    delta_time = times[-1] - times[-2]

    # Ajuste para el salto de -pi a pi o viceversa (unwrap)
    if delta_angle > math.pi:
        delta_angle -= 2 * math.pi
    elif delta_angle < -math.pi:
        delta_angle += 2 * math.pi

    if delta_time == 0:
        return 0.0
    
    return delta_angle / delta_time

# --- Función de Seguimiento Automático ---
def seguimiento_automatico():
    """
    (Lógica adaptada de PFis2.py, Fase 4)
    Esta función se llama al presionar 'e'.
    Analiza el video completo, calcula la velocidad y muestra el gráfico.
    """
    global cap, fps, total_frames
    global centro_rotacion, roi_seleccionada, lower_white, upper_white
    global puntos_ciclo_actual, df_voltaje 

    print("\n--- INICIANDO FASE DE ANÁLISIS AUTOMÁTICO ---")
    print("   (Presione 'ESPACIO' para pausar, 'q' para cancelar)")

    # 1. Resetear historiales y video
    centroids_history = []
    frame_times_history = [] 
    angles_history = [] 
    angular_velocities_history = []
    voltages_for_plot = []
    puntos_ciclo_actual = {} # Limpiar puntos previos

    MIN_PIXEL_DISTANCE = 25
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rebobinar video
    is_paused = False
    
    # 2. Bucle de análisis (Frame por Frame)
    while True:
        
        # 3. Lógica de Pausa
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("--- Fin del video. ---")
                break
            
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_time = current_frame_num / fps # Tiempo en segundos

            # 4. ANÁLISIS (Usando las funciones del Paso 4)
            centroid, area, largest_contour, _ = analyze_white_cluster_in_fixed_roi(
                frame, roi_seleccionada, lower_white, upper_white
            )

            frame_display = frame.copy()
            angular_velocity = 0.0

            # 5. Si se detecta la mancha
            if centroid:
                distancia_al_centro = math.dist(centro_rotacion, centroid)
                
                if distancia_al_centro > MIN_PIXEL_DISTANCE:
                
                    puntos_ciclo_actual[current_frame_num] = centroid # Guardar para dibujar
                    centroids_history.append(centroid)
                    frame_times_history.append(frame_time)

                    # Dibujar contorno y centroide
                    if largest_contour is not None:
                        cv2.drawContours(frame_display, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame_display, centroid, 5, (0, 0, 255), -1)

                    # Calcular ángulo (Usando Paso 4)
                    angle_rad = calculate_angle(centro_rotacion, centroid)
                    if angle_rad is not None:
                        angles_history.append(angle_rad)

                    # Calcular velocidad angular (Usando Paso 4)
                    if len(angles_history) >= 2:
                        angular_velocity = calculate_angular_velocity(angles_history, frame_times_history)
                        angular_velocities_history.append(angular_velocity)

                        if df_voltaje is not None:
                                try:
                                    # Buscar voltaje por número de frame
                                    voltage = df_voltaje.iloc[current_frame_num]['valor_display']
                                    voltages_for_plot.append(voltage)
                                except IndexError:
                                    # Ocurre si el video es más largo que el CSV
                                    voltages_for_plot.append(np.nan) # np.nan es mejor que 0
                        else:
                                # Ocurre si el CSV no se cargó
                                voltages_for_plot.append(np.nan)
                    
            # 6. Dibujar información en pantalla
            if centro_rotacion:
                cv2.circle(frame_display, centro_rotacion, 7, (0, 255, 0), -1)
            if roi_seleccionada:
                cv2.rectangle(frame_display, (roi_seleccionada[0], roi_seleccionada[1]), (roi_seleccionada[2], roi_seleccionada[3]), (0, 255, 255), 2)
            
            cv2.putText(frame_display, f"Procesando: {current_frame_num}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_display, "Presione 'ESPACIO' para PAUSA, 'q' para SALIR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar RPM (w * 60 / 2*pi)
            rpm = abs(angular_velocity * 9.5493) 
            cv2.putText(frame_display, f"Velocidad: {rpm:.2f} RPM", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Visor de video", frame_display)

        # 7. Manejo de Teclas (Pausa y Salir)
        # (Usamos un delay bajo para que el video corra rápido, 
        # o 0 si está en pausa para que espere)
        delay = 0 if is_paused else int(1000 / fps) # Delay para velocidad normal
        if delay <= 0:
            delay = 1
            
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'): 
            print("--- Análisis cancelado por el usuario. ---")
            break
        
        elif key == ord(' '): # ESPACIO para pausar
            is_paused = not is_paused
            if is_paused:
                print(f"--- PAUSA en frame {current_frame_num}. Use 'a'/'d' para navegar, 'ESPACIO' para continuar. ---")
            else:
                print("--- Reanudando análisis. ---")
                
        elif is_paused and (key == ord('a') or key == ord('d')):
            # --- INICIO: Lógica de navegación en pausa ---
            
            # 1. Calcular nuevo frame
            if key == ord('a'): # Frame anterior
                current_frame_num = max(current_frame_num - 1, 0)
            elif key == ord('d'): # Frame siguiente
                current_frame_num = min(current_frame_num + 1, total_frames - 1)
            
            # 2. Cargar ese frame específico
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Error al leer frame {current_frame_num} en modo pausa.")
                continue

            frame_display = frame.copy()

            # 3. Volver a dibujar los marcadores estáticos y el punto guardado
            if centro_rotacion:
                cv2.circle(frame_display, centro_rotacion, 7, (0, 255, 0), -1)
            if roi_seleccionada:
                cv2.rectangle(frame_display, (roi_seleccionada[0], roi_seleccionada[1]), (roi_seleccionada[2], roi_seleccionada[3]), (0, 255, 255), 2)
            
            # Buscar si tenemos un punto guardado para este frame
            if current_frame_num in puntos_ciclo_actual:
                centroid_guardado = puntos_ciclo_actual[current_frame_num]
                cv2.circle(frame_display, centroid_guardado, 5, (0, 0, 255), -1)
                if centro_rotacion:
                    cv2.line(frame_display, centro_rotacion, centroid_guardado, (255, 255, 0), 1)

            # 4. Actualizar texto en pantalla
            cv2.putText(frame_display, f"PAUSADO (Navegando): {current_frame_num}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_display, "Use 'a'/'d'. 'ESPACIO' para reanudar, 'q' para SALIR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # (No mostramos RPM pues no estamos calculando)
            cv2.imshow("Visor de video", frame_display)     

# FIN DEL BUCLE: CÁLCULO FINAL Y GRÁFICO
    print("--- ANÁLIS FINALIZADO ---")
    
    if angular_velocities_history:
        # Calcular promedio
        average_angular_velocity_rad_s = np.mean(angular_velocities_history)
        average_rpm = abs(average_angular_velocity_rad_s * 9.5493) # 60 / 2*pi

        print(f"\n--- RESULTADOS DEL ANÁLISIS ---")
        print(f"Puntos analizados: {len(angular_velocities_history)}")
        print(f"Velocidad Angular Promedio: {abs(average_angular_velocity_rad_s):.2f} rad/s")
        print(f"Velocidad de Giro Promedio: {average_rpm:.2f} RPM")

        # 9. GENERACIÓN DEL GRÁFICO (RPM vs Tiempo)
        
        # --- CAMBIO 1: Ponerle nombre a la Ventana 1 ---
        plt.figure(num="Figura 1: RPM vs. Tiempo", figsize=(10, 6))
        
        # Graficamos RPM vs Tiempo
        rpms_history = np.abs(np.array(angular_velocities_history) * 9.5493)
        # Asegurarse de que los tiempos coincidan (historial de velocidad empieza en el 2do frame)
        times_plot = frame_times_history[1:len(rpms_history)+1] 
        
        plt.plot(times_plot, rpms_history, marker='o', linestyle='-', color='b', markersize=2)
        plt.title('Velocidad (RPM) en Función del Tiempo')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Velocidad de Giro (RPM)')
        plt.grid(True)
        plt.axhline(y=average_rpm, color='r', linestyle='--', label=f'Promedio: {average_rpm:.2f} RPM')
        plt.legend()

        # --- INICIO: NUEVO GRÁFICO (RPM vs Voltaje) ---
        if voltages_for_plot and rpms_history.size == len(voltages_for_plot):
            
            # --- Ventana 2 ---
            plt.figure(num="Figura 2: RPM vs. Voltaje", figsize=(10, 6)) # Nueva figura
            
            v_data = np.array(voltages_for_plot)
            rpm_data = rpms_history

            valid_indices = ~np.isnan(v_data) & ~np.isnan(rpm_data)
            v_clean = v_data[valid_indices]
            rpm_clean = rpm_data[valid_indices]

            if v_clean.size > 1:
                plt.scatter(v_clean, rpm_clean, alpha=0.5, s=10, label="Datos (frame-a-frame)")
                
                try:
                    model = np.polyfit(v_clean, rpm_clean, 1)
                    m, b = model[0], model[1]
                    v_line = np.linspace(v_clean.min(), v_clean.max(), 100)
                    rpm_line = m * v_line + b
                    plt.plot(v_line, rpm_line, color='r', linestyle='--', 
                             label=f'Tendencia: RPM = {m:.2f}*V + {b:.2f}')
                except Exception as e:
                    print(f"No se pudo calcular la línea de tendencia: {e}")

                plt.title('Velocidad (RPM) en Función del Voltaje')
                plt.xlabel('Voltaje (V)')
                plt.ylabel('Velocidad de Giro (RPM)')
                plt.grid(True)
                plt.legend()
                
            else:
                print("No se generó el gráfico de Voltaje vs RPM (no hay suficientes datos limpios).")
        
        elif not voltages_for_plot:
            print("No se generó el gráfico de Voltaje vs RPM (CSV no cargado o sin datos).")
        else:
            print(f"Error de coincidencia de datos: RPMs ({rpms_history.size}) vs Voltajes ({len(voltages_for_plot)})")
        
        # --- CAMBIO 4: Se llama a plt.show() UNA SOLA VEZ AL FINAL ---
        print("Mostrando resultados. Cierre las ventanas de gráficos para continuar.")
        plt.show() 
    
    else:
        print("\nNo se pudieron calcular velocidades. Asegúrate de que la mancha fue detectada.")

    # 10. Regresar al modo normal
    puntos_ciclo_actual = {}
    on_trackbar(0) # Volver al frame 0
    print("--- Volviendo al modo de espera. ---")

# --- Funciones de Visualización y Callback ---

def show_frame_with_markers(frame, frame_number):
    """
    Dibuja todos los marcadores, geometrías y texto de estado en el frame.
    """
    global dibujando_roi, roi_puntos_temp # Necesita acceso para el dibujado en vivo
    
    # Dibuja el centro
    if centro_rotacion:
        cv2.circle(frame, centro_rotacion, 7, (0, 255, 0), -1)
        cv2.putText(frame, 'CENTRO', (centro_rotacion[0] + 10, centro_rotacion[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dibuja la ROI FINALIZADA
    if roi_seleccionada:
        p1 = (roi_seleccionada[0], roi_seleccionada[1])
        p2 = (roi_seleccionada[2], roi_seleccionada[3])
        cv2.rectangle(frame, p1, p2, (0, 255, 255), 2) # Amarillo
        cv2.putText(frame, 'ROI', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Dibuja la ROI TEMPORAL (mientras se arrastra el mouse)
    if dibujando_roi and len(roi_puntos_temp) == 2:
        p1 = roi_puntos_temp[0]
        p2 = roi_puntos_temp[1]
        cv2.rectangle(frame, p1, p2, (255, 100, 100), 2) # Azul claro
    
    # Dibuja el punto de seguimiento (si existe en este frame)
    if frame_number in puntos_ciclo_actual:
        point = puntos_ciclo_actual[frame_number]
        if centro_rotacion:
            cv2.line(frame, centro_rotacion, point, (255, 255, 0), 1)
        cv2.drawMarker(frame, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    # Mensajes de estado
    mensaje = ""
    if estado_actual == "DEFINIR_CENTRO":
        mensaje = "PASO 1: Haga clic para definir el CENTRO de rotacion."
    elif estado_actual == "DEFINIR_ROI":
        mensaje = "PASO 2: Haga clic y arrastre para definir la ROI."
    elif estado_actual == "DEFINIR_COLOR":
        mensaje = "PASO 3: Ajuste 'Tolerancia' y HAGA CLIC en la mancha blanca."
    elif estado_actual == "MARCAR_PUNTOS":
        mensaje = f"Modo Marcado/Listo. Frame: {frame_number}. (Use 'e' para tracking)"
        
    cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Visor de video", frame)

# --- Creación de la ventana y controles ---
cv2.namedWindow("Visor de video")
cv2.createTrackbar("Frame", "Visor de video", 0, total_frames - 1, on_trackbar)
cv2.createTrackbar("Tolerancia", "Visor de video", initial_tolerance, 255, on_tolerance_trackbar)

# --- Función callback para el click del mouse ---
def mouse_callback(event, x, y, flags, param):
    global current_frame, current_frame_image
    global estado_actual, centro_rotacion
    global roi_seleccionada, roi_puntos_temp, dibujando_roi
    global lower_white, upper_white

    # --- LÓGICA DEL ESTADO: DEFINIR_CENTRO ---
    if estado_actual == "DEFINIR_CENTRO":
        if event == cv2.EVENT_LBUTTONDOWN:
            centro_rotacion = (x, y)
            print(f"\n*** Centro de rotación definido en: {centro_rotacion} ***")
            on_trackbar(current_frame)

    # --- LÓGICA DEL ESTADO: DEFINIR_ROI ---
    elif estado_actual == "DEFINIR_ROI":
        if event == cv2.EVENT_LBUTTONDOWN:
            # Empezamos a dibujar
            dibujando_roi = True
            # Guardamos el punto inicial (y el final temporalmente)
            roi_puntos_temp = [(x, y), (x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if dibujando_roi:
                # Actualizamos solo el punto final mientras arrastramos
                roi_puntos_temp[1] = (x, y)
                
                # --- Dibujado en vivo (sin llamar a on_trackbar) ---
                frame_display = current_frame_image.copy()
                if centro_rotacion:
                    cv2.circle(frame_display, centro_rotacion, 7, (0, 255, 0), -1)
                
                # Dibujamos el rectángulo temporal
                p1 = roi_puntos_temp[0]
                p2 = roi_puntos_temp[1]
                cv2.rectangle(frame_display, p1, p2, (255, 100, 100), 2) # Azul claro
                cv2.putText(frame_display, "Suelte para confirmar", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                cv2.imshow("Visor de video", frame_display)
                # --- Fin dibujado en vivo ---

        elif event == cv2.EVENT_LBUTTONUP:
            # Dejamos de dibujar
            dibujando_roi = False
            # Guardamos el punto final definitivo
            roi_puntos_temp[1] = (x, y)
            
            # Normalizamos las coordenadas (x1, y1) debe ser arriba-izquierda
            x1 = min(roi_puntos_temp[0][0], roi_puntos_temp[1][0])
            y1 = min(roi_puntos_temp[0][1], roi_puntos_temp[1][1])
            x2 = max(roi_puntos_temp[0][0], roi_puntos_temp[1][0])
            y2 = max(roi_puntos_temp[0][1], roi_puntos_temp[1][1])
            
            # Guardamos la ROI final
            roi_seleccionada = (x1, y1, x2, y2)
            
            print(f"\n*** ROI definida en: {roi_seleccionada} ***")
            on_trackbar(current_frame)

    # --- LÓGICA DEL ESTADO: DEFINIR_COLOR ---
    elif estado_actual == "DEFINIR_COLOR":
        if event == cv2.EVENT_LBUTTONDOWN:
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
            
            # Mostramos el resultado del Flood Fill
            cv2.imshow("Visor de video", frame_copy)
            print("¡Segmentación completada!")

            # 3. Analizar la Mancha y Guardar Rangos Globales
            if num_pixels_filled > 0:
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_area = mask[1:-1, 1:-1]
                
                # Solo píxeles dentro de la mancha
                hsv_pixels_in_stain = hsv_frame[mask_area == 1]
                
                if hsv_pixels_in_stain.size > 0:
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
                    estado_actual = "MARCAR_PUNTOS" # "MARCAR_PUNTOS" es el estado "Listo"
                else:
                    print("Error: No se pudieron analizar píxeles en la mancha (posiblemente fuera de la ROI).")
            else:
                print("No se encontraron píxeles. Pruebe una tolerancia mayor o un píxel diferente.")
            print("====================================")
            
            # Esperamos 1 segundo y volvemos al frame normal
            cv2.waitKey(1000)
            on_trackbar(current_frame)
    
    # --- LÓGICA DEL ESTADO: LISTO (MARCAR_PUNTOS) ---
    elif estado_actual == "MARCAR_PUNTOS":
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Modo listo. Presione 'e' para iniciar el tracking.")

# Conectar la ventana a la función de callback del mouse
cv2.setMouseCallback("Visor de video", mouse_callback)

# Mostrar el primer frame
on_trackbar(0)

# --- Bucle principal de Interacción ---
print("\n--- Controles ---")
print(" 'q': Salir")
print(" 'a' / 'd': Navegar frames")
print(" 'c': (Re)definir el CENTRO de rotación")
print(" 'r': (Re)definir la ROI (Región de Interés)")
print(" 'k': Definir el color de la mancha (MODO FLOOD FILL)")
print(" 'e': Empezar seguimiento AUTOMÁTICO")
print(" 'l': Limpiar (reiniciar) todos los puntos marcados")
print("-----------------\n")

while True:
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'): 
        break
    
    # --- Navegación ---
    elif key == ord('d'):
        current_frame = min(current_frame + 1, total_frames - 1)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)

    elif key == ord('a'):
        current_frame = max(current_frame - 1, 0)
        cv2.setTrackbarPos("Frame", "Visor de video", current_frame)
        on_trackbar(current_frame)
    
    elif key == ord('c'): # 'C' para CENTRO
        estado_actual = "DEFINIR_CENTRO"
        centro_rotacion = None
        print("\n--- Modo 'Definir Centro' activado. ---")
        print("Haga clic en el video para definir el centro.")
        on_trackbar(current_frame) # Actualizar visual
    
    elif key == ord('r'): # 'R' para ROI
        estado_actual = "DEFINIR_ROI"
        roi_seleccionada = None
        roi_puntos_temp = []
        dibujando_roi = False
        print("\n--- Modo 'Definir ROI' activado. ---")
        print("Haga clic y arrastre para dibujar el rectángulo (ROI).")
        on_trackbar(current_frame) # Actualizar visual

    elif key == ord('k'): # 'K' para Color
        if not centro_rotacion or not roi_seleccionada:
            print("\nError: Primero debe definir el Centro ('c') y la ROI ('r').")
        else:
            estado_actual = "DEFINIR_COLOR"
            print("\n--- Modo 'Definir Color' activado. ---")
            print("Ajuste la barra 'Tolerancia' y haga clic en la mancha blanca.")
            on_trackbar(current_frame) # Actualizar visual

    elif key == ord('e'): # 'E' para Empezar Tracking
        if not centro_rotacion or not roi_seleccionada:
            print("\nError: Faltan Centro ('c') o ROI ('r') por definir.")
        elif lower_white is None:
            print("\nError: Falta definir el Color ('k') antes de empezar.")
        else:
            seguimiento_automatico() 


    elif key == ord('l'): # 'L' para Limpiar
        print("\n¡Datos de seguimiento y color limpiados!")
        clicked_points = {}
        puntos_ciclo_actual = {}
        lower_white = None
        upper_white = None
        on_trackbar(current_frame)

# --- Funciones de Visualización y Callback ---

def show_frame_with_markers(frame, frame_number):
    """
    Dibuja todos los marcadores, geometrías y texto de estado en el frame.
    """
    global dibujando_roi, roi_puntos_temp # Necesita acceso para el dibujado en vivo
    
    # Dibuja el centro
    if centro_rotacion:
        cv2.circle(frame, centro_rotacion, 7, (0, 255, 0), -1)
        cv2.putText(frame, 'CENTRO', (centro_rotacion[0] + 10, centro_rotacion[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dibuja la ROI FINALIZADA
    if roi_seleccionada:
        p1 = (roi_seleccionada[0], roi_seleccionada[1])
        p2 = (roi_seleccionada[2], roi_seleccionada[3])
        cv2.rectangle(frame, p1, p2, (0, 255, 255), 2) # Amarillo
        cv2.putText(frame, 'ROI', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Dibuja la ROI TEMPORAL (mientras se arrastra el mouse)
    if dibujando_roi and len(roi_puntos_temp) == 2:
        p1 = roi_puntos_temp[0]
        p2 = roi_puntos_temp[1]
        cv2.rectangle(frame, p1, p2, (255, 100, 100), 2) # Azul claro
    
    # Dibuja el punto de seguimiento (si existe en este frame)
    if frame_number in puntos_ciclo_actual:
        point = puntos_ciclo_actual[frame_number]
        if centro_rotacion:
            cv2.line(frame, centro_rotacion, point, (255, 255, 0), 1)
        cv2.drawMarker(frame, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    # Mensajes de estado
    mensaje = ""
    if estado_actual == "DEFINIR_CENTRO":
        mensaje = "PASO 1: Haga clic para definir el CENTRO de rotacion."
    elif estado_actual == "DEFINIR_ROI":
        mensaje = "PASO 2: Haga clic y arrastre para definir la ROI."
    elif estado_actual == "DEFINIR_COLOR":
        mensaje = "PASO 3: Ajuste 'Tolerancia' y HAGA CLIC en la mancha blanca."
    elif estado_actual == "MARCAR_PUNTOS":
        mensaje = f"Modo Marcado/Listo. Frame: {frame_number}. (Use 'e' para tracking)"
        
    cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Visor de video", frame)
    

# --- Cierre de OpenCV ---
cap.release()
cv2.destroyAllWindows()
print("Aplicación cerrada.")