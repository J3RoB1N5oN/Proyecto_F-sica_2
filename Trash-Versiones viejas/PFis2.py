import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# --- Constantes y variables globales ---
# Para la ROI de la mancha blanca
selected_roi_spot = []
drawing_roi_spot = False
roi_spot_finalized = False 

# Para el centro del disco
selected_center = None 
selecting_center = False 
center_finalized = False 

initial_frame_for_selection = None # Frame para las fases de selección inicial
initial_frames_for_hsv_selection = [] # Lista para almacenar los primeros N frames para la selección HSV
NUM_FRAMES_FOR_HSV_SELECTION = 5 # Número de frames a considerar para la selección HSV

# Nombres de las ventanas
MAIN_WINDOW_NAME = 'Video Analysis - Angular Velocity'
ROI_SPOT_WINDOW_NAME = 'Select Spot ROI - Press C to Confirm' 
CENTER_WINDOW_NAME = 'Select Disk Center - Click and Press C' 
HSV_RANGE_SELECTION_WINDOW_NAME = "Select HSV Ranges for Spot - Press Q to Finish" 

# Variables para la detección de rango HSV (del segundo script)
rangos_hsv_guardados = [] # Lista global para guardar todos los rangos (lower, upper)
HUE_MARGIN = 10 # Margen para el HUE en la detección de rango
initial_floodfill_tolerance = 35 # Tolerancia inicial para floodFill

# Variables para almacenar los datos de movimiento para velocidad angular
centroids_history = []
frame_times_history = [] 
angles_history = [] 
angular_velocities_history = [] # Nueva: Para almacenar todas las velocidades angulares

# Variables para control de reproducción
is_paused = False
current_hsv_selection_frame_idx = 0

# --- Funciones de callback para eventos del ratón ---

def draw_roi_spot_callback(event, x, y, flags, param):
    """Callback para la selección de la ROI de la mancha blanca."""
    global selected_roi_spot, drawing_roi_spot, roi_spot_finalized, initial_frame_for_selection

    if roi_spot_finalized:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_roi_spot = True
        selected_roi_spot = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_roi_spot:
            temp_frame = initial_frame_for_selection.copy()
            cv2.rectangle(temp_frame, selected_roi_spot[0], (x, y), (0, 255, 0), 2)
            cv2.putText(temp_frame, "Arrastra y suelta. Pulsa 'C' para confirmar, 'R' para resetear.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(ROI_SPOT_WINDOW_NAME, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_roi_spot = False
        selected_roi_spot.append((x, y))
        x1 = min(selected_roi_spot[0][0], selected_roi_spot[1][0])
        y1 = min(selected_roi_spot[0][1], selected_roi_spot[1][1])
        x2 = max(selected_roi_spot[0][0], selected_roi_spot[1][0])
        y2 = max(selected_roi_spot[0][1], selected_roi_spot[1][1])

        h_frame, w_frame = initial_frame_for_selection.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)
        
        selected_roi_spot = [(x1,y1), (x2,y2)]
        
        temp_frame = initial_frame_for_selection.copy()
        cv2.rectangle(temp_frame, selected_roi_spot[0], selected_roi_spot[1], (0, 255, 0), 2)
        cv2.putText(temp_frame, "Arrastra y suelta. Pulsa 'C' para confirmar, 'R' para resetear.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(ROI_SPOT_WINDOW_NAME, temp_frame)

def select_disk_center_callback(event, x, y, flags, param):
    """Callback para la selección del centro del disco."""
    global selected_center, selecting_center, center_finalized, initial_frame_for_selection

    if not selecting_center or center_finalized:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_center = (x, y)
        temp_frame = initial_frame_for_selection.copy()
        cv2.circle(temp_frame, selected_center, 5, (0, 0, 255), -1)
        cv2.putText(temp_frame, "Centro seleccionado. Pulsa 'C' para confirmar.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(CENTER_WINDOW_NAME, temp_frame)

def on_click_floodfill_hsv_range(event, x, y, flags, param):
    """Callback para la selección de rangos HSV usando floodFill."""
    global rangos_hsv_guardados, current_hsv_selection_frame_idx, initial_frames_for_hsv_selection

    if event == cv2.EVENT_LBUTTONDOWN:
        if not initial_frames_for_hsv_selection:
            print("Error: No hay frames cargados para la selección HSV.")
            return

        frame_for_hsv_selection = initial_frames_for_hsv_selection[current_hsv_selection_frame_idx]
        height, width, _ = frame_for_hsv_selection.shape

        # Asegurarse de que las coordenadas x, y estén dentro de los límites del frame
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        bgr_color = frame_for_hsv_selection[y, x]
        hsv_color = cv2.cvtColor(np.uint8([[[bgr_color[0], bgr_color[1], bgr_color[2]]]]), cv2.COLOR_BGR2HSV)[0][0]
        h_seed, s_seed, v_seed = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])

        print("====================================")
        print(f"* Píxel Semilla Seleccionado para Rango HSV (Frame {current_hsv_selection_frame_idx + 1}) *")
        print(f"Color HSV: [H={h_seed}, S={s_seed}, V={v_seed}]")
        print("------------------------------------")
        
        frame_copy_hsv = frame_for_hsv_selection.copy()
        
        # Obtenemos la tolerancia del trackbar específico de esta ventana
        tolerance = cv2.getTrackbarPos("Tolerancia", HSV_RANGE_SELECTION_WINDOW_NAME)
        
        diff = (tolerance, tolerance, tolerance)
        mask_floodfill = np.zeros((height + 2, width + 2), np.uint8)

        print(f"Ejecutando Flood Fill con Tolerancia={tolerance}...")
        num_pixels_filled, _, _, _ = cv2.floodFill(frame_copy_hsv, mask_floodfill, (x, y), (0, 255, 0), diff, diff, cv2.FLOODFILL_FIXED_RANGE)
        cv2.imshow(HSV_RANGE_SELECTION_WINDOW_NAME, frame_copy_hsv)
        print("¡Segmentación completada!")

        if num_pixels_filled > 0:
            hsv_frame_full = cv2.cvtColor(frame_for_hsv_selection, cv2.COLOR_BGR2HSV)
            mask_area = mask_floodfill[1:-1, 1:-1]
            hsv_pixels_in_stain = hsv_frame_full[mask_area == 1]
            
            min_h = max(0, h_seed - HUE_MARGIN)
            max_h = min(179, h_seed + HUE_MARGIN)
            
            if hsv_pixels_in_stain.size > 0:
                min_s = int(np.min(hsv_pixels_in_stain[:, 1]))
                max_s = int(np.max(hsv_pixels_in_stain[:, 1]))
                min_v = int(np.min(hsv_pixels_in_stain[:, 2]))
                max_v = int(np.max(hsv_pixels_in_stain[:, 2]))
            else: # Fallback si no hay píxeles segmentados
                min_s, max_s = s_seed, s_seed
                min_v, max_v = v_seed, v_seed

            lower_range = np.array([min_h, min_s, min_v])
            upper_range = np.array([max_h, max_s, max_v])

            print("\n--- RANGO ÓPTIMO (Este Clic) ---")
            print(f"lower_hsv = {lower_range}")
            print(f"upper_hsv = {upper_range}")
            
            rangos_hsv_guardados.append((lower_range, upper_range))
            print(f"-> Rango guardado. (Total guardados: {len(rangos_hsv_guardados)})")
            
        else:
            print("No se encontraron píxeles con esa tolerancia.")
        print("====================================")

def on_hsv_tolerance_trackbar(val):
    pass # No hace nada directamente

def on_hsv_frame_trackbar(val):
    """Callback para el trackbar de selección de frame HSV."""
    global current_hsv_selection_frame_idx, initial_frames_for_hsv_selection
    current_hsv_selection_frame_idx = val
    if initial_frames_for_hsv_selection:
        cv2.imshow(HSV_RANGE_SELECTION_WINDOW_NAME, initial_frames_for_hsv_selection[current_hsv_selection_frame_idx])


# --- Función de análisis de la mancha blanca (MODIFICADA para usar rangos HSV dinámicos) ---
def analyze_white_cluster_in_fixed_roi(frame_to_analyze, roi_coords, final_lower_hsv, final_upper_hsv):
    """
    Analiza una mancha dentro de una ROI fija en un solo fotograma,
    usando los rangos HSV calculados por el usuario.
    Retorna el centroide, área, contorno y la imagen segmentada.
    """
    if frame_to_analyze is None or roi_coords is None or final_lower_hsv is None or final_upper_hsv is None:
        return None, None, None, np.zeros(frame_to_analyze.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = roi_coords
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

        if area > 40:  # umbral mínimo de área
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
            
    return centroid, area, largest_contour, full_segmented_image


# --- Nuevas funciones para el cálculo de velocidad angular (sin cambios) ---

def calculate_angle(center, point):
    """ Calcula el ángulo en radianes entre el centro y un punto dado. """
    if center is None or point is None:
        return None
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return math.atan2(dy, dx) 

def calculate_angular_velocity(angles, times):
    """ 
    Calcula la velocidad angular entre dos puntos consecutivos.
    Retorna la velocidad angular en rad/s.
    """
    if len(angles) < 2 or len(times) < 2:
        return 0.0

    delta_angle = angles[-1] - angles[-2]
    delta_time = times[-1] - times[-2]

    # Ajuste para el salto de -pi a pi o viceversa
    if delta_angle > math.pi:
        delta_angle -= 2 * math.pi
    elif delta_angle < -math.pi:
        delta_angle += 2 * math.pi

    if delta_time == 0:
        return 0.0
    
    return delta_angle / delta_time


def run_angular_velocity_analysis(video_path='VideoRecortado.mp4'):
    global selected_roi_spot, drawing_roi_spot, roi_spot_finalized, initial_frame_for_selection
    global selected_center, selecting_center, center_finalized
    global rangos_hsv_guardados, centroids_history, frame_times_history, angles_history, angular_velocities_history
    global initial_frames_for_hsv_selection, current_hsv_selection_frame_idx, is_paused

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    # Leer el primer frame para las selecciones iniciales (ROI y Centro)
    ret, initial_frame_for_selection = cap.read()
    if not ret:
        print("No se pudo leer el primer fotograma del video.")
        cap.release()
        return

    frame_height, frame_width, _ = initial_frame_for_selection.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Advertencia: No se pudieron obtener los FPS del video. Se asumirá 30 FPS.")
        fps = 30.0

    # --- Cargar los primeros N frames para la selección HSV ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Asegurarse de empezar desde el inicio
    for i in range(NUM_FRAMES_FOR_HSV_SELECTION):
        ret, frame = cap.read()
        if ret:
            initial_frames_for_hsv_selection.append(frame.copy())
        else:
            print(f"Advertencia: No se pudieron leer suficientes frames para la selección HSV. Leídos {len(initial_frames_for_hsv_selection)} de {NUM_FRAMES_FOR_HSV_SELECTION}.")
            break
    
    if not initial_frames_for_hsv_selection:
        print("Error: No se cargó ningún frame para la selección HSV. Saliendo.")
        cap.release()
        return
    
    # --- FASE 1: SELECCIÓN DE RANGOS HSV DE LA MANCHA BLANCA ---
    print("\n--- FASE 1: SELECCIÓN DE RANGOS HSV DE LA MANCHA BLANCA ---")
    print(f"Haz clic en las manchas blancas en cualquiera de los primeros {len(initial_frames_for_hsv_selection)} frames para recolectar rangos HSV.")
    print("Usa el trackbar 'Frame' para cambiar entre los frames.")
    print("Cuando termines de hacer clics, presiona 'Q' para finalizar esta fase.")
    
    cv2.namedWindow(HSV_RANGE_SELECTION_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(HSV_RANGE_SELECTION_WINDOW_NAME, initial_frames_for_hsv_selection[0])
    cv2.setMouseCallback(HSV_RANGE_SELECTION_WINDOW_NAME, on_click_floodfill_hsv_range) # Se quita el parámetro 'frame_hsv' porque el callback lo obtiene de la global

    cv2.createTrackbar("Tolerancia", HSV_RANGE_SELECTION_WINDOW_NAME, initial_floodfill_tolerance, 255, on_hsv_tolerance_trackbar)
    cv2.createTrackbar("Frame", HSV_RANGE_SELECTION_WINDOW_NAME, 0, len(initial_frames_for_hsv_selection) - 1, on_hsv_frame_trackbar)

    while True:
        # Actualizar el frame para el callback del ratón
        if initial_frames_for_hsv_selection:
            frame_to_show = initial_frames_for_hsv_selection[current_hsv_selection_frame_idx].copy()
            cv2.putText(frame_to_show, f"Frame: {current_hsv_selection_frame_idx + 1}/{len(initial_frames_for_hsv_selection)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(HSV_RANGE_SELECTION_WINDOW_NAME, frame_to_show)
            # Ya no es necesario setear el callback con el frame, ya que se usa la global current_hsv_selection_frame_idx
            # cv2.setMouseCallback(HSV_RANGE_SELECTION_WINDOW_NAME, on_click_floodfill_hsv_range) 


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyWindow(HSV_RANGE_SELECTION_WINDOW_NAME)

    if not rangos_hsv_guardados:
        print("No se guardó ningún rango HSV. Saliendo.")
        cap.release()
        return

    # Calcular el rango HSV final combinando todos los guardados
    all_lowers = np.array([r[0] for r in rangos_hsv_guardados])
    all_uppers = np.array([r[1] for r in rangos_hsv_guardados])
    final_lower_hsv = np.min(all_lowers, axis=0)
    final_upper_hsv = np.max(all_uppers, axis=0)
    print(f"\n--- RANGO HSV FINAL TOTAL ---")
    print(f"lower_hsv = {final_lower_hsv}")
    print(f"upper_hsv = {final_upper_hsv}")

    # --- FASE 2: SELECCIÓN DE LA ROI DE LA MANCHA BLANCA ---
    print("\n--- FASE 2: SELECCIÓN DE LA ROI (Región de Interés) para la Mancha ---")
    print("Selecciona un área rectangular donde esperas que se mueva la mancha blanca.")
    print("Pulsa 'C' para confirmar, 'R' para resetear, 'Q' para salir.")

    cv2.namedWindow(ROI_SPOT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(ROI_SPOT_WINDOW_NAME, initial_frame_for_selection)
    cv2.setMouseCallback(ROI_SPOT_WINDOW_NAME, draw_roi_spot_callback)

    while not roi_spot_finalized:
        if not drawing_roi_spot and len(selected_roi_spot) < 2:
            temp_frame = initial_frame_for_selection.copy()
            cv2.putText(temp_frame, "Arrastra y suelta. Pulsa 'C' para confirmar, 'R' para resetear.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(ROI_SPOT_WINDOW_NAME, temp_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(selected_roi_spot) == 2:
            roi_spot_finalized = True
        elif key == ord('r'):
            selected_roi_spot = []
            drawing_roi_spot = False
            cv2.imshow(ROI_SPOT_WINDOW_NAME, initial_frame_for_selection)
            print("Selección de ROI reiniciada.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            return

    cv2.destroyWindow(ROI_SPOT_WINDOW_NAME)

    if not selected_roi_spot or len(selected_roi_spot) != 2:
        print("No se seleccionó una ROI válida para la mancha. Saliendo.")
        cap.release()
        return

    x1_roi_spot, y1_roi_spot = selected_roi_spot[0]
    x2_roi_spot, y2_roi_spot = selected_roi_spot[1]
    
    effective_roi_spot_x1 = max(0, min(x1_roi_spot, x2_roi_spot))
    effective_roi_spot_y1 = max(0, min(y1_roi_spot, y2_roi_spot))
    effective_roi_spot_x2 = min(frame_width, max(x1_roi_spot, x2_roi_spot))
    effective_roi_spot_y2 = min(frame_height, max(y1_roi_spot, y2_roi_spot))
    
    fixed_roi_spot_coords = (effective_roi_spot_x1, effective_roi_spot_y1, effective_roi_spot_x2, effective_roi_spot_y2)
    print(f"ROI de la mancha seleccionada: ({effective_roi_spot_x1},{effective_roi_spot_y1}) a ({effective_roi_spot_x2},{effective_roi_spot_y2})")

    # --- FASE 3: SELECCIÓN MANUAL DEL CENTRO DEL DISCO ---
    print("\n--- FASE 3: SELECCIÓN MANUAL DEL CENTRO DEL DISCO ---")
    print("Haz clic en el centro del disco para establecer el punto de rotación.")
    print("Pulsa 'C' para confirmar, 'Q' para salir.")

    cv2.namedWindow(CENTER_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(CENTER_WINDOW_NAME, initial_frame_for_selection)
    cv2.setMouseCallback(CENTER_WINDOW_NAME, select_disk_center_callback)
    selecting_center = True

    while not center_finalized:
        temp_frame = initial_frame_for_selection.copy()
        if selected_center:
            cv2.circle(temp_frame, selected_center, 5, (0, 0, 255), -1)
            cv2.putText(temp_frame, "Centro seleccionado. Pulsa 'C' para confirmar.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(temp_frame, "Haz clic para seleccionar el centro del disco. Pulsa 'C' para confirmar.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(CENTER_WINDOW_NAME, temp_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and selected_center is not None:
            center_finalized = True
        elif key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            return
    
    cv2.destroyWindow(CENTER_WINDOW_NAME)
    selecting_center = False
    print(f"Centro del disco seleccionado: {selected_center}")

    # --- FASE 4: ANÁLISIS AUTOMÁTICO DE VELOCIDAD ANGULAR ---
    print("\n--- FASE 4: ANÁLISIS AUTOMÁTICO DE VELOCIDAD ANGULAR ---")
    print("El video se reproducirá automáticamente. Pulsa 'ESPACIO' para pausar/reanudar, 'Q' para salir.")

    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW_NAME, frame_width, frame_height)

    current_frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resetear la captura de video al inicio para el análisis
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        if not is_paused:
            ret, full_frame = cap.read()
            if not ret:
                print("Fin del video o error al leer fotograma.")
                break
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_time = current_frame_number / fps # Tiempo en segundos

            # --- Detección de la mancha usando los rangos HSV y la ROI ---
            centroid, area, largest_contour, _ = analyze_white_cluster_in_fixed_roi(
                full_frame, fixed_roi_spot_coords, final_lower_hsv, final_upper_hsv
            )

            # Dibujar centro del disco
            if selected_center:
                cv2.circle(full_frame, selected_center, 7, (255, 0, 0), -1) # Azul para el centro

            angular_velocity = 0.0
            if centroid:
                centroids_history.append(centroid)
                frame_times_history.append(frame_time)

                if largest_contour is not None:
                    cv2.drawContours(full_frame, [largest_contour], -1, (0, 255, 0), 2)
                
                cv2.circle(full_frame, centroid, 5, (0, 0, 255), -1) # Rojo para el centroide de la mancha
                
                # Dibujar línea desde el centro del disco al centroide de la mancha
                if selected_center:
                    cv2.line(full_frame, selected_center, centroid, (255, 255, 0), 2) # Cyan
                    
                    # Calcular y mostrar el ángulo
                    angle_rad = calculate_angle(selected_center, centroid)
                    if angle_rad is not None:
                        angles_history.append(angle_rad)
                        angle_deg = math.degrees(angle_rad)
                        cv2.putText(full_frame, f"Angulo: {angle_deg:.2f} deg", (10, full_frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Calcular velocidad angular si hay suficientes datos
                if len(angles_history) >= 2:
                    angular_velocity = calculate_angular_velocity(angles_history, frame_times_history)
                    angular_velocities_history.append(angular_velocity) # Guardar para el promedio

                cv2.putText(full_frame, f"Centroide M: {centroid}", (10, full_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(full_frame, f"Area: {area}", (10, full_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(full_frame, "No se detecto mancha blanca", (10, full_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(full_frame, f"Frame: {current_frame_number}/{total_frames-1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(full_frame, f"Vel. Angular: {angular_velocity:.2f} rad/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(full_frame, "Pulsa 'ESPACIO' para pausar/reanudar, 'Q' para salir.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(MAIN_WINDOW_NAME, full_frame)
        
        key = cv2.waitKey(int(2500/fps)) & 0xFF # Reproducción a la velocidad real del video

        if key == ord('q'):
            break
        elif key == ord(' '): # Espacio para pausar/reanudar
            is_paused = not is_paused
            if is_paused:
                print("Video PAUSADO. Pulsa 'ESPACIO' para reanudar.")
            else:
                print("Video REANUDADO.")
        
    cap.release()
    cv2.destroyAllWindows()

    # --- CÁLCULO Y MOSTRADO DEL PROMEDIO DE VELOCIDAD ANGULAR ---
    if angular_velocities_history:
        average_angular_velocity = np.mean(angular_velocities_history)
        print(f"\n--- RESULTADOS DEL ANÁLISIS ---")
        print(f"Velocidad Angular Promedio: {average_angular_velocity:.2f} rad/s")

        # --- GENERACIÓN DEL GRÁFICO DE VELOCIDAD ANGULAR vs. TIEMPO ---
        plt.figure(figsize=(10, 6))
        plt.plot(frame_times_history[1:], angular_velocities_history, marker='o', linestyle='-', color='b', markersize=4) # Empieza desde el segundo dato ya que el primero no tiene velocidad angular
        plt.title('Velocidad Angular en Función del Tiempo')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Velocidad Angular (rad/s)')
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--', label='Cero rad/s') # Línea de referencia en cero
        plt.legend()
        plt.show()
        # -----------------------------------------------------------------

    else:
        print("\nNo se pudieron calcular velocidades angulares. Asegúrate de que la mancha fue detectada.")


if _name_ == "_main_":
    # Asegúrate de que 'VideoRecortado.mp4' esté en la misma carpeta o proporciona la ruta completa
    run_angular_velocity_analysis('VideoRecortado.mp4')