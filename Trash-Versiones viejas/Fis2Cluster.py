import cv2
import math
import numpy as np

# --- Configuración ---
# ATENCIÓN: Debes cambiar esto a la ruta correcta de tu videoq
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
clicked_points = {} 
puntos_ciclo_actual = {}
punto_marcado_en_frame_actual = False

# --- VALORES PREDEFINIDOS ---
centro_rotacion = (280,280)
estado_actual = "MARCAR_PUNTOS"
posicion_radio_izquierda = (90,280)
posicion_radio_derecha = (460,280)
posicion_radio_superior = (280,80)
radio_promedio = 0

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

# --- REEMPLAZA ESTA FUNCIÓN EN TU CÓDIGO Fis2Cluster.py ---

def encontrar_mancha_cluster(frame, centro, radio):
    """
    Filtro de blancos (V8) - Implementado con connectedComponents
    y limpieza morfológica (Opening) para eliminar ruido de reflejos.
    """
    
    # 1. Convertir a HSV (Igual que antes)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Filtro de "Blanco Puro" (Usa los valores de tu ColorSampler)
    # (Pongo los últimos que probamos, pero reemplázalos si tienes nuevos)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([179, 25, 255])
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    
    
    # --- CAMBIO 1: Morfología de APERTURA (OPENING) ---
    # Esto elimina el ruido "sal" (puntos blancos pequeños)
    # que son la causa más probable del problema.
    kernel = np.ones((3,3), np.uint8) # Un kernel de 3x3
    mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # --- FIN CAMBIO 1 ---

    
    # 3. Encontrar clusters usando la MÁSCARA LIMPIA
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_limpia, connectivity=8) 
    
    candidatos = []
    
    tolerancia_radio = 15.0 
    centro_x, centro_y = centro 

    # 4. Iterar sobre los clusters
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # --- CAMBIO 2: Filtro de Área Mínima ---
        # Subimos el mínimo. Si 1-2 píxeles de ruido pasan, 
        # este filtro los elimina.
        # Si tu mancha real desaparece, prueba bajando esto a 3 o 4.
        if area < 5: 
            continue
        # --- FIN CAMBIO 2 ---
            
        # Filtro de área MÁXIMA (se mantiene)
        if area > 35: 
            continue
        
        # Obtener centro del cluster (se mantiene)
        cX, cY = centroids[i]
        
        # Filtro de Semicírculo (se mantiene)
        if cY >= centro_y:
            continue 

        # Filtro de ubicación (se mantiene)
        dist_actual = math.dist(centro, (cX, cY))
        dif_radio = abs(dist_actual - radio)
        
        if dif_radio <= tolerancia_radio:
            candidatos.append((area, i)) 

    # 5. Selección (se mantiene)
    if candidatos:
        candidatos.sort(key=lambda x: x[0]) 
        best_label_index = candidatos[0][1]     
        
        x = stats[best_label_index, cv2.CC_STAT_LEFT]
        y = stats[best_label_index, cv2.CC_STAT_TOP]
        w = stats[best_label_index, cv2.CC_STAT_WIDTH]
        h = stats[best_label_index, cv2.CC_STAT_HEIGHT]
        
        # Devolvemos la MÁSCARA LIMPIA para depuración
        return (x, y, w, h), mask_limpia
    else:
        # Devolvemos la MÁSCARA LIMPIA para depuración
        return None, mask_limpia

# --- Función para dibujar todo en el frame ---
def show_frame_with_markers(frame, frame_number):
    # Dibuja el centro
    if centro_rotacion:
        cv2.circle(frame, centro_rotacion, 7, (0, 255, 0), -1)
        cv2.putText(frame, 'CENTRO', (centro_rotacion[0] + 10, centro_rotacion[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(frame, (0, centro_rotacion[1]), (frame.shape[1], centro_rotacion[1]), (255, 0, 255), 1)
        cv2.line(frame, (centro_rotacion[0], 0), (centro_rotacion[0], frame.shape[0]), (255, 0, 255), 1)
    
    # --- DIBUJO DEL SEMICÍRCULO Y RADIOS ---
    if posicion_radio_izquierda: cv2.circle(frame, posicion_radio_izquierda, 5, (255, 0, 255), -1)
    if posicion_radio_derecha: cv2.circle(frame, posicion_radio_derecha, 5, (255, 0, 255), -1)
    if posicion_radio_superior: cv2.circle(frame, posicion_radio_superior, 5, (255, 0, 255), -1)
    
    if radio_promedio > 0 and centro_rotacion:
        cv2.line(frame, centro_rotacion, posicion_radio_izquierda, (255, 100, 255), 1)
        cv2.line(frame, centro_rotacion, posicion_radio_derecha, (255, 100, 255), 1)
        cv2.line(frame, centro_rotacion, posicion_radio_superior, (255, 100, 255), 1)
        radio_dibujo = int(radio_promedio)
        cv2.ellipse(frame, centro_rotacion, (radio_dibujo, radio_dibujo), 0, 180, 360, (0, 255, 255), 2)

    # Dibuja el punto de seguimiento del frame actual (si se marcó manually)
    if frame_number in puntos_ciclo_actual:
        point = puntos_ciclo_actual[frame_number]
        if centro_rotacion:
            cv2.line(frame, centro_rotacion, point, (255, 255, 0), 1)
        cv2.drawMarker(frame, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    # --- DIBUJAR RECTÁNGULO DE CLUSTER (CON DEPURACIÓN) ---
    if estado_actual == "MARCAR_PUNTOS" and radio_promedio > 0 and centro_rotacion:
        
        # Corrección del error 'cvtColor': Leer frame limpio
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_o, frame_original_limpio = cap.read() 
        
        if ret_o:
            # --- ACTUALIZACIÓN DE LLAMADA ---
            # Llamamos a la nueva función
            rectangulo, mask_debug = encontrar_mancha_cluster(frame_original_limpio, centro_rotacion, radio_promedio)
            # --- FIN ACTUALIZACIÓN ---
            
            # Mostramos la máscara para depuración
            if mask_debug is not None:
                cv2.imshow("Mascara de Deteccion", mask_debug)
                
            if rectangulo is not None:
                (x, y, w, h) = rectangulo
                # Dibujamos el rectángulo sobre 'frame' (la copia que se mostrará)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # --- FIN DEL CAMBIO ---

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
            print("Puede marcar puntos manualmente con el clic.")
        
        elif estado_actual == "MARCAR_PUNTOS":
            frame_key = current_frame
            clicked_points[frame_key] = (x, y)
            puntos_ciclo_actual[frame_key] = (x, y)
            punto_marcado_en_frame_actual = True
            print(f"Marca manual añadida en frame {frame_key}: ({x}, {y}).")
    
    on_trackbar(current_frame)

# --- Función callback para la trackbar ---
def on_trackbar(val):
    global current_frame, punto_marcado_en_frame_actual
    current_frame = val
    punto_marcado_en_frame_actual = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        show_frame_with_markers(frame.copy(), current_frame) # Usar una copia

# --- Creación de la ventana y controles ---
cv2.namedWindow("Visor de video")
cv2.createTrackbar("Frame", "Visor de video", 0, total_frames - 1, on_trackbar)
cv2.setMouseCallback("Visor de video", mouse_callback)
# Se crea la ventana de depuración
cv2.namedWindow("Mascara de Deteccion") 
on_trackbar(0)

# --- CÁLCULO INICIAL CON VALORES FIJOS ---
print("--- Usando valores fijos predefinidos para centro y radio. ---")
calcular_y_mostrar_radios()
print("--- ¡Listo! Modo 'Marcar Puntos' activado. ---")
on_trackbar(0) # Refrescar la ventana con el radio dibujado
# ----------------------------------------

# --- Bucle principal de Interacción ---
print("\n--- Controles ---")
print("Valores de centro y radio CARGADOS por defecto.")
print(" 'r': Iniciar re-calibración manual (centro y bordes).")
print("-----------------")
print(" Click izquierdo: Marcar un punto (en modo de marcado).")
print(" 'a' / 'd': Navegar frames.")
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
            
    elif key == ord('r'):
        # RE-CALIBRACIÓN MANUAL
        estado_actual = "DEFINIR_CENTRO"
        centro_rotacion = None
        posicion_radio_izquierda = None
        posicion_radio_derecha = None
        posicion_radio_superior = None
        radio_promedio = 0
        print("\n--- Modo 'Definir Centro' activado. ---")
        on_trackbar(current_frame)
    
    elif key == ord('b'):
        # Se mantiene por si 'r' es presionado
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

# --- Cierre de OpenCV ---
cap.release()
cv2.destroyAllWindows()