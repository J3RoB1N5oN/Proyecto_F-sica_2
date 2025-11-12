import cv2
import numpy as np

# --- 1. CONFIGURACIÓN INICIAL ---
# Reemplaza con la ruta a tu video. Usa barras '/' para evitar errores.
video_path = 'C:/Users/julio/Documents/UNSTA (laptop local)/Fisica II files/WhatsApp Video 2025-09-29 at 20.17.52.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

# Leemos el primer fotograma para seleccionar la región de interés (ROI)
ret, first_frame = cap.read()
if not ret:
    print("No se pudo leer el video.")
    exit()

# --- 2. SELECCIÓN INTERACTIVA DE LA REGIÓN DE INTERÉS (ROI) ---
# Se abrirá una ventana con el primer fotograma.
# Haz clic y arrastra el ratón para dibujar un rectángulo sobre la barra de metal.
# Cuando estés satisfecho con la selección, presiona ENTER o ESPACIO.
# Si quieres cancelar, presiona la tecla 'c'.
print("Selecciona la Región de Interés (ROI) y presiona ENTER.")
roi = cv2.selectROI("Selecciona la barra de metal y presiona ENTER", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Selecciona la barra de metal y presiona ENTER")

# Extraemos las coordenadas del ROI (x, y, ancho, alto)
x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
print(f"Región de interés seleccionada en: x={x}, y={y}, w={w}, h={h}")

# --- 3. PREPARACIÓN PARA EL ANÁLISIS ---
# Inicializa el sustractor de fondo
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Variables para el control frame a frame
frame_index = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- 4. BUCLE PRINCIPAL DE ANÁLISIS ---
while True:
    # Posicionamos el video en el fotograma actual
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    if not ret:
        print("Fin del video o error al leer el fotograma.")
        frame_index -=1 # Nos quedamos en el último fotograma válido
        continue

    # Recortamos el fotograma original a la región de interés (ROI)
    frame_roi = frame[y:y+h, x:x+w]

    # Aplicamos la detección de movimiento SOLO en la ROI
    foreground_mask = background_subtractor.apply(frame_roi)
    
    # Limpiamos el ruido de la máscara
    kernel = np.ones((5,5), np.uint8)
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)

    # Encontramos contornos de los objetos en movimiento dentro de la ROI
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujamos los resultados en el fotograma COMPLETO original
    # Primero, dibujamos un rectángulo azul para visualizar nuestra ROI
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for contour in contours:
        if cv2.contourArea(contour) > 100: # Umbral de área para ignorar ruido pequeño
            # Obtenemos las coordenadas del contorno RELATIVAS a la ROI
            (cx, cy, cw, ch) = cv2.boundingRect(contour)
            
            # MUY IMPORTANTE: Sumamos las coordenadas de la ROI para dibujar en el lugar correcto
            # del fotograma completo.
            cv2.rectangle(frame, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 255, 0), 2)
            
    # Mostramos información en pantalla
    info_text = f"Fotograma: {frame_index + 1}/{total_frames}"
    cv2.putText(frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Usa flechas <- -> para navegar. 'q' para salir.", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    # Mostramos el fotograma con los resultados
    cv2.imshow('Analisis del Motor', frame)

    # --- 5. CONTROL DE TECLADO ---
    # cv2.waitKey(0) pausa el script hasta que se presione una tecla
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'): # Tecla 'q' para salir
        break
    elif key == 83: # Flecha Derecha (el código puede variar)
        if frame_index < total_frames - 1:
            frame_index += 1
    elif key == 81: # Flecha Izquierda (el código puede variar)
        if frame_index > 0:
            frame_index -= 1
    # else:
        # print(f"Código de tecla presionado: {key}") # Descomenta esta línea si las flechas no funcionan

# --- 6. LIMPIEZA FINAL ---
cap.release()
cv2.destroyAllWindows()