import cv2
import numpy as np

# --- PASO 1: Variables Globales ---
# Almacenará el (x, y) del clic
clicked_point = None
# Almacenará los rangos del color seleccionado
# Empezamos con un rango por defecto (blanco brillante)
target_lower = np.array([0, 0, 200]) 
target_upper = np.array([179, 50, 255])
# Variable para saber si ya seleccionamos un color
color_selected = False

# --- PASO 2: Función de Callback del Mouse ---
def click_event(event, x, y, flags, params):
    """Manejador de eventos del mouse."""
    global clicked_point, color_selected
    # Si se hace clic izquierdo
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        color_selected = True # Marcamos que se ha seleccionado un color
        print(f"Clic detectado en ({x}, {y})")

# !! IMPORTANTE: Asegúrate de que la ruta del video sea correcta !!
#Local:
# video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
#Drive:
video_path = 'g:/Otros ordenadores/Asus Laptop/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
cap = cv2.VideoCapture(video_path)

# Creamos la ventana y le asignamos el callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_event)

print("Haz clic en el video para seleccionar el color 'blanco' a rastrear.")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el video o fin del video.")
        break

    # Convertimos el frame a espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- PASO 3: Actualizar el color objetivo si se hizo clic ---
    # global target_lower, target_upper
    
    # Si la variable clicked_point tiene coordenadas
    if clicked_point is not None:
        # Obtenemos el BGR del píxel clickeado
        # Nota: en numpy es (fila, columna) -> (y, x)
        bgr_color = frame[clicked_point[1], clicked_point[0]] 
        
        # Convertimos ese único píxel BGR a HSV
        # cvtColor espera un array 3D, por eso los [[]]
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        H, S, V = hsv_color[0], hsv_color[1], hsv_color[2]
        print(f"Color BGR seleccionado: {bgr_color} -> Color HSV: [{H}, {S}, {V}]")

        # Definimos un rango de tolerancia
        # Puedes ajustar estos valores de tolerancia
        h_tolerance = 10  # Tolerancia para Matiz (Hue)
        sv_tolerance = 65 # Tolerancia para Saturación y Valor (más alta)
        
        target_lower = np.array([max(0, H - h_tolerance), max(0, S - sv_tolerance), max(0, V - sv_tolerance)])
        target_upper = np.array([min(179, H + h_tolerance), min(255, S + sv_tolerance), min(255, V + sv_tolerance)])
        
        print(f"Nuevo rango Inferior (lower): {target_lower}")
        print(f"Nuevo rango Superior (upper): {target_upper}")

        # Reseteamos el punto de clic para no recalcular en cada frame
        clicked_point = None

    # --- PASO 4: Tracking (Solo si ya se seleccionó un color) ---
    if color_selected:
        # Creamos la máscara usando el rango (por defecto o el seleccionado)
        mask = cv2.inRange(hsv, target_lower, target_upper)
        
        # Opcional: Limpiamos la máscara para reducir ruido
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Encontramos el contorno más grande (asumimos que es nuestra mancha)
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            # Calculamos el centroide (Centro de Masa)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Dibujamos el centroide en el frame original
                cv2.circle(frame, (cX, cY), 7, (0, 255, 0), -1)
                cv2.putText(frame, "Centro", (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # ---
                # --- AQUÍ VA LA LÓGICA DEL PROFESOR (8 PÍXELES) ---
                # ---
                # Una vez que tenemos el centro (cX, cY), podemos analizar
                # los 8 píxeles que lo rodean.
                #
                # Ejemplo de cómo acceder a los vecinos (en el frame HSV):
                #
                # for dy in [-1, 0, 1]:
                #     for dx in [-1, 0, 1]:
                #         if dx == 0 and dy == 0:
                #             continue # Este es el píxel central, lo saltamos
                #         
                #         # Coordenadas del vecino
                #         vecino_x, vecino_y = cX + dx, cY + dy
                #
                #         # (Añadir comprobación de que no se sale de los bordes del frame)
                #         if 0 <= vecino_y < frame.shape[0] and 0 <= vecino_x < frame.shape[1]:
                #             valor_hsv_vecino = hsv[vecino_y, vecino_x]
                #             # print(f"Vecino en ({vecino_x}, {vecino_y}) tiene HSV: {valor_hsv_vecino}")
                #             # ... Aquí harías tu análisis ...
                #
                
            
        # Mostramos la máscara (útil para debug)
        cv2.imshow('Mask', mask)

    # Mostramos el frame final
    cv2.imshow('Frame', frame)

    # Salir con la tecla 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- Limpieza final ---
cap.release()
cv2.destroyAllWindows()