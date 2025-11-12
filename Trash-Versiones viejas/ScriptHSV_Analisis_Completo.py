import cv2
import numpy as np

# --- CONFIGURACIÓN ---
#Local:
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
# video_path = 'g:/Otros ordenadores/Asus Laptop/Codes Unsta Laptop/Fisica II files/VideoRecortado.mp4'
window_name = "Segmentador y Analizador de Rango"
initial_tolerance = 35
HUE_MARGIN = 10 

# --- ¡NUEVO! ---
# Lista global para guardar todos los rangos (lower, upper)
rangos_guardados = []

# Cargar el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: no se pudo abrir el video en {video_path}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# --- Función para el clic del mouse (MODIFICADA para guardar rangos) ---
def on_click_floodfill(event, x, y, flags, param):
    """
    Se activa al hacer clic.
    1. Imprime info del píxel 'semilla'.
    2. Usa floodFill para segmentar.
    3. Analiza los píxeles de la región.
    4. Imprime el rango para ESE CLIC.
    5. ¡NUEVO! Guarda ese rango en la lista global 'rangos_guardados'.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param['frame']
        
        # Parte 1: Info del Píxel Semilla
        current_frame_num = cv2.getTrackbarPos("Frame", window_name)
        bgr_color = frame[y, x]
        b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
        hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h_seed, s_seed, v_seed = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])

        print("====================================")
        print(f"** Píxel Semilla Seleccionado **")
        print(f"Frame:    {current_frame_num} (Coords: x={x}, y={y})")
        print(f"Color HSV: [H={h_seed}, S={s_seed}, V={v_seed}]")
        print("------------------------------------")
        
        # Parte 2: Lógica de Flood Fill
        frame_copy = frame.copy()
        try:
            tolerance = cv2.getTrackbarPos("Tolerancia", window_name)
        except:
            tolerance = initial_tolerance

        diff = (tolerance, tolerance, tolerance)
        mask = np.zeros((height + 2, width + 2), np.uint8)

        print(f"Ejecutando Flood Fill con Tolerancia={tolerance}...")
        num_pixels_filled, _, _, _ = cv2.floodFill(frame_copy, mask, (x, y), (0, 255, 0), diff, diff, cv2.FLOODFILL_FIXED_RANGE)
        cv2.imshow(window_name, frame_copy)
        print("¡Segmentación completada!")

        # Parte 3: Analizar la Mancha Entera
        if num_pixels_filled > 0:
            print("Analizando rango de la mancha...")
            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_area = mask[1:-1, 1:-1]
            hsv_pixels_in_stain = hsv_frame[mask_area == 1]
            
            # Calcular los rangos para ESTE CLIC
            min_h = max(0, h_seed - HUE_MARGIN)
            max_h = min(179, h_seed + HUE_MARGIN)
            min_s = int(np.min(hsv_pixels_in_stain[:, 1]))
            max_s = int(np.max(hsv_pixels_in_stain[:, 1]))
            min_v = int(np.min(hsv_pixels_in_stain[:, 2]))
            max_v = int(np.max(hsv_pixels_in_stain[:, 2]))

            # Crear los arrays de este clic
            lower_range = np.array([min_h, min_s, min_v])
            upper_range = np.array([max_h, max_s, max_v])

            print("\n--- RANGO ÓPTIMO (Este Clic) ---")
            print(f"lower_white = {lower_range}")
            print(f"upper_white = {upper_range}")
            
            # --- ¡NUEVO! Guardar el resultado ---
            rangos_guardados.append((lower_range, upper_range))
            print(f"-> Rango guardado. (Total guardados: {len(rangos_guardados)})")
            
        else:
            print("No se encontraron píxeles con esa tolerancia.")
        
        print("====================================")


# --- Funciones de Trackbar (sin cambios) ---
def on_trackbar(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, on_click_floodfill, {'frame': frame})

def on_tolerance_trackbar(val):
    pass

# --- Configuración de la ventana ---
cv2.namedWindow(window_name)
cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)
cv2.createTrackbar("Tolerancia", window_name, initial_tolerance, 255, on_tolerance_trackbar)

print("\n--- INSTRUCCIONES (Versión Análisis Total) ---")
print("1. Usa 'a' y 'd' para navegar.")
print("2. Ajusta la 'Tolerancia'.")
print("3. Haz clic en las manchas blancas que quieras analizar.")
print("4. Cada clic guardará el rango óptimo de esa mancha.")
print("5. Cuando termines, presiona 'q' para salir.")
print("   EL PROGRAMA CALCULARÁ EL RANGO TOTAL (MIN/MAX) DE TODOS TUS CLICS.")

# Mostrar el primer frame
on_trackbar(0)

# --- Bucle principal (MODIFICADO AL SALIR) ---
while True:
    current_frame = cv2.getTrackbarPos("Frame", window_name)
    key = cv2.waitKey(20) & 0xFF
    
    if key == ord('q'):
        # --- ¡NUEVO! CÁLCULO FINAL AL SALIR ---
        print("\n====================================")
        print("Saliendo... Calculando el rango MÁXIMO de la sesión...")
        
        if not rangos_guardados:
            print("No se guardó ningún rango durante la sesión.")
        else:
            # Separar todos los 'lower' y 'upper'
            all_lowers = np.array([r[0] for r in rangos_guardados])
            all_uppers = np.array([r[1] for r in rangos_guardados])
            
            # Calcular el min de los lowers y el max de los uppers
            # axis=0 calcula el min/max para cada columna (H, S, V)
            final_lower = np.min(all_lowers, axis=0)
            final_upper = np.max(all_uppers, axis=0)
            
            print(f"\nSe analizaron {len(rangos_guardados)} rangos guardados.")
            print("\n--- ¡RANGO FINAL TOTAL (Min/Max)! ---")
            print(f"Copia este rango para abarcar TODAS tus selecciones:")
            print(f"lower_white = np.array([{final_lower[0]}, {final_lower[1]}, {final_lower[2]}])")
            print(f"upper_white = np.array([{final_upper[0]}, {final_upper[1]}, {final_upper[2]}])")
        
        print("====================================")
        break # Salir del bucle
    
    elif key == ord('a'):
        new_frame = max(current_frame - 1, 0)
        cv2.setTrackbarPos("Frame", window_name, new_frame)
        on_trackbar(new_frame)
        
    elif key == ord('d'):
        new_frame = min(current_frame + 1, total_frames - 1)
        cv2.setTrackbarPos("Frame", window_name, new_frame)
        on_trackbar(new_frame)

cap.release()
cv2.destroyAllWindows()