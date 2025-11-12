import cv2
import numpy as np
import pandas as pd
import time

# --- 1. CONFIGURACIÓN ---
# ¡Ruta guardada de tu video ORIGINAL (con voltímetro)!
video_path = 'c:/Users/julio/Documents/Codes Unsta Laptop/Fisica II files/VideoCompleto.mp4' 
output_csv = 'voltaje_resultados_V3.csv' # Nuevo archivo de salida

# --- 2. DATOS DE CALIBRACIÓN (FINALES - ¡ESTÁN BIEN!) ---
ROI_DISPLAY = [606, 270, 782, 347] 
SCALE_FACTOR = 3 
UMBRAL = 32 # El valor que encontramos

# Plantilla de 7 segmentos (Basada en el '6' del Frame 0)
ABS_SEGMENT_MAP = {'a': (208, 49), 'b': (231, 77), 'c': (224, 133), 
                   'd': (192, 153), 'e': (172, 125), 'f': (179, 70), 
                   'g': (200, 99)}

# Orígenes de los dígitos (Basados en el Frame 0)
DIGIT_ORIGINS = [(174, 43), (288, 55), (398, 69)] 

# --- 3. PROCESAMIENTO AUTOMÁTICO DE CALIBRACIÓN ---
min_x = min(val[0] for val in ABS_SEGMENT_MAP.values())
min_y = min(val[1] for val in ABS_SEGMENT_MAP.values())
DIGIT_TEMPLATE_MAP = {
    key: (val[0] - min_x, val[1] - min_y) 
    for key, val in ABS_SEGMENT_MAP.items()
}
DIGIT_MAP = {
    (1, 1, 1, 1, 1, 1, 0): '0', (0, 1, 1, 0, 0, 0, 0): '1',
    (1, 1, 0, 1, 1, 0, 1): '2', (1, 1, 1, 1, 0, 0, 1): '3',
    (0, 1, 1, 0, 0, 1, 1): '4', (1, 0, 1, 1, 0, 1, 1): '5',
    (1, 0, 1, 1, 1, 1, 1): '6', (1, 1, 1, 0, 0, 0, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8', (1, 1, 1, 1, 0, 1, 1): '9'
}
# ----------------------------------------------------

def read_display(frame):
    """
    Toma un frame, recorta el display, lo procesa
    y devuelve el número leído como string.
    """
    try:
        x1, y1, x2, y2 = ROI_DISPLAY
        display_img = frame[y1:y2, x1:x2]
        gray_display = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)

        # ---
        # --- ¡AQUÍ ESTÁ LA CORRECCIÓN! ---
        # ---
        # Usamos THRESH_BINARY_INV para que los dígitos sean BLANCOS (255)
        # y el fondo sea NEGRO (0).
        _, binary_display = cv2.threshold(gray_display, UMBRAL, 255, cv2.THRESH_BINARY_INV)
        # ---
        # --- FIN DE LA CORRECCIÓN ---
        # ---

        new_width = int(binary_display.shape[1] * SCALE_FACTOR)
        new_height = int(binary_display.shape[0] * SCALE_FACTOR)
        scaled_display = cv2.resize(binary_display, (new_width, new_height), 
                                    interpolation=cv2.INTER_NEAREST)

        recognized_string = ""
        
        for i, (dig_x, dig_y) in enumerate(DIGIT_ORIGINS):
            segment_states = [] 
            for seg_name in sorted(DIGIT_TEMPLATE_MAP.keys()):
                (rel_x, rel_y) = DIGIT_TEMPLATE_MAP[seg_name]
                abs_x, abs_y = dig_x + rel_x, dig_y + rel_y
                
                if 0 <= abs_y < scaled_display.shape[0] and 0 <= abs_x < scaled_display.shape[1]:
                    # Ahora, los segmentos ON son > 0 (Blancos)
                    if scaled_display[abs_y, abs_x] > 0: 
                        segment_states.append(1)
                    else:
                        segment_states.append(0)
                else:
                    segment_states.append(0)
            
            states_tuple = tuple(segment_states)
            digit = DIGIT_MAP.get(states_tuple, '?')
            recognized_string += digit
            
            if i == 0: 
                recognized_string += "."

        return recognized_string

    except Exception as e:
        return "Error"

# --- 5. BUCLE PRINCIPAL DE ANÁLISIS ---
print(f"Abriendo video: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total de frames: {total_frames}, FPS: {fps:.2f}")

resultados = []
start_time = time.time()
cv2.namedWindow("Video Original con OCR (V3 Corregido)", cv2.WINDOW_NORMAL)

for frame_num in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        print(f"Fin del video en el frame {frame_num}.")
        break
    
    valor_leido = read_display(frame)
    tiempo_s = frame_num / fps
    
    resultados.append({
        'frame': frame_num,
        'tiempo_s': tiempo_s,
        'valor_display': valor_leido
    })
    
    if frame_num % 100 == 0:
        print(f"Procesando frame {frame_num}/{total_frames}... Valor: {valor_leido}")
        
    cv2.rectangle(frame, (ROI_DISPLAY[0], ROI_DISPLAY[1]), (ROI_DISPLAY[2], ROI_DISPLAY[3]), (0, 255, 0), 2)
    cv2.putText(frame, f"Display: {valor_leido}", (ROI_DISPLAY[0], ROI_DISPLAY[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Video Original con OCR (V3 Corregido)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        print("Análisis cancelado por el usuario.")
        break

end_time = time.time()
print("\n--- ANÁLISIS DE VOLTÍMETRO COMPLETADO ---")
print(f"Tiempo total: {end_time - start_time:.2f} segundos.")

# --- 6. GUARDAR EN CSV ---
if not resultados:
    print("No se procesó ningún frame.")
else:
    df = pd.DataFrame(resultados)
    df.to_csv(output_csv, index=False)

    print(f"Resultados guardados exitosamente en: {output_csv}")
    print("\nContenido de muestra (filtrado para ver cambios):")
    
    df_changes = df[df['valor_display'] != df['valor_display'].shift()]
    print(df_changes)

cap.release()
cv2.destroyAllWindows()