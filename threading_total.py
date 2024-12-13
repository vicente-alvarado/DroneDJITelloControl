import random
from djitellopy import Tello
import cv2
from ultralytics import YOLO
import time
import socket
import threading  # Para tareas en paralelo

# Inicializa el dron
tello = Tello()
tello.connect()
tello.streamon()  # Inicia el streaming de video

# Configuración de la comunicación UDP con la ESP32
esp32_ip = "192.168.4.1"  # Reemplaza con la IP de la ESP32
esp32_port = 12345
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Cargar el modelo YOLOv8
model = YOLO("models/yolov8n.pt")

# Función para generar colores aleatorios
def generate_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Función para manejar la ruta predeterminada del dron (rectángulo)
def drone_route():
    try:
        # Despegar
        tello.takeoff()

        # Dimensiones del rectángulo (en cm)
        longitud = 90  # Longitud del rectángulo
        anchura = 40   # Anchura del rectángulo

        # Movimiento en forma de rectángulo
        tello.move_forward(longitud)
        tello.rotate_clockwise(90)
        tello.move_forward(anchura)
        tello.rotate_clockwise(90)
        tello.move_forward(longitud)
        tello.rotate_clockwise(90)
        tello.move_forward(anchura)
        tello.rotate_clockwise(90)

        # Aterrizar después de completar la ruta
        tello.land()
    except KeyboardInterrupt:
        print("Interrupción detectada en la ruta. Aterrizando...")
        tello.land()
    finally:
        tello.land()

# Función principal para visión con YOLO
def drone_vision():
    # Variables de inicialización
    sample_time = 0.1
    pos_x, pos_y, pos_z = 0, 0, 0  # Posición acumulada en cada eje
    class_colors = {}

    try:
        while True:
            # Verifica el nivel de batería
            battery_level = tello.get_battery()
            if battery_level <= 0:
                print("Batería baja. Apagando el dron.")
                break

            # Obtiene el frame del video
            frame = tello.get_frame_read().frame
            if frame is not None:
                # Aplica el modelo YOLOv8 para detectar objetos
                results = model(frame)

                # Dibuja las cajas de los objetos detectados
                for box, label, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                    label = int(label.item())
                    if label not in class_colors:
                        class_colors[label] = generate_random_color()

                    color = class_colors[label]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{model.names[label]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Enviar comando a la ESP32 si se detecta una clase específica
                    if model.names[label] == "person":
                        command = "DETECCION_PERSONA"
                        try:
                            udp_sock.sendto(command.encode(), (esp32_ip, esp32_port))
                            print(f"Comando enviado a ESP32: {command}")
                        except Exception as e:
                            print(f"Error al enviar el comando: {e}")
                
                # Obtén la velocidad en cada eje
                speed_x = tello.get_speed_x()
                print(speed_x)
                speed_y = tello.get_speed_y()
                speed_z = tello.get_speed_z()
                
                # Calcular el desplazamiento acumulado en cada eje
                pos_x += speed_x * sample_time
                pos_y += speed_y * sample_time
                pos_z -= speed_z * sample_time

                # Muestra la altitud, coordenadas estimadas y nivel de batería en el frame
                cv2.putText(frame, f'Bateria: {battery_level}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Posicion: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}) cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Muestra el video con las detecciones
                cv2.imshow("Tello Video Stream - YOLO Detection", frame)
            
                # Espera el tiempo de muestreo antes del siguiente cálculo
                time.sleep(sample_time)
            else:
                print("No se pudo capturar el frame.")

            # Salir con tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupción detectada. Aterrizando el dron...")
        tello.land()
    finally:
        cv2.destroyAllWindows()
        tello.streamoff()
        udp_sock.close()

# Ejecuta las tareas en paralelo
if __name__ == "__main__":
    try:
        # Crea dos hilos: uno para la ruta y otro para la visión
        vision_thread = threading.Thread(target=drone_vision)
        route_thread = threading.Thread(target=drone_route)

        # Inicia ambos hilos
        vision_thread.start()
        route_thread.start()

        # Espera a que ambos hilos terminen
        vision_thread.join()
        route_thread.join()
    except KeyboardInterrupt:
        print("Programa interrumpido. Aterrizando...")
        tello.land()
