import cv2
import numpy as np
import csv
from datetime import datetime 
import time
import glob
from geopy.geocoders import Nominatim
from geopy.location import Location
from geopy.distance import geodesic
from copy import deepcopy

########################################################## TP FINAL ##########################################################################

# PUNTO 7

def reconocer_color(img, tipo_articulo: str, cinta_transportadora: dict) -> dict:
    """
    Recibe la imagen a estudiar, el tipo de artículo y el diccionario "cinta transportadora" con sus contadores.
    Devuelve el diccionario "cinta transportadora" con sus respectivos contadores actualizados.
    """

    # Lee la imagen y la convierte en HSV usando cvtColor():
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_resultante: str = ''

    # Creo el diccionario para cada color con umbrales bajos y altos para luego completar la máscara.
  
    colores = {'azul': [np.array([98,50,20]), np.array([139,255,255])],
              'verde': [np.array([40,50,20]), np.array([75,255,255])],
              'amarillo': [np.array([20,100,20]), np.array([30,255,255])],
              'rojo': [np.array([0, 50, 20]), np.array([10, 255, 255])],
              'negro': [np.array([0, 0, 0]), np.array([180, 255, 30])]}

    for color in colores: 

        mascara = cv2.inRange(hsv_img, colores[color][0], colores[color][1])
        pixeles = cv2.countNonZero(mascara)        

        if (pixeles > 0):
            color_resultante = color

            if tipo_articulo == "vaso" and color_resultante in ['negro', 'azul']:
                cinta_transportadora[f'{tipo_articulo} {color_resultante}'] += 1

            elif tipo_articulo == "botella":
                cinta_transportadora[f'{tipo_articulo} {color_resultante}'] += 1

    return cinta_transportadora

def reconocimiento_de_objetos(cinta_transportadora: dict) -> dict:
    """Recibe el diccionario de cinta transportadora para luego detectar los objetos y diferenciarlos, siendo válidas las botellas y los vasos."""

    # Recorro imagen por imagen de la carpeta de Lote
    for img in glob.glob("Lote0001/*.jpg"):
        img = cv2.imread(img)
        # Carga los nombres de las clases y obtiene  los colores.
        classes = open('coco.names').read().strip().split('\n')
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

        # Determina la red neuronal a través de los archivos de configuración (.cfg) y de peso (.weights)
        net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        # Determina la capa de salida.
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # Cada imagen es construida bajo las mismas dimensiones para luego ser analizadas con el mismo formato.
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]

        net.setInput(blob)
        t0 = time.time()
        outputs = net.forward(ln)
        t = time.time()
        print('time=', t-t0)

        print(len(outputs))
        for out in outputs:
            print(out.shape)

        def trackbar2(x):
            confidence = x/100
            r = r0.copy()
            for output in np.vstack(outputs):
                if output[4] > confidence:
                    x, y, w, h = output[:4]
                    p0 = int((x-w/2)*416), int((y-h/2)*416)
                    p1 = int((x+w/2)*416), int((y+h/2)*416)
                    cv2.rectangle(r, p0, p1, 1, 1)
            cv2.imshow('blob', r)

        r0 = blob[0, 0, :, :]
        r = r0.copy()

        cajas = []
        confidences = [] #exactitud
        classIDs = [] #ids de los objetos
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    caja = detection[:4] * np.array([w, h, w, h])
                    (centroX, centroY, ancho, altura) = caja.astype("int")
                    x = int(centroX - (ancho / 2))
                    y = int(centroY - (altura / 2))
                    caja = [x, y, int(ancho), int(altura)]
                    cajas.append(caja)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(cajas, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (cajas[i][0], cajas[i][1])
                (w, h) = (cajas[i][2], cajas[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                texto = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(img, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                articulo_detectado = f"{classes[classIDs[i]]}"  # Nombre del articulo

                # Agregar articulo al inventario y llamar a función para reconocer su color o mostrar señal de animal
                if articulo_detectado == 'bottle' or articulo_detectado == 'tie': 
                    tipo_articulo = 'botella'
                    reconocer_color(img, tipo_articulo, cinta_transportadora)

                elif articulo_detectado == 'refrigerator' or articulo_detectado == 'vase':
                    tipo_articulo = 'vaso'
                    reconocer_color(img, tipo_articulo, cinta_transportadora)

                elif articulo_detectado == 'cow' or articulo_detectado == 'cat' or articulo_detectado == 'dog':
                    print('PROCESO DETENIDO. Remueva el animal de la cinta por favor.')

                else:
                    tipo_articulo = 'vaso' # 00 blue glass bug
                    reconocer_color(img, tipo_articulo, cinta_transportadora)

    return cinta_transportadora


def escribir_archivo_botellas(botellas: dict) -> None:
    """Recibe el diccionario de botellas para luego escribir en un archivo .txt el color y la cantidad."""
    with open("botellas.txt", newline='', mode='w', encoding="UTF-8") as archivo_txt:
        csv_writer = csv.writer(archivo_txt, delimiter=' ', quotechar='"')
        for color, cantidad in botellas.items():
            csv_writer.writerow([color, cantidad])
            
def escribir_archivo_vasos(vasos: dict) -> None:
    """Recibe el diccionario de vasos para luego escribir en un archivo .txt el color y la cantidad"""
    with open("vasos.txt", newline='', mode='w', encoding="UTF-8") as archivo_txt:
        csv_writer = csv.writer(archivo_txt, delimiter=' ', quotechar='"')
        for color, cantidad in vasos.items():
            csv_writer.writerow([color, cantidad])

def analizar_productos(cinta_transportadora: dict):
    """Recibe el diccionario de cinta transportadora para analizar y diferenciar las botellas y vasos."""
    botellas: dict = {}
    vasos: dict = {}
    for producto, cantidad in cinta_transportadora.items():
        lista_producto: list = producto.split()
        color: str = lista_producto[1].capitalize()
        if lista_producto[0] == "botella":
            botellas[color] = cantidad
            escribir_archivo_botellas(botellas)
        if lista_producto[0] == "vaso":
            vasos[color] = cantidad
            escribir_archivo_vasos(vasos)

# PUNTO 1

def leer_archivo() -> list:
    """Lee el archivo 'pedidos.csv' y devuelve los datos."""
    datos = list()
    nombre_archivo: str = 'pedidos.csv'

    try:
        with open(nombre_archivo, newline='', encoding="UTF-8") as archivo_csv:
            csv_reader = csv.reader(archivo_csv, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                if len(row) == 0:
                    continue
                datos.append(row)

    except FileNotFoundError:
        print(f"No se encontró el archivo {nombre_archivo}")

    except IOError as exce:
        print(f"Error al leer el archivo {nombre_archivo}: {exce.strerror}")
        
    finally:
        return datos

def crear_pedido(indice: int,datos: list) -> dict:
    """Recibe el índice y los datos para luego crear el pedido con todas sus características."""
    pedido: dict= {}
    pedido["Fecha"] = datos[indice][1]
    pedido["Cliente"] = datos[indice][2]
    pedido["Ciudad"] = datos[indice][3]
    pedido["Provincia"] = datos[indice][4]
    pedido["Cod. Artículo"] = []
    pedido["Cod. Artículo"].append(datos[indice][5])
    pedido["Color"] = []
    pedido["Color"].append(datos[indice][6])
    pedido["Cantidad"] = []
    pedido["Cantidad"].append(datos[indice][7])
    pedido["Descuento"] = []
    pedido["Descuento"].append(datos[indice][8])
    pedido["Latitud"] = 0
    return pedido

def generar_pedidos(datos: list) -> list:
    """Recibe los datos para luego ir agregando cada pedido en la lista de pedidos y devolverlo."""
    lista_pedidos: list = []

    if len(datos) == 0:
        return lista_pedidos
    else:
        pedido: dict = crear_pedido(0, datos) #para el primer caso usas el 0 (el primer elemento o linea del archivo) como indice porque necesito tenerlo

        for indice in range(1,len(datos)):
            if datos[indice-1][0] == datos[indice][0]:
                pedido["Cod. Artículo"].append(datos[indice][5])
                pedido["Color"].append(datos[indice][6])
                pedido["Cantidad"].append(datos[indice][7])
                pedido["Descuento"].append(datos[indice][8])
            else:
                lista_pedidos.append(pedido)
                pedido = crear_pedido(indice,datos)  

        lista_pedidos.append(pedido)
            
        return lista_pedidos

def agregar_codigo_pedidos_inicial(lista_pedidos: list) -> list:
    """Recibe la lista de pedidos para luego agregarle el código de pedido inicial a cada pedido."""
    contador_codigo_pedido: int = 1
    for pedido in lista_pedidos:
        pedido["Codigo de pedido"] = contador_codigo_pedido
        contador_codigo_pedido = contador_codigo_pedido + 1
    return lista_pedidos

def agregar_codigo_pedido(pedido: dict,codigo_pedido: int) -> dict:
    """Recibe el pedido y el código del pedido para asignarle el correspondiente valor al diccionario."""
    pedido["Codigo de pedido"] =  codigo_pedido
    return pedido

def color_por_numero(ingresar_numero:str)-> str:
    """Se crea un diccionario de colores para que al momento del ingreso por el usuario, retorne el color y no el numero"""
    colores: dict = {'1':'verde','2':'rojo','3':'azul','4':'negro','5':'amarillo','6':'negro','7':'azul' }
    return colores[ingresar_numero]

def ingresar_colores(lista_colores: list, codigo_articulo: str) -> list:
    """recibe la lista de colores y el codigo de color, se valida lo que ingresa el usuario y se devuelve la lista de colores"""
    if codigo_articulo == '1334' or codigo_articulo == '568':
        ingresar_color: str = input("\nIngrese el numero del color: "
                                        "\n BOTELLAS: "
                                        "\n 1. VERDE"
                                        "\n 2. ROJO"
                                        "\n 3. AZUL"
                                        "\n 4. NEGRO"
                                        "\n 5. AMARILLO"
                                        "\n VASOS: "
                                        "\n 6. NEGRO"
                                        "\n 7. AZUL \n")

        while not(validar_input(ingresar_color,['1','2','3','4','5','6','7'])):
            ingresar_color = input("\n Color incorrecto. \nIngrese el numero del color: "
                                        "\n BOTELLAS: "
                                        "\n 1. VERDE"
                                        "\n 2. ROJO"
                                        "\n 3. AZUL"
                                        "\n 4. NEGRO"
                                        "\n 5. AMARILLO"
                                        "\n VASOS: "
                                        "\n 6. NEGRO"
                                        "\n 7. AZUL \n")
        lista_colores.append(color_por_numero(ingresar_color))
    return lista_colores

def dar_de_alta_pedido() -> dict:
    """Devuelve el pedido armado con sus correspondientes características."""
    pedido: dict = {}
    pedido["Fecha"] = input("\nIngrese la fecha del pedido (DD/MM/YYYY): ")
    while not(validar_fecha(pedido["Fecha"])):
        pedido["Fecha"] = input("\nFormato incorrecto. Debe ser DD/MM/YYYY. \nIngrese la fecha del pedido: ")
    pedido["Cliente"] = input("\nIngrese el nombre del cliente: ")
    pedido["Ciudad"] = input("\nIngrese la ciudad de destino: ")
    pedido["Provincia"] = input("\nIngrese la provincia de destino: ")
    lista_articulos: list = []
    print(f"Codigo botellas: 1334 \n"
    "Codigo vasos: 568")
    ingresar_cod_articulo: str = input("\nIngrese el codigo de articulo: ")
    while not(validar_input(ingresar_cod_articulo,['1334','568'])):
        ingresar_cod_articulo = input("\n Codigo de articulo incorrecto."
                                                    "\nIngrese el codigo de articulo: ")
    lista_articulos.append(ingresar_cod_articulo)
    lista_colores: list = []
    if ingresar_cod_articulo == '1334':
        ingresar_colores(lista_colores, '1334')
    else:
        ingresar_colores(lista_colores, '568')
    lista_cantidad: list = []
    ingresar_cantidad: str = input("\nIngrese la cantidad: ")
    lista_cantidad.append(ingresar_cantidad)
    lista_descuento: list = []
    ingresar_descuento: str = input("\nIngrese el descuento: ")
    lista_descuento.append(ingresar_descuento)
    ingresar_mas_articulos: str = input("\nDesea agregar otro articulo? SI/NO: ").upper()
    while ingresar_mas_articulos == 'SI':
        print(f"\n Codigo botellas: 1334 \n"
        "\n Codigo vasos: 568")
        ingresar_cod_articulo = input("\nIngrese el codigo de articulo: ")
        while not(validar_input(ingresar_cod_articulo,['1334','568'])):
            ingresar_cod_articulo = input("\n Codigo de articulo incorrecto."
                                                        "\nIngrese el codigo de articulo: ")
        lista_articulos.append(ingresar_cod_articulo)
        if ingresar_cod_articulo == '1334':
            ingresar_colores(lista_colores, '1334')
        else:
            ingresar_colores(lista_colores, '568')
        ingresar_cantidad = input("\nIngrese la cantidad: ")
        lista_cantidad.append(ingresar_cantidad)
        ingresar_descuento = input("\nIngrese el descuento: ")
        lista_descuento.append(ingresar_descuento)
        ingresar_mas_articulos = input("\nDesea agregar otro articulo? SI/NO: ").upper()
    pedido["Cod. Artículo"] = lista_articulos
    pedido["Color"] = lista_colores
    pedido["Cantidad"] = lista_cantidad
    pedido["Descuento"] = lista_descuento
    pedido["Latitud"] = 0
    print(f"\nSu pedido es: ")
    for key, value in pedido.items() :
        print(key, ':', value)
    return pedido 

def validar_fecha(fecha):
    """Se valida que el usuario ingrese la fecha en un formato dia, mes, año"""
    try:
        return bool(datetime.strptime(fecha, '%d/%m/%Y'))
    except ValueError:
        return False


def validar_input(input: str, rango_a_validar: list)-> bool:
    """Se valida que lo que ingrese el usuario sea correcto"""
    for valor in rango_a_validar:
        if input == valor:
            return True
    return False

def buscar_pedido_por_codigo_pedido(lista_pedidos: list, codigo_de_pedido: int) -> dict:
    """Recibe la lista de pedidos y el código del pedido para comparar y buscar la similitud entre ellos, devolviendo el diccionario (pedido) en caso de encontrarlo. De lo contrario, no devuelve nada."""
    for pedido in lista_pedidos:
        if pedido["Codigo de pedido"] == codigo_de_pedido:
            return pedido
    return {}

def dar_de_baja_pedido(lista_pedidos: list, pedido: dict) -> list:
    """Recibe la lista de pedidos y el pedido a eliminar, devolviendo la lista de pedidos actualizada sin el pedido que se quiso remover."""
    lista_pedidos.remove(pedido)
    return lista_pedidos

def modificar_pedido(lista_pedidos: list, pedido: dict) -> list:
    """Recibe la lista de pedidos y el pedido a modificar para devolver la lista de pedidos con el pedido actualizado."""
    lista_pedidos = dar_de_baja_pedido(lista_pedidos, pedido)
    pedido_modificado: dict = dar_de_alta_pedido() #le digo a dar de alta, que tome el codigo de pedido ya creado 
    pedido_modificado_con_codigo_pedido: dict = agregar_codigo_pedido(pedido_modificado,pedido["Codigo de pedido"])
    lista_pedidos.append(pedido_modificado_con_codigo_pedido)
    return lista_pedidos

def validar_menu(opcion_minimo: int, opcion_maximo: int) -> int:
    """Permite al usuario ingresar determinados numeros enteros para validar el menu"""
    respuesta_usuario: str = input("\n INGRESE UNA OPCION: ")
    while not respuesta_usuario.isnumeric() or int(respuesta_usuario) > opcion_maximo or int(respuesta_usuario) < opcion_minimo:
        print(f"\n OPCION INCORRECTA, VUELVA A INGRESAR: ")
        respuesta_usuario = input("\n INGRESE NUEVAMENTE SU OPCION: ")
    return int(respuesta_usuario)

def mostar_menu(opciones_menu: list) -> None:
    """Recibe la lista con opciones del menu y hace un print de cada opcion"""
    print(f"\nQUE DESEA REALIZAR?: \n")
    contador_opciones_menu :int = 1
    for opcion in opciones_menu:
        print(f"{contador_opciones_menu}: {opcion} ")
        contador_opciones_menu = contador_opciones_menu + 1


def menu_abm(lista_pedidos: list) ->None:
    """Se crea un submenu solamente del abm"""
    cerrar_menu: bool = False
    nuevo_codigo_pedido: int = len(lista_pedidos) + 1
    submenu_abm: list = ['ALTA PEDIDO','DAR DE BAJA PEDIDO','MODIFICAR PEDIDO','CERRAR SUBMENU']
    while cerrar_menu == False:
        mostar_menu(submenu_abm)
        respuesta_usuario = validar_menu(1, len(submenu_abm))
        if respuesta_usuario == 1:
                pedido_nuevo: dict = dar_de_alta_pedido()
                pedido_nuevo = agregar_codigo_pedido(pedido_nuevo,nuevo_codigo_pedido)
                lista_pedidos.append(pedido_nuevo)
                nuevo_codigo_pedido = nuevo_codigo_pedido + 1
                guardar_archivo_actualizado(lista_pedidos)

        elif respuesta_usuario == 2:
                respuesta_usuario: int = int(input("\n Ingrese el codigo de pedido para dar de baja: "))
                lista_pedidos = dar_de_baja_pedido(lista_pedidos, buscar_pedido_por_codigo_pedido(lista_pedidos, respuesta_usuario))
                print(f"\nLa baja se realizo con exito\n")
                guardar_archivo_actualizado(lista_pedidos)

        elif respuesta_usuario == 3: 
            respuesta_modificar_pedido: int = int(input("\n Que pedido desea modificar?: "))
            lista_pedidos = modificar_pedido(lista_pedidos,buscar_pedido_por_codigo_pedido(lista_pedidos,respuesta_modificar_pedido))
            guardar_archivo_actualizado(lista_pedidos)
        
        else: 
            cerrar_menu = True
        
 
# PUNTO 2

def identificar_ciudades_en_zonas_geograficas(lista_pedidos: list) -> tuple:
    """Recibe la lista de pedidos para luego devolver una tupla con las ciudades que pertenecen a cada respectiva zona geográfica."""
    ciudades_zona_norte: list = []
    ciudades_zona_centro: list = []
    ciudades_zona_sur: list = []
    caba: list = ["CABA"]

    #Creo objeto "Nominatim"
    geolocator = Nominatim(user_agent="my-app")

    #Introduzco ubicación de cada ciudad
    for pedido in lista_pedidos:
        ubicacion_ciudad: Location = geolocator.geocode(pedido["Ciudad"])
        latitud_ciudad: float = ubicacion_ciudad.latitude
        pedido["Latitud"] = latitud_ciudad
    
    #Clasifico segun latitud
    for pedido in lista_pedidos:
        if 0 >= pedido["Latitud"] >= -35 or 0 <= pedido["Latitud"] <= 35:
            ciudades_zona_norte.append(pedido["Ciudad"])
        elif -35 >= pedido["Latitud"] >= -40 or 35 <= pedido["Latitud"] <= 40 :
            ciudades_zona_centro.append(pedido["Ciudad"])
        elif pedido["Latitud"] <= -40 or pedido["Latitud"] >= 40:
            ciudades_zona_sur.append(pedido["Ciudad"]) 

    #La latitud de CABA es -34, menor a 35 , por ende la elimino de la lista de ciudades de zona norte. No es necesario latitud para CABA ni ninguna lista
    if 'CABA' in ciudades_zona_norte:  
        ciudades_zona_norte.remove("CABA")

    tupla_ciudades: tuple = (ciudades_zona_norte, ciudades_zona_centro, ciudades_zona_sur, caba)

    return tupla_ciudades

def calcular_distancias(lista_pedidos: list) -> list:
    """Recibe la lista de pedidos para luego devolver una lista de diccionarios que contiene todas las distancias entre las ciudades, indicando origen, destino y kilómetros de distancia."""
    latitud_longitud: dict = {}
    distancias_entre_ciudades: list = []
    lista_pedidos_respaldo: list = lista_pedidos.copy()
    geolocator = Nominatim(user_agent="my-app")

    #Obtengo las tuplas de latitudes y longitudes que me van a servir para luego calcular la distancia entre cada una de las ciudades

    for pedido in lista_pedidos:
        ciudad = pedido["Ciudad"]
        tupla_latitud_longitud: tuple = (geolocator.geocode(ciudad).latitude, geolocator.geocode(ciudad).longitude)
        latitud_longitud[pedido["Ciudad"]] = tupla_latitud_longitud

    for pedido in lista_pedidos_respaldo:
        for otro_pedido in lista_pedidos_respaldo:
            distancia_ciudades: float = geodesic(latitud_longitud[pedido["Ciudad"]], latitud_longitud[otro_pedido["Ciudad"]]).kilometers
            comparacion: dict = {}
            comparacion["Origen"] = pedido["Ciudad"]
            comparacion["Destino"] = otro_pedido["Ciudad"]
            comparacion["Distancia"] = distancia_ciudades
            distancias_entre_ciudades.append(comparacion)
            if comparacion["Origen"] == comparacion["Destino"]:
                distancias_entre_ciudades.remove(comparacion)
    
    return distancias_entre_ciudades

# PUNTO 3

def agregar_distancias(distancias_entre_ciudades: list, lista_pedidos: list, ciudades_en_zona: list, distancias_zona: list, recorridos_optimos: dict) -> None:
    ciudades_zona_norte: list = identificar_ciudades_en_zonas_geograficas(lista_pedidos)[0]
    ciudades_zona_centro: list = identificar_ciudades_en_zonas_geograficas(lista_pedidos)[1]
    ciudades_zona_sur: list = identificar_ciudades_en_zonas_geograficas(lista_pedidos)[2]
    for recorrido in distancias_entre_ciudades:
        if recorrido["Origen"] == "CABA" and recorrido["Destino"] in ciudades_en_zona:
            distancias_zona.append(recorrido["Distancia"])
    distancias_zona.sort()
    for distancia in distancias_zona:
        for recorrido in distancias_entre_ciudades:
            if distancia == recorrido["Distancia"] and recorrido["Destino"] != "CABA":
                if ciudades_en_zona == ciudades_zona_norte:
                    recorridos_optimos["Zona Norte"].append(recorrido["Destino"])
                elif ciudades_en_zona == ciudades_zona_centro:
                    recorridos_optimos["Zona Centro"].append(recorrido["Destino"])
                elif ciudades_en_zona == ciudades_zona_sur:
                    recorridos_optimos["Zona Sur"].append(recorrido["Destino"])

def recorrer_caminos_optimos_punto_3(lista_pedidos: list) -> dict:
    """
    Función que se utiliza para obtener todos los recorridos optimos por zona.
    Devuelve un diccionario con el recorrido óptimo por distancia por zona.
    """
    ciudades_zona_norte: list = identificar_ciudades_en_zonas_geograficas(lista_pedidos)[0]
    ciudades_zona_centro: list = identificar_ciudades_en_zonas_geograficas(lista_pedidos)[1]
    ciudades_zona_sur: list = identificar_ciudades_en_zonas_geograficas(lista_pedidos)[2]
    recorridos_optimos: dict = {
        "Zona CABA": ["CABA"],
        "Zona Norte": [],
        "Zona Centro": [],
        "Zona Sur": []
    }
    distancias_zona_norte: list = []
    distancias_zona_centro: list = []
    distancias_zona_sur: list = []
    distancias_entre_ciudades: list = calcular_distancias(lista_pedidos)

    agregar_distancias(distancias_entre_ciudades, lista_pedidos, ciudades_zona_norte, distancias_zona_norte, recorridos_optimos)
    agregar_distancias(distancias_entre_ciudades, lista_pedidos, ciudades_zona_centro, distancias_zona_centro, recorridos_optimos)
    agregar_distancias(distancias_entre_ciudades, lista_pedidos, ciudades_zona_sur, distancias_zona_sur, recorridos_optimos)

    return recorridos_optimos

def preguntar_zona_a_recorrer(lista_pedidos: list) -> None:
    """Recibe la lista de pedidos e imprime el recorrido óptimo que elija el usuario según la zona."""

    zona_geografica_ingresada: str = input("""
    ZONAS GEOGRÁFICAS:
    - Zona Norte,
    - Zona Centro,
    - Zona Sur, 
    - CABA. 
    
    Ingrese la zona geográfica de destino: """).upper()

    while zona_geografica_ingresada != "CABA" and zona_geografica_ingresada != "ZONA NORTE" and zona_geografica_ingresada != "ZONA CENTRO" and zona_geografica_ingresada != "ZONA SUR":
        zona_geografica_ingresada = input("Ingrese una zona geográfica de destino correcta del listado: ")
    
    recorridos_optimos: dict = recorrer_caminos_optimos_punto_3(lista_pedidos)
    # Para cada zona geográfica, según sea la zona ingresada por el usuario, me fijo en cada distancia de la lista que comience con CABA y que el destino sea una ciudad de la zona ingresada por el usuario, para luego ordenarlo de manera ascendente, de esta manera ya queda ordenado por menor km de distancia y forma el recorrido óptimo a realizar.

    if zona_geografica_ingresada == "ZONA NORTE":
        print(", ".join(recorridos_optimos["Zona Norte"]))
    
    elif zona_geografica_ingresada == "CABA":
        print(", ".join(recorridos_optimos["Zona CABA"]))
    
    elif zona_geografica_ingresada == "ZONA CENTRO":
        print(", ".join(recorridos_optimos["Zona Centro"]))

    elif zona_geografica_ingresada == "ZONA SUR":
        print(", ".join(recorridos_optimos["Zona Sur"]))

def crear_diccionario_ciudades_y_pesos(pedido: dict,\
    ciudades_y_pesos_por_zona: dict, zona: str) -> None:
    '''
    Procedimiento que crea el diccionario que separa las ciudades por zonas
    con sus respectivos pesos asociados.
    '''
    for i in range(len(pedido["Cod. Artículo"])):
        if pedido["Ciudad"] in ciudades_y_pesos_por_zona[zona]:
            if pedido["Cod. Artículo"][i] == "1334":
                ciudades_y_pesos_por_zona[zona][pedido["Ciudad"]].\
                    append(int(pedido["Cantidad"][i]) * 0.45)
            elif pedido["Cod. Artículo"][i] == "568":
                ciudades_y_pesos_por_zona[zona][pedido["Ciudad"]].\
                    append(int(pedido["Cantidad"][i]) * 0.35)
        elif pedido["Ciudad"] not in ciudades_y_pesos_por_zona[zona]:
            if pedido["Cod. Artículo"][i] == "1334":
                ciudades_y_pesos_por_zona[zona][pedido["Ciudad"]] =\
                    [int(pedido["Cantidad"][i]) * 0.45]
            elif pedido["Cod. Artículo"][i] == "568":
                ciudades_y_pesos_por_zona[zona][pedido["Ciudad"]] =\
                    [int(pedido["Cantidad"][i]) * 0.35]


def crear_diccionario_ciudades(ciudades_y_pesos_por_zona: dict,\
    recorrido_optimo: dict, ciudades_por_zona: dict) -> None:
    '''
    Procedimiento que crea el diccionario que separa las ciudades por zonas.
    '''
    for zona, dict_ciudades in ciudades_y_pesos_por_zona.items():
        for ciudad, lista_codigos in dict_ciudades.items():
            for _ in lista_codigos:
                if ciudad in recorrido_optimo[zona]:
                    ciudades_por_zona[zona].append(ciudad)


def crear_diccionario_pesos(ciudades_y_pesos_por_zona: dict,\
    recorrido_optimo: dict, pesos_por_zona: dict) -> None:
    '''
    Procedimiento que crea el diccionario que separa los pesos por zonas.
    '''
    for zona, dict_ciudades in ciudades_y_pesos_por_zona.items():
        for ciudad, lista_codigos in dict_ciudades.items():
            for i in lista_codigos:
                if ciudad in recorrido_optimo[zona]:
                    pesos_por_zona[zona].append(i)


def crear_diccionarios_punto_3(lista_pedidos: list) -> tuple:
    '''
    Función que crea los diccionarios correspondientes al punto 3 y los devuelve.
    '''
    recorrido_optimo: dict = recorrer_caminos_optimos_punto_3(lista_pedidos)
    ciudades_y_pesos_por_zona: dict = {"Zona CABA": {}, "Zona Norte": {},
    "Zona Centro": {}, "Zona Sur": {}}
    pesos_por_zona: dict = {"Zona CABA": [], "Zona Norte": [], "Zona Centro": [], "Zona Sur": []}
    ciudades_por_zona: dict = {"Zona CABA": [], "Zona Norte": [], "Zona Centro": [], "Zona Sur": []}
    for zona in recorrido_optimo:
        for pedido in lista_pedidos:
            if pedido["Ciudad"] in recorrido_optimo[zona]:
                crear_diccionario_ciudades_y_pesos(pedido, ciudades_y_pesos_por_zona, zona)
    crear_diccionario_ciudades(ciudades_y_pesos_por_zona, recorrido_optimo, ciudades_por_zona)
    crear_diccionario_pesos(ciudades_y_pesos_por_zona, recorrido_optimo, pesos_por_zona)

    return pesos_por_zona, ciudades_por_zona


def asignar_utilitarios(pesos_por_zona: dict) -> tuple:
    '''
    Función que ordena los pesos por zona y en base a eso se le asigna un utilitario.
    Devuelve una tupla que contiene las zonas correspondientes a cada utilitario.
    '''
    lista_ordenada_por_peso: list = sorted(pesos_por_zona.items(),\
        key = lambda x: sum(x[1]), reverse = True)
    utilitario_4: str = lista_ordenada_por_peso[0][0]
    utilitario_2: str = lista_ordenada_por_peso[1][0]
    utilitario_1: str = lista_ordenada_por_peso[2][0]
    utilitario_3: str = lista_ordenada_por_peso[3][0]
    zonas_utilitarios: tuple = utilitario_4, utilitario_2, utilitario_1, utilitario_3

    return zonas_utilitarios


def crear_diccionario_utilitarios(zonas_utilitarios: tuple) -> dict:
    '''
    Función que crea un diccionario con las distintas zonas,
    asignándole a éstas su utilitario, peso y nombre correspondientes.
    Devuelve el diccionario.
    '''
    utilitarios_por_zona: dict = {}
    utilitarios_por_zona[zonas_utilitarios[0]] = [2000, "Utilitario 4"]
    utilitarios_por_zona[zonas_utilitarios[1]] = [1000, "Utilitario 2"]
    utilitarios_por_zona[zonas_utilitarios[2]] = [600, "Utilitario 1"]
    utilitarios_por_zona[zonas_utilitarios[3]] = [500, "Utilitario 3"]

    return utilitarios_por_zona


def descargar_peso_extra(pesos_por_zona: dict, ciudades_por_zona: dict) -> dict:
    '''
    Función que suelta parte de la carga del utilitario si esta sobrepasa su límite establecido.
    Devuelve un diccionario que contiene como a clave la ciudad y
    como valor una lista de pesos tirados.
    '''
    pesos_tirados_por_ciudad: dict = {}
    pesos_tirados_por_zona: dict = {"Zona CABA": [], "Zona Norte": [], "Zona Centro": [], "Zona Sur": []}
    suma_pesos: dict = {"Zona CABA": [], "Zona Norte": [], "Zona Centro": [], "Zona Sur": []}
    zonas_utilitarios: tuple = asignar_utilitarios(pesos_por_zona)
    utilitarios_por_zona: dict = crear_diccionario_utilitarios(zonas_utilitarios)
    for zona, datos in utilitarios_por_zona.items():
        for peso in pesos_por_zona[zona]:
            suma_pesos[zona].append(peso)
            if sum(suma_pesos[zona]) > datos[0]:
                indice: int = pesos_por_zona[zona].index(peso)
                ciudad: str = ciudades_por_zona[zona][indice]
                print(f"Cuidado, se ha dejado una carga que iba a la ciudad {ciudad} en {zona}.")
                print(f"La carga era de {peso} kg, el utilitario no soportaría el peso.")
                elemento_quitado: float = suma_pesos[zona].pop()
                pesos_tirados_por_zona[zona].append(elemento_quitado)
                if ciudad in pesos_tirados_por_ciudad:
                    pesos_tirados_por_ciudad[ciudad].append(peso)
                elif ciudad not in pesos_tirados_por_ciudad:
                    pesos_tirados_por_ciudad[ciudad] = [peso]

    for zona in utilitarios_por_zona:
        for i in reversed(pesos_por_zona[zona]):
            if i in pesos_tirados_por_zona[zona]:
                pesos_por_zona[zona].reverse()
                pesos_por_zona[zona].remove(i)
                pesos_por_zona[zona].reverse()
                pesos_tirados_por_zona[zona].remove(i)

    return pesos_tirados_por_ciudad


def crear_recorrido_optimo_por_carga(ciudades_por_zona: dict) -> dict:
    '''
    Función que crea un diccionario que contiene el recorrido óptimo según la carga del utilitario por zona.
    Cada utilitario parte de CABA. Devuelve el diccionario.
    '''
    recorrido_optimo_por_carga: dict = {"Zona CABA": ["CABA"], "Zona Norte": ["CABA"],
    "Zona Centro": ["CABA"], "Zona Sur": ["CABA"]}
    for zona in ciudades_por_zona:
        for elemento in ciudades_por_zona[zona]:
            if elemento not in recorrido_optimo_por_carga[zona]:
                recorrido_optimo_por_carga[zona].append(elemento)

    return recorrido_optimo_por_carga


def escribir_archivo_salida_txt(pesos_por_zona: dict, ciudades_por_zona: dict) -> None:
    '''
    Procedimiento que escribe el archivo informativo sobre los utilitarios en todas las zonas.
    '''
    zonas_utilitarios: tuple = asignar_utilitarios(pesos_por_zona)
    recorrido_optimo_por_carga: dict = crear_recorrido_optimo_por_carga(ciudades_por_zona)
    with open("salida.txt", newline='', mode='w', encoding="UTF-8") as archivo_csv:
        csv_writer = csv.writer(archivo_csv, delimiter='|', quotechar='"')
        numero_utilitario: int = 4
        for i in range(4):
            if i == 1:
                numero_utilitario = 2
            elif i == 2:
                numero_utilitario = 1
            elif i == 3:
                numero_utilitario = 3
            csv_writer.writerow([zonas_utilitarios[i]])
            csv_writer.writerow([f"Utilitario 00{numero_utilitario}"])
            csv_writer.writerow([f"{sum(pesos_por_zona[zonas_utilitarios[i]])}kg"])
            csv_writer.writerow([f"{', '.join(recorrido_optimo_por_carga[zonas_utilitarios[i]])}"])
    print("Tu archivo salida.txt ha sido creado correctamente.")


# PUNTO 4

def eliminar_pedidos_vacios(lista_pedidos_entregados: list) -> None:
    '''
    Procedimiento que elimina un pedido entregado en caso de que este haya sido
    vaciado por completo.
    '''
    for _ in range(2):
        for pedido_entregado in lista_pedidos_entregados:
            if pedido_entregado["Cod. Artículo"] == []:
                del pedido_entregado["Fecha"]
                del pedido_entregado["Cliente"]
                del pedido_entregado["Ciudad"]
                del pedido_entregado["Provincia"]
                del pedido_entregado["Cod. Artículo"]
                del pedido_entregado["Color"]
                del pedido_entregado["Cantidad"]
                del pedido_entregado["Descuento"]
                del pedido_entregado["Latitud"]
            while {} in lista_pedidos_entregados:
                lista_pedidos_entregados.remove({})


def crear_lista_pedidos_entregados(lista_pedidos: list, pesos_tirados_por_ciudad: dict) -> list:
    '''
    Función que crea una lista de pedidos entregados en base a la lista de pedidos inicial.
    Devuelve esta lista.
    '''
    lista_pedidos_entregados = deepcopy(lista_pedidos)
    for pedido_no_entregado in lista_pedidos_entregados:
        for cantidad in pedido_no_entregado["Cantidad"]:
            ciudad: str = pedido_no_entregado["Ciudad"]
            if pedido_no_entregado["Ciudad"] in pesos_tirados_por_ciudad and\
                (int(cantidad) * 0.45 in pesos_tirados_por_ciudad[ciudad] or\
                    int(cantidad) * 0.35 in pesos_tirados_por_ciudad[ciudad]):
                indice = pedido_no_entregado["Cantidad"].index(cantidad)
                pedido_no_entregado["Cod. Artículo"].pop(indice)
                pedido_no_entregado["Color"].pop(indice)
                pedido_no_entregado["Cantidad"].pop(indice)
                pedido_no_entregado["Descuento"].pop(indice)
    eliminar_pedidos_vacios(lista_pedidos_entregados)
    lista_pedidos_entregados.sort(key = lambda pedidos: datetime.strptime(pedidos["Fecha"], '%d/%m/%Y'))

    return lista_pedidos_entregados

def mostrar_pedidos_completos(lista_pedidos_entregados: list) -> None:
    """Recibe la lista de pedidos entregados para mostrar cada pedido con su código y la fecha, el cliente, la ciudad y la provincia."""
    contador_pedidos: int = 0
    print("LISTADO DE PEDIDOS ENTREGADOS ORDENADOS POR FECHA:\n")
    for pedido_entregado in lista_pedidos_entregados:
        print(f"""- Pedido número: {pedido_entregado["Codigo de pedido"]}
    Fecha: {pedido_entregado["Fecha"]},
    Cliente: {pedido_entregado["Cliente"]},
    Ciudad: {pedido_entregado["Ciudad"]},
    Provincia: {pedido_entregado["Provincia"]}
    """)

    for i in lista_pedidos_entregados:
        contador_pedidos += 1

    print(f"La cantidad total de pedidos entregados fueron: {contador_pedidos}")

# PUNTO 5

def averiguar_costo_vasos_botellas(lista_pedidos: list) -> None:
    """Recibe la lista de pedidos para agregarle a cada pedido (que es un diccionario) el costo total de botellas y vasos correspondiente."""
    precio_por_botella: int = 15
    precio_por_vaso: int = 8
    for pedido in lista_pedidos:
        for i in range(0, len(pedido["Cod. Artículo"])): #toma el indice de cada elemento de la lista de "Cod. Artículo" en el dict
            if pedido["Cod. Artículo"][i] == 1334:
                pedido["Costo de botellas"] = pedido["Cantidad"][i] * precio_por_botella
            else:
                pedido["Costo de vasos"] = pedido["Cantidad"][i] * precio_por_vaso

def listar_pedidos_en_rosario(lista_pedidos_entregados: list) -> None:
    """Recibe la lista de pedidos entregados y llama a la función que averigua el costo de los vasos y las botellas para luego encontrar los pedidos que se encuentran ubicados en Rosario y mostrar el costo total de vasos y botellas."""
    averiguar_costo_vasos_botellas(lista_pedidos_entregados)
    print(f"""
        A continuación, los siguientes pedidos fueron entregados en Rosario:""")
    for pedido in lista_pedidos_entregados:
        if pedido["Ciudad"] == "Rosario":
            print(f"""- El pedido de {pedido["Cliente"]} a un precio de:
    Costo total de vasos: {pedido["Costo de vasos"]}.
    Costo total de botellas: {pedido["Costo de botellas"]}""")

# PUNTO 6

def articulo_mas_pedido(lista_pedidos: list) -> None:
    """Recibe la lista de pedidos para realizar comparaciones entre las botellas y vasos con respectivos colores e imprimir el artículo más pedido."""
    acumulador_botellas_verdes: list = [0,'botella','verde']
    acumulador_botellas_azul: list = [0,'botella','azul']
    acumulador_botellas_rojo: list = [0,'botella','rojo']
    acumulador_botellas_negro: list = [0,'botella','negro']
    acumulador_botellas_amarillo: list = [0,'botella','amarillo']
    acumulador_vasos_negros: list = [0,'vaso','negro']
    acumulador_vasos_azules: list = [0,'vaso','azul']
    for pedido in lista_pedidos:
        articulos: list = pedido["Cod. Artículo"]
        colores: list = pedido["Color"]
        cantidad: list = pedido["Cantidad"]
        codigo_botella: str = "1334"
        for indice in range(0, len(articulos)):
            if articulos[indice] == codigo_botella:
                if colores[indice].lower() == 'verde':
                    acumulador_botellas_verdes[0] = acumulador_botellas_verdes[0] + int(cantidad[indice])
                elif colores[indice].lower() == 'rojo':
                    acumulador_botellas_rojo[0] = acumulador_botellas_rojo[0] + int(cantidad[indice])
                elif colores[indice].lower() == 'azul':
                    acumulador_botellas_azul[0] = acumulador_botellas_azul[0] + int(cantidad[indice])
                elif colores[indice].lower() == 'negro':
                    acumulador_botellas_negro[0] = acumulador_botellas_negro[0] + int(cantidad[indice])
                elif colores[indice].lower() == 'amarillo':
                    acumulador_botellas_amarillo[0] = acumulador_botellas_amarillo[0] + int(cantidad[indice])
            else: 
                if colores[indice].lower() == 'negro':
                    acumulador_vasos_negros[0] = acumulador_vasos_negros[0] + int(cantidad[indice])
                elif colores[indice].lower() == 'azul':
                    acumulador_vasos_azules[0] = acumulador_vasos_azules[0] + int(cantidad[indice])
    articulo_mas_pedido: list = sorted([acumulador_vasos_azules,acumulador_botellas_amarillo,acumulador_vasos_negros,acumulador_botellas_verdes,acumulador_botellas_azul,
    acumulador_botellas_negro,acumulador_botellas_rojo], key=lambda x: x[0], reverse=True)[0]

    print(f"El articulo mas pedido fue {articulo_mas_pedido[1]} de color {articulo_mas_pedido[2]} con {articulo_mas_pedido[0]} pedidos")
       
def contar_vasos_y_botellas_entregados(lista_pedidos_entregados: list) -> None:
    """Recibe la lista de pedidos entregados para encontrar las botellas y los vasos y sumarlos a los respectivos contadores. Finalmente los imprime."""
    contador_vasos: int = 0
    contador_botellas: int = 0
    for pedido in lista_pedidos_entregados:
        for i in range(0, len(pedido["Cod. Artículo"])): #toma el indice de cada elemento de la lista de "Cod. Artículo" en el dict
            if pedido["Cod. Artículo"][i] == '1334':
                contador_botellas = contador_botellas + int(pedido["Cantidad"][i])
            else:
                contador_vasos = contador_vasos + int(pedido["Cantidad"][i])
    
    print(f"""
    La cantidad de vasos entregados (Cod. Artículo: 568) es: {contador_vasos},
    La cantidad de botellas entregadas (Cod. Artículo: 1334) es: {contador_botellas}.
    """)

# FUNCION GUARDAR ARCHIVO ACTUALIZADO DEL PUNTO 1

def guardar_archivo_actualizado(lista_pedidos: list) -> None: 
    """Recibe la lista de pedidos para copiar el formato csv del archivo de pedidos para actualizar los datos."""
    nombre_archivo = 'pedidos.csv'

    try:

        with open(nombre_archivo, 'w', newline='', encoding="UTF-8") as archivo_csv:
            csv_writer = csv.writer(archivo_csv, delimiter=',', quotechar='"', quoting= csv.QUOTE_NONNUMERIC)
            csv_writer.writerow(["Nro. Pedido", "Fecha", "Cliente", "Ciudad", "Provincia", "Cod. Artículo", "Color" ,"Cantidad" , "Descuento"])

            for pedido in lista_pedidos:
                numero_pedido = pedido["Codigo de pedido"]
                fecha = pedido["Fecha"]
                cliente = pedido["Cliente"]
                ciudad = pedido["Ciudad"]
                provincia = pedido["Provincia"]
                for indice in range(0, len(pedido["Cod. Artículo"])):
                    codigo_articulo = pedido["Cod. Artículo"][indice] #cada indice es un elemento de la lista
                    color = pedido["Color"][indice]
                    cantidad = pedido["Cantidad"][indice]
                    descuento = pedido["Descuento"][indice]
                    csv_writer.writerow((numero_pedido, fecha, cliente, ciudad, provincia, codigo_articulo, color , cantidad , descuento))

        print(f"\n El archivo se guardó correctamente.")

    except IOError as exce:
        print(f"Error al guardar el archivo {nombre_archivo}: {exce.strerror}")

def main() -> None:
    """Es la función principal del programa. Ejecuta varias funciones del archivo y el ABM."""
    
    cinta_transportadora: dict = {
                  'botella verde': 0,
                  'botella rojo': 0,
                  'botella azul': 0,
                  'botella negro': 0,
                  'botella amarillo': 0,
                  'vaso negro': 0,
                  'vaso azul': 0
                  }
		
    reconocimiento_de_objetos(cinta_transportadora)
    print(cinta_transportadora)
    analizar_productos(cinta_transportadora)

    datos: list = leer_archivo()
    lista_pedidos: list = generar_pedidos(datos)
    lista_pedidos = agregar_codigo_pedidos_inicial(lista_pedidos)
    pesos_por_zona, ciudades_por_zona = crear_diccionarios_punto_3(lista_pedidos)
    peso_tirados_por_ciudad: dict = descargar_peso_extra(pesos_por_zona, ciudades_por_zona)
    lista_pedidos_entregados: list = crear_lista_pedidos_entregados(lista_pedidos, peso_tirados_por_ciudad)
    cerrar_programa: bool = False
    menu_general: list = ['ALTA / BAJA / MODIFICACION DE PEDIDOS',
                          'MOSTRAR PEDIDOS ENTREGADOS',
                          'MOSTRAR ARTICULO MAS PEDIDO',
                          'PEDIDOS QUE FUERON A LA CIUDAD DE ROSARIO Y SU COSTO',
                          'CREAR ARCHIVO salida.txt',
                          'MOSTRAR RECORRIDO ÓPTIMO SEGÚN ZONA',
                          'CERRAR PROGRAMA']
    while cerrar_programa == False:
        mostar_menu(menu_general)
        respuesta_usuario = validar_menu(1, len(menu_general))

        if respuesta_usuario == 1:
            menu_abm(lista_pedidos)
            recorrer_caminos_optimos_punto_3(lista_pedidos)
            lista_pedidos_entregados = crear_lista_pedidos_entregados(lista_pedidos, peso_tirados_por_ciudad)

        elif respuesta_usuario == 2:
            mostrar_pedidos_completos(lista_pedidos)
            
        elif respuesta_usuario == 3:
            articulo_mas_pedido(lista_pedidos)
            contar_vasos_y_botellas_entregados(lista_pedidos_entregados)
        
        elif respuesta_usuario == 4:
            listar_pedidos_en_rosario(lista_pedidos_entregados)
        
        elif respuesta_usuario == 5:
            escribir_archivo_salida_txt(pesos_por_zona, ciudades_por_zona) 

        elif respuesta_usuario == 6:
            preguntar_zona_a_recorrer(lista_pedidos)

        else:
            guardar_archivo_actualizado(lista_pedidos)
            print(f"\n EL PROGRAMA SE CERRARA."
                "\n HASTA LUEGO."
                "\n")
            cerrar_programa = True


main()
