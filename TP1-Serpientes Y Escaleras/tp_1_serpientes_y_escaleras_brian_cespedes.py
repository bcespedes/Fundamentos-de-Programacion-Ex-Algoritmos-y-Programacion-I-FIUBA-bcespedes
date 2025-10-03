'''
                                    ████████╗██████╗      ██╗
                                    ╚══██╔══╝██╔══██╗    ███║
                                       ██║   ██████╔╝    ╚██║
                                       ██║   ██╔═══╝      ██║
                                       ██║   ██║          ██║
                                       ╚═╝   ╚═╝          ╚═╝

            ███████╗███████╗██████╗ ██████╗ ██╗███████╗███╗   ██╗████████╗███████╗███████╗
            ██╔════╝██╔════╝██╔══██╗██╔══██╗██║██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔════╝
            ███████╗█████╗  ██████╔╝██████╔╝██║█████╗  ██╔██╗ ██║   ██║   █████╗  ███████╗
            ╚════██║██╔══╝  ██╔══██╗██╔═══╝ ██║██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ╚════██║
            ███████║███████╗██║  ██║██║     ██║███████╗██║ ╚████║   ██║   ███████╗███████║
            ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝

                                            ██╗   ██╗
                                            ╚██╗ ██╔╝
                                             ╚████╔╝
                                              ╚██╔╝
                                               ██║
                                               ╚═╝

                ███████╗███████╗ ██████╗ █████╗ ██╗     ███████╗██████╗  █████╗ ███████╗
                ██╔════╝██╔════╝██╔════╝██╔══██╗██║     ██╔════╝██╔══██╗██╔══██╗██╔════╝
                █████╗  ███████╗██║     ███████║██║     █████╗  ██████╔╝███████║███████╗
                ██╔══╝  ╚════██║██║     ██╔══██║██║     ██╔══╝  ██╔══██╗██╔══██║╚════██║
                ███████╗███████║╚██████╗██║  ██║███████╗███████╗██║  ██║██║  ██║███████║
                ╚══════╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
El módulo os se usó para realizar el limpiado de pantalla.
El módulo time se usó para realizar el efecto de "espera".
El módulo random se usó para obtener números aleatorios.
'''

from os import name, system

from time import sleep

from random import randint


DICCIONARRIO: dict = {3: 18, 6: 67, 57: 83, 72: 89, 85: 96,
                    86: 45, 88: 31, 98: 79, 63: 22, 58: 37, 48: 12, 36: 17}

INSTRUCCIONES_DEL_JUEGO: str = """
                            Bienvenidos a "Serpientes y Escaleras"
    El juego es muy simple. Se mostrarán casilleros del 1 al 100. Se juega por turnos y lanzando un
    dado de 6 caras. El número que obtengas en el dado será la cantidad de casilleros que avanzarás.
                            \x1b[4;37mVerás dos fichas moverse por el tablero:\x1b[;37m
            la ficha \x1b[;34m➊\x1b[;37m corresponde al jugador 1 y la ficha \x1b[;31m➋\x1b[;37m corresponde al jugador 2
    Puedes rendirte durante la partida, pero recuerda que en cuánto haces esto, la misma finalizará.
            \x1b[1;37mEl primer jugador que llegue al casillero 100 ganará la partida.\x1b[;37m
                            \x1b[4;37mCASILLEROS ESPECIALES:\x1b[;37m
    \x1b[1;30;41mS x\x1b[;37m: Serpientes.
    Si pisas justo este casillero tu jugador bajará por la serpiente.
    (Bajarás hasta llegar a la serpiente que tenga el mismo número y color).
    Cantidad total: 7 serpientes.
    \x1b[1;37;41mE x\x1b[;37m: Escaleras.
    Si pisas justo este casillero tu jugador subirá por la escalera.
    (Subirás hasta llegar a la escalera que tenga el mismo número y color).
    Cantidad total: 5 escaleras.
            \x1b[1;37m*** NO SE PUEDE NI BAJAR POR LAS ESCALERAS NI SUBIR POR LAS SERPIENTES ***\x1b[;37m
            \x1b[4;37mLOS SIGUIENTES CASILLEROS ESPECIALES PUEDEN APARECER O NO EN LA PARTIDA:\x1b[;37m
    \x1b[1;33m↓B↓\x1b[;37m: Cáscara de banana.
    Si pisas justo este casillero tu jugador bajará 2 pisos.
    Cantidad máxima: 5 cáscaras de banana por partida.
    \x1b[1;36mR ⮆\x1b[;37m: Rushero.
    Si pisas justo este casillero tu jugador irá hasta el mayor casillero del piso.
    Cantidad máxima: 1 casillero rushero por partida.
    \x1b[1;32m⮄ H\x1b[;37m: Hongos locos.
    Si pisas justo este casillero tu jugador irá hasta el menor casillero del piso.
    Cantidad máxima: 1 casillero hongos locos por partida.
    \x1b[1;35m¿M?\x1b[;37m: ¿¿¿¿¿?????
    ¿¿¿¿¿ คุณไม่ควรอ่านสิ่งนี้. คุณไม่ควรอ่านสิ่งนี้. คุณไม่ควรอ่านสิ่งนี้. คุณไม่ควรอ่านสิ่งนี้. ?????
    Cantidad máxima: 3 por partida."""


def limpiar_pantalla() -> None:
    '''
    Procedimiento utilizado para limpiar la pantalla.
    '''
    if name == 'posix':
        _: int = system('clear')
    else:
        _ = system('cls')


def validar_opcion(numero_min: int, numero_max: int) -> int:
    '''
    Función que permite validar para que solo se puedan ingresar ciertos números enteros.

    PRE: Recibe dos números enteros que simbolizan la cantidad de opciones.
    POST: Devuelve un número entero dentro del rango de opciones.
    '''
    decision: str = input("Ingrese su opción: ")
    while not decision.isnumeric() or int(decision) > numero_max or int(decision) < numero_min:
        print("La opción ingresada, no es valida.")
        decision = input("Intente nuevamente, ingrese su opción: ")

    return int(decision)


def validar_nombre(numero_jugador: int) -> str:
    '''
    Función que permite validar los nombres de los jugadores.

    PRE: El nombre del jugador no puede contener espacios ni números.
    POST: Devuelve el nombre ingresado.
    '''
    nombre: str = input(f"Ingrese el nombre del jugador {numero_jugador}: ")
    while not nombre.isalpha() or nombre.isspace():
        print("Lo que usted ha ingresado no es válido.")
        nombre = input(f"Intente nuevamente, el nombre del jugador {numero_jugador} es: ")

    return nombre.capitalize()


def ingresar_nombres(nombre_jugador: list) -> None:
    '''
    Procedimiento que recibe los nombres validados en la función anterior
    y los almacena en una lista.

    PRE: Recibe una lista en donde se almacenarán los nombres de los jugadores.
    POST: Se valida el nombre con la función anterior y se almacena en la lista.
    '''
    for i in range(1, 3):
        nombre: str = validar_nombre(i)
        nombre_jugador.append(nombre)


def dar_bienvenida_usuarios(lista_nombres: list) -> None:
    '''
    Procedimiento que da la bienvenida a los jugadores y explica la modalidad de juego.

    PRE: Recibe los nombres de los jugadores usando la lista anterior.
    POST: Imprime una bienvenida y explica la jugabilidad.
    '''
    ingresar_nombres(lista_nombres)
    print(INSTRUCCIONES_DEL_JUEGO)


def elegir_primer_turno(nombre_jugador: list) -> list:
    '''
    Función que decide de manera aleatoria quién será el primero en jugar.

    PRE: Recibe los nombres de los jugadores usando la lista anterior.
    POST: Muestra por pantalla el nombre del jugador que tendrá el primer turno.
    Devuelve una lista en la que está almacenada el dato del turno.
    '''
    print("Decidiendo quién será el primero en jugar, por favor espere...")
    sleep(3)
    turno_jugador: list = [-10, 0, 0]
    primer_turno: int = randint(1, 2)
    if primer_turno == 1:
        turno_jugador[1] = 1
        print(f"El jugador {nombre_jugador[1]} (Jugador 1) será el primero en jugar.")
    elif primer_turno == 2:
        turno_jugador[2] = 1
        print(f"El jugador {nombre_jugador[2]} (Jugador 2) será el primero en jugar.")

    return turno_jugador


def obtener_numero_de_casillero(piso_jugador: int, columna_jugador: int) -> int:
    '''
    Función que sirve para obtener el numero de casillero actual.

    PRE: Recibe 2 números enteros, correspondientes a coordenadas dentro del tablero.
    POST: Devuelve un entero correspondiente al número de casillero dentro del tablero.
    '''
    numero: int = (10 - piso_jugador) * 10
    if numero % 20 != 0:
        columna_jugador = 9 - columna_jugador
    posicion_actual: int = numero - columna_jugador

    return posicion_actual


def obtener_coordenadas(posicion_actual: int) -> tuple:
    '''
    Función que sirve para obtener la coordenada del número de casillero que le pasemos.

    PRE: Recibe el número de casillero dentro del tablero.
    POST: Devuelve una tupla con las coordenadas correspondientes a ese casillero.
    '''
    if posicion_actual % 10 == 0:
        piso_jugador: int = 10
        columna_nueva: int = 9
    elif posicion_actual % 10 != 0:
        piso_jugador = 9
        columna_nueva = -1
    while posicion_actual % 10 != 0:
        posicion_actual -= 1
        columna_nueva += 1
    while posicion_actual % 10 == 0 and posicion_actual != 0:
        piso_jugador -= 1
        posicion_actual -= 10
    if piso_jugador % 2 == 0:
        columna_jugador: int = 9 - columna_nueva
    elif piso_jugador % 2 != 0:
        columna_jugador = columna_nueva
    coordenadas: tuple = piso_jugador, columna_jugador

    return coordenadas


def crear_y_almacenar_casilleros_normales() -> list:
    '''
    Función que crea los valores de los casilleros normales en el tablero
    y los almacena en una lista.

    PRE: No recibe nada (el tamaño del tablero es fijo).
    POST: Devuelve los valores de casilleros, agregando ceros si es necesario.
    '''
    casilleros: list = []
    for i in range(100, 0, -1):
        if len(str(i)) == 1:
            agregar_ceros: str = "00" + str(i)
        elif len(str(i)) == 2:
            agregar_ceros = "0" + str(i)
        else:
            agregar_ceros = str(i)
        casilleros.append(agregar_ceros)

    return casilleros


def crear_y_almacenar_casilleros_especiales(casilleros_especiales: dict) -> None:
    '''
    Procedimiento que crea y almacena los valores de los casilleros especiales
    en una clave de un diccionario y luego en otra para que no se repitan.

    PRE: Recibe el diccionario de casilleros especiales.
    POST: Crea (o busca en el diccionario constante en el caso de las serpientes
    y escaleras) y almacena los valores correspondientes a casilleros especiales
    en una clave específica y luego en otra clave para que no se repitan los mismos
    valores.
    '''
    for clave, valor in DICCIONARRIO.items():
        if clave < valor:
            casilleros_especiales["X"].append(clave)
            casilleros_especiales["X"].append(valor)
    for clave, valor in DICCIONARRIO.items():
        if clave > valor:
            casilleros_especiales["X"].append(clave)
            casilleros_especiales["X"].append(valor)
    cantidad_b: int = randint(0, 5)
    for _ in range(cantidad_b):
        posicion_b: int = randint(21, 99)
        while posicion_b in casilleros_especiales["X"]:
            posicion_b = randint(21, 99)
        casilleros_especiales["B"].append(posicion_b)
        casilleros_especiales["X"].append(posicion_b)
    cantidad_m: int = randint(0, 3)
    for _ in range(cantidad_m):
        posicion_m: int = randint(1, 99)
        while posicion_m in casilleros_especiales["X"]:
            posicion_m = randint(1, 99)
        casilleros_especiales["M"].append(posicion_m)
        casilleros_especiales["X"].append(posicion_m)
    cantidad_r: int = randint(0, 1)
    for _ in range(cantidad_r):
        posicion_r: int = randint(1, 99)
        while posicion_r in casilleros_especiales["X"] or posicion_r % 10 == 0:
            posicion_r = randint(2, 99)
        casilleros_especiales["R"].append(posicion_r)
        casilleros_especiales["X"].append(posicion_r)
    cantidad_h: int = randint(0, 1)
    for _ in range(cantidad_h):
        posicion_h: int = randint(1, 99)
        while posicion_h in casilleros_especiales["X"] or posicion_h % 10 == 1:
            posicion_h = randint(2, 99)
        casilleros_especiales["H"].append(posicion_h)
        casilleros_especiales["X"].append(posicion_h)


def recorrer_tablero(i: int, j: int) -> int:
    ''''
    Función que permite recorrer el tablero con el efecto de "zigzag".

    PRE: Recibe 2 enteros que representan 2 coordenadas dentro del tablero.
    POST: Devuelve la posición de columna desde donde se debe recorrer el tablero
    en la fila actual.
    '''
    variable_auxiliar: int = j
    if i % 2 != 0:
        j = 9
        j -= variable_auxiliar

    return j


def colocar_casilleros_normales(tablero: list) -> None:
    '''
    Procedimiento que coloca los casilleros normales en un tablero.

    PRE: Se recibe un tablero.
    POST: Se colocan los casilleros normales en este tablero y crea
    un efecto de "zigzag".
    '''
    casilleros: list = crear_y_almacenar_casilleros_normales()
    contador: int = 0
    for i in range(10):
        tablero.append([])
        for _ in range(10):
            tablero[i].append(casilleros[contador])
            contador += 1
            if contador % 20 == 0 and contador != 0:
                tablero[i] = tablero[i][::-1]


def colocar_casilleros_especiales(tablero: list, casilleros_especiales: dict) -> None:
    '''
    Procedimiento que usa los valores de casilleros almacenados en el diccionario previo
    para colocar los mismos en un tablero cualquiera.

    PRE: Recibe el tablero y el diccionario en los que están almacenados los valores de
    casilleros especiales.
    POST: Se verifica si en el casillero actual hay algún valor correspondiente a un
    casillero especial. De ser así se reemplaza el casillero actual con el valor que
    corresponda.
    '''
    contador_e: int = 0
    contador_s: int = 0
    for i in range(len(tablero), -1, -1):
        for j in range(len(tablero), -1, -1):
            j = recorrer_tablero(i, j)
            casillero: int = obtener_numero_de_casillero(i, j)
            if casillero in casilleros_especiales["B"]:
                coordenada: tuple = obtener_coordenadas(casillero)
                tablero[coordenada[0]][coordenada[1]] = "↓B↓"
            elif casillero in casilleros_especiales["M"]:
                coordenada = obtener_coordenadas(casillero)
                tablero[coordenada[0]][coordenada[1]] = "¿M?"
            elif casillero in casilleros_especiales["R"]:
                coordenada = obtener_coordenadas(casillero)
                if coordenada[0] % 2 == 0:
                    tablero[coordenada[0]][coordenada[1]] = "⮄ R"
                else:
                    tablero[coordenada[0]][coordenada[1]] = "R ⮆"
            elif casillero in casilleros_especiales["H"]:
                coordenada = obtener_coordenadas(casillero)
                if coordenada[0] % 2 == 0:
                    tablero[coordenada[0]][coordenada[1]] = "H ⮆"
                else:
                    tablero[coordenada[0]][coordenada[1]] = "⮄ H"
            elif casillero in DICCIONARRIO.keys() and casillero < DICCIONARRIO[casillero]:
                contador_e += 1
                coordenada = obtener_coordenadas(DICCIONARRIO[casillero])
                tablero[coordenada[0]][coordenada[1]] =\
                    f"\x1b[1;37;4{str(contador_e)}mE {contador_e}"
                coordenada = obtener_coordenadas(casillero)
                tablero[coordenada[0]][coordenada[1]] =\
                    f"\x1b[1;37;4{str(contador_e)}mE {contador_e}"
            elif casillero in DICCIONARRIO.keys() and casillero > DICCIONARRIO[casillero]:
                contador_s += 1
                coordenada = obtener_coordenadas(casillero)
                tablero[coordenada[0]][coordenada[1]] =\
                    f"\x1b[1;30;4{str(contador_s)}mS {contador_s}"
                coordenada = obtener_coordenadas(DICCIONARRIO[casillero])
                tablero[coordenada[0]][coordenada[1]] =\
                    f"\x1b[1;30;4{str(contador_s)}mS {contador_s}"
    # En el caso de las escaleras y serpientes se les aplica color en esta instancia para
    # que cada par de las mismas tenga su color específico


def crear_tablero_juego(casilleros_especiales: dict) -> list:
    '''
    Función que crea el tablero del juego colocando los casilleros normales
    y especiales creados previamente.

    PRE: Recibe los valores de los casilleros especiales.
    POST: Coloca los casilleros normales en el tablero, si es necesario los reemplaza
    por casilleros especiales. Devuelve el tablero.
    '''
    tablero: list = []
    colocar_casilleros_normales(tablero)
    colocar_casilleros_especiales(tablero, casilleros_especiales)

    return tablero


def asignar_colores(tablero: list, i: int, j: int) -> None:
    '''
    Procedimiento que asigna colores específicos a la ficha de la posición actual
    de los jugadores y a los casilleros especiales en el tablero.

    PRE: Recibe un tablero, y 2 números correspondientes a coordenadas dentro del mismo.
    POST: En caso de encontrar un casillero especial o la ficha del jugador, se le asigna
    el color correspondiente. En caso de no encontrar ningún casillero especial ni ficha
    de jugador se asigna el color por defecto (blanco).
    '''
    if " ➊ " in tablero[i - 1][j]:
        print("│","\x1b[;34m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    elif " ➋ " in tablero[i - 1][j]:
        print("│","\x1b[;31m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    elif tablero[i - 1][j] == "➊ ➋":
        print("│","\x1b[;34m➊ \x1b[;31m➋", end = "\x1b[;37m | ")
    elif tablero[i - 1][j] == "➋ ➊":
        print("│","\x1b[;31m➋ \x1b[;34m➊", end = "\x1b[;37m | ")
    elif "↓B↓" in tablero[i - 1][j]:
        print("│","\x1b[1;33m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    elif "¿M?" in tablero[i - 1][j]:
        print("│","\x1b[1;35m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    elif "⮄ R" in tablero[i - 1][j]:
        print("│","\x1b[1;36m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    elif "R ⮆" in tablero[i - 1][j]:
        print("│","\x1b[1;36m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    elif "⮄ H" in tablero[i - 1][j]:
        print("│","\x1b[1;32m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    elif "H ⮆" in tablero[i - 1][j]:
        print("│","\x1b[1;32m"+tablero[i - 1][j], end = "\x1b[;37m | ")
    else:
        print("│",tablero[int(i - 1)][j], end = "\x1b[;37m | ")


def mostrar_tablero(tablero: list) -> None:
    '''
    Procedimiento que muestra en pantalla el tablero que le indiquemos.

    PRE: Recibe un tablero cualquiera.
    POST: Se imprime en pantalla el tablero, aplicando detalles estéticos.
    '''
    # Esto es puramente estético, se agregan los valores correspondientes a las columnas.
    for columnas in range(1, 11):
        if int(columnas) == 1:
            print("■■ » ",end = " ")
        if len(str(columnas)) == 1:
            agregar_espacios: str = " "+str(columnas)
            print("",agregar_espacios,end = "     ")
        else:
            print(" ",str(columnas),end = "    ")
    print("\n")
    # Esto también es estético, se agregan los valores correspondientes a las filas.
    for i in range(1, 11):
        if len(str(i)) == 1:
            agregar_espacios = " "+str(i)
            print(agregar_espacios,end=" » ")
        else:
            print(str(i),end=" » ")
        for j in range(10):
            asignar_colores(tablero, i, j)
        print("\n")


def pisar_escalera(posicion_actual: int, piso_jugador: list,
    columna_jugador: list, indice: int) -> None:
    '''
    Procedimiento que ocurre cuando el jugador se para sobre un casillero
    del tipo escalera.

    PRE: Recibe el piso y la columna del jugador (el índice es para saber a qué jugador corresponde)
    (la posición actual es para mostrar por pantalla en qué casillero se encontraba el jugador).
    POST: Realiza la acción correspondiente a casillero escalera.
    '''
    print(f"Te has parado en el casillero {posicion_actual} y allí había una escalera.")
    print("* Subes por la escalera *")
    posicion_actual = DICCIONARRIO[posicion_actual]
    coords: tuple = obtener_coordenadas(posicion_actual)
    piso_jugador[indice] = coords[0]
    columna_jugador[indice] = coords[1]


def pisar_serpiente(posicion_actual: int, piso_jugador: list,
    columna_jugador: list, indice: int) -> None:
    '''
    Procedimiento que ocurre cuando el jugador se para sobre un casillero
    del tipo serpiente.

    PRE: Recibe el piso y la columna del jugador (el índice es para saber a qué jugador corresponde)
    (la posición actual es para mostrar por pantalla en qué casillero se encontraba el jugador).
    POST: Realiza la acción correspondiente a casillero serpiente.
    '''
    print(f"Te has parado en el casillero {posicion_actual} y allí había una serpiente.")
    print("* Bajas por la serpiente *")
    posicion_actual = DICCIONARRIO[posicion_actual]
    coords: tuple = obtener_coordenadas(posicion_actual)
    piso_jugador[indice] = coords[0]
    columna_jugador[indice] = coords[1]


def pisar_cascara_de_banana(posicion_actual: int, piso_jugador: list,
    columna_jugador: list, indice: int) -> None:
    '''
    Procedimiento que ocurre cuando el jugador se para sobre un casillero
    del tipo cáscara de banana.

    PRE: Recibe el piso y la columna del jugador (el índice es para saber a qué jugador corresponde)
    (la posición actual es para mostrar por pantalla en qué casillero se encontraba el jugador).
    POST: Realiza la acción correspondiente a cáscara de banana.
    '''
    print(f"Te has parado en el casillero {posicion_actual} y allí había una cáscara de banana.")
    print("* Te resbalas y caes dos pisos* ")
    coords: tuple = obtener_coordenadas(posicion_actual)
    piso_jugador[indice] = coords[0] + 2
    columna_jugador[indice] = coords[1]


def pisar_portal_magico(posicion_actual: int, piso_jugador: list,
    columna_jugador: list, indice: int) -> None:
    '''
    Procedimiento que ocurre cuando el jugador se para sobre un casillero
    del tipo portal mágico.

    PRE: Recibe el piso y la columna del jugador (el índice es para saber a qué jugador corresponde)
    (la posición actual es para mostrar por pantalla en qué casillero se encontraba el jugador).
    POST: Realiza la acción correspondiente a portal mágico.
    '''
    print(f"Te has parado en el casillero {posicion_actual} y allí había un portal mágico.")
    print("* Te teletransportas a una ubicación aleatoria *")
    piso_jugador[indice] = randint(0, 9)
    columna_jugador[indice] = randint(0, 9)
    if piso_jugador[indice] == 0 or piso_jugador[indice] == 9:
        while columna_jugador[indice] == 0:
            columna_jugador[indice] = randint(1, 9)


def pisar_rushero(posicion_actual: int, piso_jugador: list,
    columna_jugador: list, indice: int) -> None:
    '''
    Procedimiento que ocurre cuando el jugador se para sobre un casillero
    del tipo rushero.

    PRE: Recibe el piso y la columna del jugador (el índice es para saber a qué jugador corresponde)
    (la posición actual es para mostrar por pantalla en qué casillero se encontraba el jugador).
    POST: Realiza la acción correspondiente a casillero rushero.
    '''
    print(f"Te has parado en el casillero {posicion_actual} y allí había un casillero rushero.")
    print("* Corres hacia el final del piso actual *")
    if piso_jugador[indice] % 2 == 0:
        while columna_jugador[indice] != 0:
            columna_jugador[indice] -= 1
    elif piso_jugador[indice] % 2 != 0:
        while columna_jugador[indice] != 9:
            columna_jugador[indice] += 1


def pisar_hongos_locos(posicion_actual: int, piso_jugador: list,
    columna_jugador: list, indice: int) -> None:
    '''
    Procedimiento que ocurre cuando el jugador se para sobre un casillero
    del tipo hongos locos.

    PRE: Recibe el piso y la columna del jugador (el índice es para saber a qué jugador corresponde)
    (la posición actual es para mostrar por pantalla en qué casillero se encontraba el jugador).
    POST: Realiza la acción correspondiente a hongos locos.
    '''
    print(f"Te has parado en el casillero {posicion_actual} y allí había hongos locos.")
    print("* Te comes los hongos y corres hacia el inicio del piso actual *")
    if piso_jugador[indice] % 2 == 0:
        while columna_jugador[indice] != 9:
            columna_jugador[indice] += 1
    elif piso_jugador[indice] % 2 != 0:
        while columna_jugador[indice] != 0:
            columna_jugador[indice] -= 1


def verificar_casillero_actual(piso_jugador: list, columna_jugador: list,
    casilleros_especiales: dict, contador_casilleros_especiales: dict, indice: int) -> None:
    '''
    Procedimiento que verifica si en el casillero actual no se encuentra ningún casillero especial.
    En caso de haberlo, se realiza la acción correspondiente a ese casillero.

    PRE: Recibe el piso y la columna del jugador (el indice es para saber a qué jugador corresponde)
    recibe también los valores de los casilleros especiales y un contador para los mismos.
    POST: Verifica si en el casillero actual del jugador hay algún casillero especial y realiza la
    acción correspondiente. En caso de haberlo, también se suma 1 al contador del casillero especial
    pisado.
    '''
    es_casillero_normal: bool = False
    interacciones_casilleros_especiales: int = 0
    while not es_casillero_normal and interacciones_casilleros_especiales != 10:
        posicion_actual: int = obtener_numero_de_casillero(piso_jugador[indice],\
            columna_jugador[indice])
        if posicion_actual in DICCIONARRIO and posicion_actual < DICCIONARRIO[posicion_actual]:
            pisar_escalera(posicion_actual, piso_jugador, columna_jugador, indice)
            contador_casilleros_especiales["Escalera"] += 1
        elif posicion_actual in DICCIONARRIO and posicion_actual > DICCIONARRIO[posicion_actual]:
            pisar_serpiente(posicion_actual, piso_jugador, columna_jugador, indice)
            contador_casilleros_especiales["Serpiente"] += 1
        elif posicion_actual in casilleros_especiales["B"]:
            pisar_cascara_de_banana(posicion_actual, piso_jugador, columna_jugador, indice)
            contador_casilleros_especiales["Cascara de Banana"] += 1
        elif posicion_actual in casilleros_especiales["M"]:
            pisar_portal_magico(posicion_actual, piso_jugador, columna_jugador, indice)
            contador_casilleros_especiales["Portal Magico"] += 1
        elif posicion_actual in casilleros_especiales["R"]:
            pisar_rushero(posicion_actual, piso_jugador, columna_jugador, indice)
            contador_casilleros_especiales["Rushero"] += 1
        elif posicion_actual in casilleros_especiales["H"]:
            pisar_hongos_locos(posicion_actual, piso_jugador, columna_jugador, indice)
            contador_casilleros_especiales["Hongos Locos"] += 1
        else:
            es_casillero_normal = True
        interacciones_casilleros_especiales += 1
        # Este contador se hizo para evitar un loop infinito entre un casillero rushero al inicio
        # del piso actual y un casillero hongos locos al final del piso actual.
        if interacciones_casilleros_especiales == 10:
            print("""
            Ya has tenido suficientes interacciones de casilleros especiales.
            El dios de serpientes y escaleras tendrá piedad contigo...
            """)


def avanzar_casilleros(piso_jugador: list, columna_jugador: list, indice: int) -> None:
    '''
    Procedimiento utilizado para que los jugadores avancen por el tablero.

    PRE: Recibe el piso y la columna actual del jugador (el indice es para saber a
    qué jugador corresponde).
    POST: Se va modificando la posición del jugador, respetando la forma de zigzag
    que tiene el tablero.
    '''
    if piso_jugador[indice] % 2 == 0:
        columna_jugador[indice] -= 1
    elif piso_jugador[indice] % 2 != 0:
        columna_jugador[indice] += 1
    if columna_jugador[indice] > 9:
        piso_jugador[indice] -= 1
        columna_jugador[indice] = 9
    elif columna_jugador[indice] < 0:
        piso_jugador[indice] -= 1
        columna_jugador[indice] = 0
    if columna_jugador[indice] <= 0 and piso_jugador[indice] <= 0:
        columna_jugador[indice] = 0
        piso_jugador[indice] = 0


def lanzar_el_dado(indice: int) -> int:
    '''
    Función que se encarga de lanzar el dado de 6 caras.

    PRE: Recibe el índice (para saber quién es el jugador del turno actual).
    POST: Devuelve el resultado aleatorio obtenido por el dado.
    '''
    dado: int = randint(1, 6)
    print("Lanzando el dado...")
    sleep(0.5)
    limpiar_pantalla()
    print(f"El dado arroja el número: {dado}.")
    print(f"El jugador {indice} avanza esa cantidad de casilleros.")

    return dado


def actualizar_y_mostrar_tablero(tablero: list, turno_jugador: list,
    piso_jugador: list, columna_jugador: list) -> None:
    '''
    Procedimiento que se encarga de ubicar la ficha de cada jugador
    en su posición actual.

    PRE: Recibe un tablero, el turno del jugador, su piso y columna actual.
    POST: Dependiendo el turno del jugador, ubica su ficha en el tablero,
    en su posición actual y luego imprime el tablero.
    '''
    if turno_jugador[1] == 1:
        tablero[piso_jugador[1]][columna_jugador[1]] = " ➊ "
        if columna_jugador[1] == columna_jugador[2] and piso_jugador[1] == piso_jugador[2]:
            tablero[piso_jugador[2]][columna_jugador[2]] = "➊ ➋"
        elif columna_jugador[2] >= 0:
            tablero[piso_jugador[2]][columna_jugador[2]] = " ➋ "
        mostrar_tablero(tablero)
    elif turno_jugador[2] == 1:
        tablero[piso_jugador[2]][columna_jugador[2]] = " ➋ "
        if columna_jugador[2] == columna_jugador[1] and piso_jugador[2] == piso_jugador[1]:
            tablero[piso_jugador[2]][columna_jugador[2]] = "➋ ➊"
        elif columna_jugador[1] >= 0:
            tablero[piso_jugador[1]][columna_jugador[1]] = " ➊ "
        mostrar_tablero(tablero)


def verificar_turno_actual(turno_jugador: list) -> tuple:
    '''
    Función que verifica de quién es el turno actual.

    PRE: Recibe el turno actual del jugador.
    POST: Devuelve los índices del jugador actual y el índice contrario.
    '''
    if turno_jugador[1] == 1:
        indice: int = 1
        indice_contrario: int = 2
    elif turno_jugador[2] == 1:
        indice = 2
        indice_contrario = 1
    indices: tuple = indice, indice_contrario

    return indices


def rendirse(turno_jugador: list, piso_jugador: list, columna_jugador: list,
    casilleros_especiales: dict) -> int:
    '''
    Función que se ejecuta cuando alguno de los jugadores decide rendirse.

    PRE: Recibe el turno, el piso y columna de los jugadores y los casilleros especiales.
    POST: Cambia la posición del jugador adversario al final del tablero, dándole
    la victoria. El jugador rendido sin embargo cambia su posición al inicio. Devuelve
    un entero, indicando que la partida finalizó por rendición.
    '''
    rendicion: int = 1
    indice, indice_contrario = verificar_turno_actual(turno_jugador)
    piso_jugador[indice_contrario] = 0
    columna_jugador[indice_contrario] = 0
    piso_jugador[indice] = 9
    columna_jugador[indice] = 0
    tablero: list = crear_tablero_juego(casilleros_especiales)
    limpiar_pantalla()
    actualizar_y_mostrar_tablero(tablero, turno_jugador, piso_jugador, columna_jugador)

    return rendicion


def menu_turnos(piso_jugador: list, columna_jugador: list, turno_jugador: list,
    casilleros_especiales: dict, contador_casilleros_especiales: dict) -> int:
    '''
    Función encargada de ejecutar el menú correspondiente a los turnos.

    PRE: Recibe el piso y columna de los jugadores, los turnos, los casilleros especiales
    y el contador de casilleros especiales.
    POST: Ejecuta la acción en base a lo que decida el jugador. Puede decidir jugar su turno
    o rendirse (el juego termina automáticamente). Devuelve un entero, indicando si se ha producido
    una rendición o no.
    '''
    rendicion: int = 0
    indices: tuple = verificar_turno_actual(turno_jugador)
    print(f"\n\t\tEs el turno del jugador {indices[0]}")
    opcion: str = input("Ingrese cualquier tecla para lanzar el dado o 'x' para rendirse: ")
    if opcion.capitalize() != 'X':
        jugar_turnos(piso_jugador, columna_jugador, turno_jugador,
        casilleros_especiales, contador_casilleros_especiales)
    else:
        rendicion = rendirse(turno_jugador, piso_jugador, columna_jugador, casilleros_especiales)

    return rendicion


def jugar_turnos(piso_jugador: list, columna_jugador: list, turno_jugador: list,
    casilleros_especiales: dict, contador_casilleros_especiales: dict) -> None:
    '''
    Procedimiento que permite hacer el intercambio de turnos entre jugadores.

    PRE: Recibe el piso y columna de los jugadores, los turnos, los casilleros especiales
    y el contador de casilleros especiales.
    POST: Se ejecuta el turno del jugador, realizando las acciones correspondientes (avanzar
    por el casillero, verificar que no hayan casilleros especiales, etc) y luego se intercambia
    el turno con el otro jugador (siempre y cuando no se haya llegado al casillero 100).
    '''
    indice, indice_contrario = verificar_turno_actual(turno_jugador)
    dado: int = lanzar_el_dado(indice)
    for _ in range(dado):
        avanzar_casilleros(piso_jugador, columna_jugador, indice)
    verificar_casillero_actual(piso_jugador, columna_jugador,\
    casilleros_especiales, contador_casilleros_especiales, indice)
    tablero: list = crear_tablero_juego(casilleros_especiales)
    actualizar_y_mostrar_tablero(tablero, turno_jugador, piso_jugador, columna_jugador)
    posicion_actual: int = obtener_numero_de_casillero(piso_jugador[indice],\
    columna_jugador[indice])
    if posicion_actual < 100:
        print(f"Ahora el jugador {indice} está en el casillero numero {posicion_actual}")
        turno_jugador[indice] = 0
        turno_jugador[indice_contrario] = 1
    elif posicion_actual == 100:
        print(f"¡Felicitaciones jugador {indice}! Has llegado a la meta.")


def mostrar_ganador(piso_jugador: list, columna_jugador: list, rendicion: int) -> None:
    '''
    Procedimiento que muestra quién ganó la partida.

    PRE: Recibe la columna y el piso en el que quedaron ambos jugadores. Además recibe un
    entero para verificar si algún jugador decidió rendirse.
    POST: Imprime un mensaje correspondiente a las distintas situaciones.
    '''
    if rendicion == 0:
        if piso_jugador[1] == 0 and columna_jugador[1] == 0 and piso_jugador[2] >= 5:
            print("Ha ganado el jugador 1.")
            print("El jugador 2 ha estado bastante lejos de la victoria.\n\n")
        elif piso_jugador[2] == 0 and columna_jugador[2] == 0 and piso_jugador[1] >= 5:
            print("Ha ganado el jugador 2.")
            print("El jugador 1 ha estado bastante lejos de la victoria.\n\n")
        elif piso_jugador[1] == 0 and columna_jugador[1] == 0 and piso_jugador[2] <= 5:
            print("Ha ganado el jugador 1.")
            print("¡Ha sido una excelente partida!\n\n")
        elif piso_jugador[2] == 0 and columna_jugador[2] == 0 and piso_jugador[1] <= 5:
            print("Ha ganado el jugador 2.")
            print("¡Ha sido una excelente partida!\n\n")
    elif rendicion == 1:
        if piso_jugador[1] == 0 and columna_jugador[1] == 0:
            print("Ha ganado el jugador 1.")
            print("El jugador 2 se ha rendido.\n\n")
        elif piso_jugador[2] == 0 and columna_jugador[2] == 0:
            print("Ha ganado el jugador 2.")
            print("El jugador 1 se ha rendido.\n\n")


def crear_base_del_tablero() -> tuple:
    '''
    Función que crea las bases del tablero para usarlas en la función jugabilidad().

    PRE: No recibe nada.
    POST: Devuelve una tupla que contiene los datos iniciales del tablero.
    '''
    piso_jugador: list = [-10, 9, 9]
    columna_jugador: list = [-10, -1, -1]
    casilleros_especiales: dict = {"B": [], "M": [], "R": [], "H": [], "X": []}
    crear_y_almacenar_casilleros_especiales(casilleros_especiales)
    datos_iniciales_tablero: tuple = piso_jugador, columna_jugador, casilleros_especiales

    return datos_iniciales_tablero


def menu_jugabilidad(nombre_jugador: list, contador_casilleros_especiales: dict) -> None:
    '''
    Procedimiento que ejecuta el menú correspondiente a la función jugabilidad.
    Permite iniciar una nueva partida o volver al menú inicial.

    PRE: Recibe los nombres de los jugadores y el contador de casilleros especiales.
    POST: Realiza la opción deseada por el usuario.
    '''
    cerrar_menu: bool = False
    while not cerrar_menu:
        print(f"""
        Bienvenido al menú del juego Serpientes y Escaleras.
        Actualmente están cargados los siguientes jugadores:
        Jugador 1: {nombre_jugador[1]}
        Jugador 2: {nombre_jugador[2]}
        Si desea cambiar estos nombres, por favor vuelva al menú inicial.
        Elija una opción:
        1. Empezar una partida.
        2. Ir al menú inicial.
        """)
        eleccion: int = validar_opcion(1, 2)
        if eleccion == 2:
            cerrar_menu = True
        else:
            jugabilidad(nombre_jugador, contador_casilleros_especiales)


def jugabilidad(nombre_jugador: list, contador_casilleros_especiales: dict) -> None:
    '''
    Procedimiento que permite la jugabilidad de serpientes y escaleras.

    PRE: Recibe la lista de nombres de jugadores y el contador de casilleros especiales.
    POST: Se crean las bases del tablero, y mientras nadie alcance el casillero 100
    se juegan los turnos.
    '''
    piso_jugador, columna_jugador, casilleros_especiales = crear_base_del_tablero()
    turno_jugador: list = elegir_primer_turno(nombre_jugador)
    while (columna_jugador[1] != 0 or piso_jugador[1] != 0) and\
        (columna_jugador[2] != 0 or piso_jugador[2] != 0):
        rendicion: int = menu_turnos(piso_jugador, columna_jugador, turno_jugador,\
            casilleros_especiales, contador_casilleros_especiales)
    mostrar_ganador(piso_jugador, columna_jugador, rendicion)


def mostrar_estadisticas_de_casilleros(contador_casilleros_especiales: dict) -> None:
    '''
    Procedimiento que muestra cuantas veces se han activado ciertos casilleros.

    PRE: Recibe el diccionario que cuenta las veces que se han pisado los casilleros
    especiales.
    POST: Imprime por pantalla cuantas veces se han activado los casilleros.
    En caso de no haberse activado ninguno, se le informa al jugador.
    '''
    if contador_casilleros_especiales == {"Escalera": 0, "Serpiente": 0,\
        "Cascara de Banana": 0, "Portal Magico": 0, "Rushero": 0, "Hongos Locos": 0}:
        print("Aún no se ha empezado ninguna partida o no se han pisado casilleros especiales.")
    else:
        for clave, valor in contador_casilleros_especiales.items():
            if valor == 1:
                print(f"El casillero especial {clave} se ha activado {valor} vez.")
            else:
                print(f"El casillero especial {clave} se ha activado {valor} veces.")


def reiniciar_estadisticas_de_casilleros(contador_casilleros_especiales: dict) -> None:
    '''
    Procedimiento que permite reiniciar las estadísticas de casilleros especiales.

    PRE: Recibe el contador de casilleros especiales.
    POST: Establece todos los valores del contador en 0.
    '''
    for clave in contador_casilleros_especiales:
        contador_casilleros_especiales[clave] = 0
    print("""
    Las estadísticas de casilleros especiales han sido reiniciadas exitosamente.""")


def realizar_accion(opcion: int, contador_casilleros_especiales: dict) -> None:
    '''
    Procedimiento que permite realizar cierta opción del menú principal a pedido del usuario.

    PRE: Recibe la opción ingresada y el diccionario de casilleros especiales.
    POST: Se realiza la acción pedida por el usuario.
    '''
    lista_nombres: list = [0]
    if opcion == 1:
        dar_bienvenida_usuarios(lista_nombres)
        menu_jugabilidad(lista_nombres, contador_casilleros_especiales)
    elif opcion == 2:
        mostrar_estadisticas_de_casilleros(contador_casilleros_especiales)
    elif opcion == 3:
        reiniciar_estadisticas_de_casilleros(contador_casilleros_especiales)


def main() -> None:
    '''
    Procedimiento principal del juego serpientes y escaleras.
    '''
    cerrar_menu: bool = False
    contador_casilleros_especiales: dict = {"Escalera": 0, "Serpiente": 0\
    , "Cascara de Banana": 0, "Portal Magico": 0, "Rushero": 0, "Hongos Locos": 0}
    while not cerrar_menu:
        print("""
        Bienvenido al menú inicial de Serpientes y Escaleras.
        Por favor, elija una opción.
        1. Iniciar partida.
        2. Mostrar estadísticas de casilleros.
        3. Reiniciar estadísticas de casilleros.
        4. Salir.
        """)
        opcion: int = validar_opcion(1, 4)
        if opcion == 4:
            print("""
            Gracias por jugar a Serpientes y Escaleras.
            Esperamos verte pronto de vuelta <3
            """)
            cerrar_menu = True
        else:
            realizar_accion(opcion, contador_casilleros_especiales)
main()
