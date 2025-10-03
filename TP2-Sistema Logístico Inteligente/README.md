# Trabajo Práctico Integrador de Algoritmos y Programación 1
## Cátedra Costa

## Integrantes 👩‍💻 👨‍💻
- Estefanía Santana
- Camila Perez
- Sebastián Ledesma
- Brian Céspedes

## Funcionamiento del TP y requerimientos 📝
- El tp consta de varios archivos que no se encuentran en el repositorio debido a su peso (salvo el de `pedidos.csv`). A continuación se deja link de Drive que redirige a los archivos faltantes: https://drive.google.com/drive/folders/1WBb_Z5_vjlQrwfPJiukMXkgh6JeGjfYN?usp=sharing
- Se utilizaron las siguientes librerías:
  - Opencv
  - Numpy
  - CSV
  - Datetime
  - Copy
  - Geopy
  - Time
 
Para llevar a cabo la correcta ejecución del programa, se requiere de la instalación de estas librerías.

### Temática del TP
Este trabajo práctico consta de un **ABM** (Alta-Baja-Modificación) de pedidos que llevamos a cabo para la empresa 'Logistik', un importador y distribuidor de Latinoamérica, la cual va a almacenar los productos provenientes de CABA para luego clasificar los pedidos, valorizarlos y distribuirlos por todo el país. 
La empresa comercializa dos productos estrellas: **vasos** y **botellas Chillys** de distintos colores.

Logistik cuenta con un proceso que consta de una cinta transportadora que lleva estos productos que se reciben a granel para pasar por un sensor (cámara fotográfica) que 
determina el tipo de producto y color, luego un brazo robot termina acomodando el objeto en la caja correspondiente que terminará recibiendo el cliente minorista. 

Este trabajo consta de un software que permite crear en forma automática los pedidos de forma que leamos una carpeta donde van llegando las imágenes que va captando el 
sensor y que con esa información (producto + color) vayamos armando en forma automática los pedidos que luego el brazo robot irá acomodando. La carpeta contiene 
todos los productos de un mismo lote. 

#### Botellas 🍾

Llevan el código de artículo Nro 1334, tienen un precio de lista de 15 dólares, pesan 450gr y pueden venir en Verde, Rojo, Azul, Negro y Amarillo. 

#### Vasos 🥤

Llevan el código de Artículo 568, tienen un precio de lista de 8 dólares, pesan 350gr. y pueden venir en Negro y Azul. 

#### Zonas geográficas donde se distribuyen los pedidos 🌍
Por otra parte la empresa dividió la argentina en 3 zonas geográficas y CABA. 
- Zona Norte: Todas las ciudades cuya latitud sea menor a 35° 
- Zona centro: Todas las ciudades entre la latitud 35 y 40 grados 
- Zona Sur: Todas las ciudades cuya latitud sea mayor a 40 grados. 
- CABA: Todos los pedidos que sean de CABA. 

#### Utilitarios 🚚
Para cubrir estas zonas posee 4 utilitarios con capacidad de carga diferentes. Estos pueden ser utilizados según convenga, pero cada uno deberá cubrir una zona 
determinada, con lo cual un utilitario no podría cubrir dos zonas en un mismo viaje. 
- Utilitario 001 posee una capacidad de carga de 600kg 
- Utilitario 002 tiene una capacidad de carga de 1000kg 
- Utilitario 003 tiene una capacidad de carga de 500kg 
- Utilitario 004 tiene una capacidad de carga de 2000kg. 
