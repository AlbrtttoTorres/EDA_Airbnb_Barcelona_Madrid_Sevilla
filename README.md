# Análisis de Alojamiento Urbano: Airbnb en Barcelona, Madrid y Sevilla

Este proyecto analiza el impacto de Airbnb en tres ciudades principales de España: Barcelona, Madrid y Sevilla. A través de un análisis exploratorio de datos (EDA), se identifican patrones de precios, demanda y estacionalidad, así como el efecto de Airbnb en el mercado de alquileres tradicionales.

## Objetivos del Proyecto
1. Identificar patrones de precios y demanda en diferentes temporadas.
2. Analizar las diferencias entre ciudades y barrios.
3. Evaluar los factores determinantes del precio en Airbnb.
4. Examinar el impacto de Airbnb en el mercado inmobiliario tradicional.
5. Proponer recomendaciones para reguladores, anfitriones y turistas.

## Contenido

### 1. **Contexto**
- Airbnb ha transformado el mercado de alojamientos, generando tanto beneficios como retos en ciudades turísticas.
- Este proyecto investiga cómo Airbnb afecta los precios de alquiler, la disponibilidad de viviendas y las dinámicas del mercado.

### 2. **Análisis Exploratorio de Datos (EDA)**
- **Análisis Univariante**: Distribuciones de precios, reseñas y disponibilidad.
- **Análisis Bivariante**: Comparaciones entre ciudades, tipos de alojamiento y popularidad de barrios.
- **Análisis Multivariante**: Factores determinantes del precio mediante regresión y clustering.

### 3. **Impacto de Airbnb**
- Reducción de la oferta de alquiler tradicional.
- Incremento de precios en barrios céntricos.
- Desplazamiento de residentes locales.

### 4. **Conclusiones**
1. Barcelona tiene los precios más altos, mientras que Sevilla ofrece opciones más económicas.
2. Los apartamentos completos lideran en demanda y precio.
3. Airbnb contribuye significativamente al aumento de los precios de alquiler tradicional.
4. Regulaciones estrictas pueden mitigar estos efectos negativos.

### 5. **Recomendaciones**
#### Para anfitriones:
- Implementar precios dinámicos y mejorar la calidad del alojamiento.
#### Para turistas:
- Reservar con anticipación en temporadas altas y explorar barrios periféricos.
#### Para reguladores:
- Establecer límites en el número de propiedades listadas y fomentar programas de vivienda asequible.

## Estructura del Proyecto
```
|-- mapas ciudades
|-- carpeta src
    |-- src/data/               # Archivos de datos utilizados en el proyecto.
    |-- src/notebooks/          # Notebooks con el EDA y visualizaciones.
    |-- src/images              # Informes y gráficos generados.
|-- main
|-- memoria
|-- presentación
|-- README.md                   # Descripción general del proyecto.
```

## Requisitos
### Librerías necesarias:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `folium`
- `scikit-learn`

### Instalación:
Ejecuta el siguiente comando para instalar las dependencias necesarias:
```
pip install -r requirements.txt
```

## Visualizaciones Destacadas
1. Histogramas de distribución de precios.
2. Series temporales que muestran patrones estacionales.
3. Mapas interactivos de concentración de precios.
4. Clusters de alojamientos basados en estacionalidad, precio y disponibilidad.

## Perspectivas Futuras
- Ampliar el análisis a otras ciudades turísticas como Lisboa o París.
- Integrar datos adicionales, como opiniones de usuarios y regulaciones locales.
- Desarrollar un modelo predictivo para precios de Airbnb basado en factores clave.
