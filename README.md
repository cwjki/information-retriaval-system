# Proyecto Final de la asignatura Sistemas de Recuperación de Información

## Ideas Generales
- Hasta ahora hemos implementado un modelo vectorial clásico, que se encuentra en el fichero vector_space_model.py, el cual a través del método compute_ranking (que recibe una query) devuelve 
una cantidad definida anteriormente (RANKING_COUNT) de documentos ordenados por relevancia a la consulta.
Para esto se calcula la similitud entre el vector que representa a la consulta y los vectores de cada documento, computando el peso de los términos de la consulta y el peso de los términos en los documentos,
para computar el peso de los términos de la consulta se agrega una constante de suavizado (ALPHA).
- En el fichero inverted_index.py se provee la clase InvertedIndex, la cual dada la colección de documentos computa
la frecuencia de cada término en cada documento, la cantidad de documentos en los que aparece un término, la mayor frecuencia de aparición de un término en un documento, etc,
datos necesarios para computar el tf e idf en el modelo vectorial.
- En el fichero cranfield_parser.py se encuentra la implementación de la clase CranfieldParser, la cual tiene como objetivo estructurar los documentos de la colección cranfield para su posterior análisis.
- En el fichero dataset, se encuentra la implementación de nuestra clase Dataset para representar los documentos de la colección.
- Hacemos uso del framework Flask para la creación de un sitio web donde dada una consulta se muestren los (RANKING_COUNT) documentos más relevantes a la misma.


## Ejecución del Proyecto
Para la ejecución del proyecto es necesario instalar los paquetes que se encuentran
en el fichero requeriments.txt
```
pip install -r requeriments.txt
```

Luego ejecutar el fichero main.py
```
python main.py
```

El proyecto puede ser visualizado en la dirección http://127.0.0.1:5000



## Pendiente 
- realizar un preprocesamiento de los datos (actualmente solo tokenizamos los términos) falta realizar
técnicas como stemming o lemmatization
- implementar un evaluador para computar las medidas estudiadas en clase (recobrado, precisión, F1)
- implementar un mecanismo de retroalimentación
- realizar comparaciones con otros modelos

