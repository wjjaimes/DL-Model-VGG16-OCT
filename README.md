# DL-Model-VGG16-OCT
El código presentado corresponde con el utilizado para crear el modelo del estudio "Detection of Retinal Diseases from OCT Images using a VGG16 and Transfer Learning".

Para el estudio se utilizó la base de imágenes de OCT macular disponible en el repositorio Kaggle https://www.kaggle.com/paultimothymooney/kermany2018, que cuenta con 84.495 imágenes de OCT dispuestas en tres conjuntos (train, val y test), los cuales contienen 4 categorías (CNV, DME, DRUSEN y NORMAL). El código realizado contiene una transferencia de aprendizaje utilizando la red convolucional VGG16, a la cual se le aplican cambios en la estructura de entrada, bloques de convolución y salida para mejorar el rendimiento clasificatorio.

La metodología utilizada incluyó un balanceo de clases y una redistribución de conjuntos de las imágenes, una validación cruzada de Monte Carlo de diez
iteraciones, un diseño de experimentos para encontrar los parámetros e hiperparámetros del modelo así como una validación cruzada de 5 iteraciones, de la cual se seleccionó el mejor modelo. Este modelo se probó con 3.448 imágenes, obteniendo resultados comparables con estudios que utilizan la misma base de imágenes.

El modelo se encuentra disponible en: https://www.dropbox.com/s/6nxdwu2tlqfzl83/DL-Model-VGG16-OCT.h5?dl=0.

Trabajo realizado por:
Wilwer J. Jaimes,
Wilson J. Arenas,
Humberto J. Navarro,
Miguel Altuve.

![Abstract](https://user-images.githubusercontent.com/67522549/231283758-30c4ad05-2762-476f-9085-6bdf8b4b82e2.jpg)
