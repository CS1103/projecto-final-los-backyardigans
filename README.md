[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final


### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#investigación-teórica)
4. [Diseño e implementación](#diseño-e-implementación)
5. [Análisis del trabajo en equipo](#análisis-del-trabajo-en-equipo)
6. [Conclusiones](#conclusiones)
7. [Bibliografía](#bibliografía)


---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `Los Backyardigans`
* **Integrantes**:

  * Leo Alexander Torres Ccencho –  202410078 (Responsable de investigación teórica)
  * Aaron Adriano Romano Castro  – 202410322 (Pruebas y benchmarking)
  * Mauricio Eduardo Terán Taica– 202410404 (Implementación del modelo)
  * Christopher Renato Perez Torres– 202410057 (Desarrollo de la arquitectura)


---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```



---

### 1. Investigación teórica

* **Objetivo**: Comprender los fundamentos matemáticos y computacionales de las redes neuronales artificiales, así como analizar su implementación modular en C++ mediante el desarrollo de un clasificador de imágenes numéricas (dígitos 0 y 1) basado en sus representaciones en píxeles.



#### 1.1 Fundamentos matemáticos
El comportamiento de una red neuronal está respaldado por una sólida base matemática, principalmente desde tres ramas: álgebra lineal, cálculo diferencial y optimización.


**Álgebra lineal**  
Esencial para representar los datos, parámetros y operaciones internas de las redes neuronales:

- **Multiplicación matricial**: permite modelar el paso de información entre capas completamente conectadas.
- **Transposición**: usada para compatibilizar las dimensiones entre tensores en operaciones de retropropagación.
- **Broadcasting**: técnica para realizar operaciones entre tensores de distintas dimensiones, replicando valores según sea necesario, lo cual facilita la eficiencia en cálculos.

**Cálculo diferencial**

- **Derivadas parciales y gradientes**: necesarios para cuantificar cómo los cambios en los pesos afectan la función de pérdida.
- **Regla de la cadena**: herramienta central para calcular gradientes en redes profundas (backpropagation).
- **Retropropagación (backpropagation)**: algoritmo que aplica la regla de la cadena para propagar errores desde la salida hasta la entrada, actualizando pesos en consecuencia.

**Optimización**  
Para que una red neuronal aprenda, es necesario minimizar una función de pérdida. Este proceso se realiza mediante algoritmos de optimización:

- **Descenso del gradiente (SGD)**: actualiza los pesos en la dirección opuesta al gradiente.
- **Momentum**: técnica que acumula parte del gradiente pasado para evitar oscilaciones.
- **Adam Optimizer**: combina adaptativamente las ventajas de RMSProp y Momentum, acelerando la convergencia.

#### 1.2 Arquitecturas de redes neuronales
Se ha investigado y tomado como base arquitecturas fundamentales de redes neuronales artificiales:


**Perceptrón multicapa (MLP)**  
- Es una red neuronal compuesta por una o más capas ocultas y una capa de salida.
- Cada capa aplica una transformación lineal seguida de una función de activación no lineal.

**Capas densas (Fully Connected)**

- Cada neurona de una capa está conectada con todas las neuronas de la siguiente capa.
- Esta estructura se implementa mediante productos matriciales entre las entradas y los pesos.

**Funciones de activación**  
Estas funciones permiten introducir no linealidad al modelo, habilitando su capacidad de aprender patrones complejos:

- **ReLU (Rectified Linear Unit)**: activa valores positivos y anula los negativos.
- **Sigmoid**: útil en tareas de clasificación binaria.
- **Tanh**: función hiperbólica que centra los datos alrededor de cero.

#### 1.3 Algoritmos de Entrenamiento

**Forward propagation**  
Propaga los datos de entrada a través de las capas hasta obtener una predicción.

**Backpropagation**  
Calcula los gradientes de la función de pérdida respecto a los pesos usando la regla de la cadena.

**Optimizadores**

- **SGD (Stochastic Gradient Descent)**: actualiza los parámetros de manera simple pero efectiva.
- **Adam**: ajusta dinámicamente la tasa de aprendizaje para cada parámetro individual.

**Funciones de pérdida**

- **MSE (Mean Squared Error)**: utilizada en problemas de regresión.
- **Binary Cross-Entropy**: adecuada para tareas de clasificación binaria.


#### 1.4 Implementación en C++

El sistema se desarrolló usando C++ moderno, haciendo uso intensivo de técnicas avanzadas del lenguaje:

**Gestión eficiente de memoria**  
Se emplean estructuras como `std::vector` y `std::array` para manejar memoria de forma segura y eficiente, evitando fugas o errores de acceso.

**Uso de templates**  
La clase `Tensor<T, Rank>` está implementada con plantillas, lo que permite una gran flexibilidad y reutilización del código para distintos tipos de datos (por ejemplo, `float`, `double`) y diferentes dimensiones.

**Patrones de diseño**

- **Strategy**: usado para intercambiar dinámicamente funciones de pérdida y optimizadores sin modificar la estructura base del código.
- **Factory (implícito)**: el sistema puede extenderse fácilmente para instanciar nuevos tipos de capas, funciones o configuraciones.


---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

El módulo `neural_network.h` representa el núcleo estructural de la arquitectura de red neuronal artificial desarrollada en el presente proyecto. La clase genérica `NeuralNetwork<T>` encapsula un enfoque modular y escalable, permitiendo construir modelos de aprendizaje profundo mediante una colección dinámica de capas (`layers`) que cumplen con la interfaz `ILayer<T>`.

El diseño está fundamentado en principios de programación orientada a objetos y hace uso intensivo de polimorfismo dinámico mediante punteros inteligentes (`std::unique_ptr`). Esto garantiza no solo una gestión eficiente de memoria, sino también una extensibilidad natural del sistema, permitiendo la integración futura de nuevas capas, funciones de activación o mecanismos de regularización sin necesidad de modificar el *core* del sistema.

Dentro de la clase, el método `add_layer()` posibilita la construcción incremental de la arquitectura de la red. Por su parte, el método `forward()` permite propagar una entrada `Tensor<T,2>` a través de todas las capas, de forma secuencial y ordenada, obteniendo así la salida final del modelo. Esta función es clave tanto para el entrenamiento como para la inferencia.

El entrenamiento se gestiona mediante el método `train()`, el cual ha sido sobrecargado para soportar distintos tipos de funciones de pérdida (`MSELoss`, `BCELoss`) y optimizadores (`SGD`, `Adam`). Esta versatilidad se logra a través de *plantillas de plantilla* (`template <template<typename> class>`), una característica avanzada de C++ que aporta flexibilidad al framework sin sacrificar eficiencia.

La clase además incorpora funciones auxiliares como `predict()` (para inferencias directas), `print_predictions()` (útil en tareas de validación visual), y `print_accuracy()` (para evaluación cuantitativa del desempeño en conjuntos de datos de prueba).



#### 2.2 Manual de uso y casos de prueba

##### 2.2.1 Carga de datos desde archivos CSV

```cpp
Tensor<float, 2> X = data::load_inputs("../data/digits01_inputs.csv");
Tensor<float, 2> Y = data::load_labels("../data/digits01_labels.csv");
```

-**¿Qué hace?**
Carga 360 muestras de imágenes (matriz X) y sus etiquetas correspondientes (Y), que indican si el dígito es un 0 o un 1.


-**Formato:**
Cada imagen tiene 64 características (8x8 pixeles planos) y cada etiqueta es un valor binario (0 o 1).

#####  2.2.2 Barajado aleatorio y división en entrenamiento/prueba

```cpp
std::vector<size_t> indices(total_samples);
std::iota(indices.begin(), indices.end(), 0);
std::shuffle(indices.begin(), indices.end(), rng);
```

-**¿Qué hace?**
Carga 360 muestras de imágenes (matriz X) y sus etiquetas correspondientes (Y), que indican si el dígito es un 0 o un 1.

-Luego divide los datos en conjuntos de entrenamiento en `X_train` y `Y_train` (80%) y prueba en `X_test` y `Y_test` (20%).

##### 2.2.3 Agregado de ruido a las entradas

```cpp
X(i, j) += noise(noise_gen);
```
-**¿Qué hace?**
Simula variabilidad en los datos, agregando ruido gaussiano con media 0 y desviación 0.1.

-**¿Por qué?**
Ayuda a que la red no memorice y generalice mejor. Aumenta la robustez.


##### 2.2.4 Inicialización de pesos (Xavier)

```cpp
auto xavier_init = [&](auto& param) { ... }
```
-**¿Qué hace?**
Inicializa los pesos de la red de forma que se mantenga el flujo de señales estable en todas las capas.

-**¿Por qué?**
Evita que los gradientes exploten o desaparezcan durante el entrenamiento.

##### 2.2.5 Construcción de la red neuronal

```cpp
net.add_layer(...);
```
-**Arquitectura**
- `Dense(64, 16)`→ capa oculta con 16 neuronas
- `ReLU` → activación no lineal
- `Dense(16, 1)` → salida con 1 neurona
- `Sigmoid` → convierte la salida en probabilidad entre 0 y 1


##### 2.2.6 Entrenamiento

```cpp
net.train<BCELoss>(X_train, Y_train, epochs, batch_size, learning_rate);
```
-**Función de pérdida**: `BCELoss` (Binary Cross Entropy)

-**Épocas**: 500 iteraciones completas sobre los datos

-**Tamaño de batch**: 32 muestras por mini-lote

-**Tasa de aprendizaje**: 0.01, controla la magnitud de las actualizaciones de pesos


##### 2.2.7 Evaluación
```cpp
auto preds = net.predict(X_test);
```
-Predice la clase de los datos de prueba.

-Imprime 10 muestras aleatorias con su probabilidad, clasificación y valor real.

-Calcula el `accuracy` del modelo como porcentaje de aciertos sobre el total de muestras.


##### 2.2.8 Interpretación de resultados

```cpp
float accuracy = ...;
std::cout << "\nTest Accuracy: " << accuracy * 100 << "%\n";

```

-Mide cuántas predicciones fueron correctas sobre el conjunto de prueba.

-Si el `accuracy` es muy alto (por ejemplo 100%) puede ser porque los datos son simples o bien linealmente separables.


---

### 3. Trabajo en equipo

| Tarea                     | Miembro                             | Rol                       |
|---------------------------|-------------------------------------|---------------------------|
| Investigación teórica     | Leo Alexander Torres Ccencho       | Documentar bases teóricas |
| Diseño de la arquitectura | Aaron Adriano Romano Castro        | UML y esquemas de clases  |
| Implementación del modelo | Mauricio Eduardo Terán Taica       | Código C++ de la NN       |
| Pruebas y benchmarking    | Christopher Renato Perez Torres     | Generación de métricas    |

---

### 4. Conclusiones

Este proyecto permitió comprender a fondo los fundamentos matemáticos y computacionales de las redes neuronales artificiales, aplicándolos en una implementación modular en C++. Se logró construir un sistema flexible y escalable, usando técnicas avanzadas del lenguaje como templates, punteros inteligentes y patrones de diseño.
El modelo fue probado con datos reales de imágenes binarias (dígitos 0 y 1), alcanzando una alta precisión gracias a buenas prácticas como la inicialización adecuada de pesos, la introducción de ruido para mejorar la generalización y el uso de optimizadores eficientes.
Además, el trabajo en equipo permitió distribuir las tareas de forma equilibrada, combinando teoría, desarrollo e implementación, lo que fortaleció tanto las habilidades técnicas como la colaboración entre los integrantes.






---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
