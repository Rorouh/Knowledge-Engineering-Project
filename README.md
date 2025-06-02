# Predicción de Mortalidad por COVID-19 mediante Aprendizaje Automático

## 📄 Descripción general

Este proyecto tiene como objetivo el análisis y predicción de la probabilidad de fallecimiento de pacientes afectados por COVID-19, usando técnicas de aprendizaje automático aplicadas sobre un conjunto de datos reales de pacientes.

Todo el flujo de trabajo ha sido integrado en el fichero `O1reducido.ipynb`, que centraliza las tareas de limpieza de datos, entrenamiento de modelos, evaluación de rendimiento y validación de resultados con datos de prueba. El enfoque principal es la clasificación binaria: predecir si un paciente fallecerá o no en función de sus características médicas.

---

## 📁 Estructura del notebook (`O1reducido.ipynb`)

### 1. **Carga y limpieza de datos**

* Se carga el conjunto de datos `custom_covid19.csv`.
* Se realiza un preprocesamiento específico:

  * Conversión de la columna `DATE_DIED` en variable binaria `FALLECIDO`.
  * Eliminación o imputación de valores nulos (valores como 97, 98, 99).
  * Transformación de todas las variables binarias al formato 1 (sí) / 0 (no).
  * Eliminación de valores de edad no válidos (fuera del rango 0-110).

### 2. **Separación de variables**

* Se define `X` como las variables predictoras (exceptuando `FALLECIDO`).
* La variable `y` representa la etiqueta objetivo: si el paciente ha fallecido (1) o no (0).

### 3. **Imputación de valores nulos**

* Se utiliza `SimpleImputer` con la estrategia `'most_frequent'` para rellenar valores ausentes en las variables predictoras.

### 4. **Entrenamiento de modelos (clasificación)**

* Se utiliza **Logistic Regression** como clasificador principal.
* Se aplica `GridSearchCV` para optimizar hiperparámetros con validación cruzada (5-fold).
* Se repite el proceso para cuatro métricas distintas:

  * **Accuracy**
  * **Recall**
  * **Precision**
  * **F1-score**

### 5. **Evaluación del modelo**

* Para cada métrica optimizada se imprime:

  * Los mejores hiperparámetros encontrados.
  * La matriz de confusión.
  * El informe de clasificación (`precision`, `recall`, `f1-score`, `support`).
* Se comparan los resultados para determinar qué modelo ofrece mejor rendimiento para distintos criterios clínicos (ej. mayor sensibilidad o mayor precisión).

### 6. **Predicción con conjunto externo**

* Se evalúa el modelo sobre un conjunto externo: `proj-test-data.csv`.
* El modelo predice la probabilidad de fallecimiento en este conjunto y compara los resultados reales con los estimados.
* Se imprime una tabla comparativa con los resultados.

---

## 📊 Métricas utilizadas

* **Accuracy**: proporción total de aciertos sobre el total de muestras.
* **Recall (sensibilidad)**: cuántos verdaderos positivos se identifican correctamente (importante para detectar muertes reales).
* **Precision**: cuántos de los positivos predichos son verdaderamente positivos (evita falsos positivos).
* **F1-score**: media armónica entre precisión y sensibilidad.

---

## 📈 Interpretación de resultados

El uso de distintas métricas permite ajustar el modelo según distintos objetivos:

* Si el objetivo es **detectar todos los fallecimientos posibles** (aunque haya falsos positivos), se prioriza el **recall**.
* Si se quiere **minimizar alarmas falsas**, se prioriza la **precisión**.
* El **f1-score** se utiliza como medida de equilibrio.

---

## 🛠️ Requisitos

* Python 3.8+
* Bibliotecas:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `matplotlib`

---

## 📂 Archivos importantes

* `O1reducido.ipynb`: notebook principal con todas las etapas del proyecto integradas.
* `custom_covid19.csv`: conjunto de datos original de entrenamiento.
* `proj-test-data.csv` y `proj-test-class.csv`: conjunto externo de validación.

---

## ✅ Conclusiones

Este proyecto demuestra cómo técnicas de clasificación como la regresión logística pueden utilizarse de forma eficiente para **predecir resultados clínicos relevantes**, siempre y cuando exista una **correcta preparación de datos** y se optimicen los modelos con una **evaluación adecuada de métricas**.

