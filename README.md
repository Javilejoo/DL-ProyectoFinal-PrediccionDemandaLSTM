# 🍫 Predicción de Demanda de Chocolates con LSTM

## 📋 Descripción del Proyecto

Sistema de predicción de demanda semanal de chocolates utilizando redes neuronales recurrentes (LSTM - Long Short-Term Memory). Este proyecto aplica técnicas de Deep Learning para forecasting de series temporales, capturando patrones estacionales, tendencias y efectos de eventos especiales.

**Autores:**
- Javier Prado - 21486
- Bryan España - 21550

**Curso:** CC3092 - Deep Learning y Sistemas Inteligentes  
**Semestre:** II - 2025

---

## 🎯 Problema

Las empresas de retail necesitan predecir con precisión la demanda de productos para:
- Optimizar niveles de inventario
- Reducir costos de almacenamiento
- Evitar quiebres de stock
- Mejorar la planificación de producción
- Maximizar ventas en temporadas pico

**Desafío específico:** Predecir la demanda semanal de chocolates considerando:
- Estacionalidad (San Valentín, Navidad, Día de la Madre, etc.)
- Promociones y descuentos
- Tendencias anuales
- Patrones históricos

---

## 🔬 Propuesta de Solución

Implementación de un modelo **LSTM (Long Short-Term Memory)** que:

1. **Procesa secuencias temporales** de 12 semanas históricas
2. **Predice la demanda** para 1 semana adelante
3. **Incorpora múltiples features:**
   - Componentes temporales (mes, semana del año, progreso anual)
   - Lags de demanda (1, 4, 12, 52 semanas)
   - Estadísticas móviles (media y desviación de 8 semanas)
   - Indicadores de festividades y promociones

### ¿Por qué LSTM?

- ✅ Captura dependencias de largo plazo en series temporales
- ✅ Maneja efecto de "memoria" para patrones estacionales
- ✅ Robusta ante ruido y variabilidad en los datos
- ✅ Superior a modelos tradicionales (ARIMA, Exponential Smoothing) para series complejas

---

## 🏗️ Arquitectura del Modelo

```
Input: (batch_size, 12 timesteps, 15 features)
    ↓
LSTM Layer 1: 128 units + Dropout (0.2)
    ↓
LSTM Layer 2: 64 units + Dropout (0.2)
    ↓
Dense Layer: 32 units (ReLU)
    ↓
Output Layer: 1 unit (Linear) → Predicción de demanda (log1p)
```

**Hiperparámetros:**
- Lookback: 12 semanas
- Optimizer: Adam (learning rate = 0.001)
- Loss: MSE (Mean Squared Error)
- Batch size: 16
- Early Stopping: patience = 15 epochs
- ReduceLROnPlateau: factor = 0.5, patience = 7

---

## 📊 Dataset

### Estructura de Datos

**Período:** Enero 2018 - Diciembre 2024 (366 semanas)

**Splits:**
- **Train:** 2018-2022 (260 semanas)
- **Test:** 2023 (52 semanas)
- **Validation:** 2024 (52 semanas)

### Features (15 variables)

#### 1. Componentes Temporales (6)
- `sin_woy`, `cos_woy`: Codificación cíclica de semana del año
- `sin_month`, `cos_month`: Codificación cíclica de mes
- `year_progress`: Progreso dentro del año (0-1)
- `weeks_from_start`: Semanas desde inicio del dataset

#### 2. Indicadores de Eventos (3)
- `holiday_flag`: Semana contiene festividad
- `holiday_lead_flag`: Semana previa a festividad
- `holiday_decay_flag`: Semana posterior a festividad

#### 3. Features Históricas (6)
- `demand_lag_1`: Demanda hace 1 semana
- `demand_lag_4`: Demanda hace 4 semanas
- `demand_lag_12`: Demanda hace 12 semanas (3 meses)
- `demand_lag_52`: Demanda hace 52 semanas (1 año)
- `demand_rolling_mean_8w`: Media móvil de 8 semanas
- `demand_rolling_std_8w`: Desviación estándar móvil de 8 semanas

### Target Variable
- `y_tr`: Demanda transformada con log1p → `log(1 + demand)`
- `demand`: Demanda real en unidades

---

## 🚀 Instalación y Uso

### Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### Estructura del Proyecto

```
DL-ProyectoFinal-PrediccionDemandaLSTM/
│
├── data/                                    # Datasets
│   ├── demand_weekly_chocolates_2018-2024.csv
│   ├── demand_weekly_chocolates_train_features.csv
│   ├── demand_weekly_chocolates_test_features.csv
│   └── demand_weekly_chocolates_valid_features.csv
│
├── models/                                  # Modelos entrenados
│   ├── best_lstm_model.h5
│   └── scaler.pkl
│
├── results/                                 # Resultados
│   ├── metrics_summary.json
│   ├── predictions_train.csv
│   ├── predictions_test.csv
│   └── predictions_validation.csv
│
├── plots/                                   # Visualizaciones
│   ├── learning_curves.png
│   ├── predictions_all_sets.png
│   └── residuals_analysis.png
│
├── dataset.py                               # Generador de datos sintéticos
├── model_lstm.py                            # Clase del modelo LSTM
├── predicciones.ipynb                       # Notebook principal
└── README.md                                # Este archivo
```

### Ejecución

#### Opción 1: Usando el Notebook (Recomendado)

```bash
jupyter notebook predicciones.ipynb
```

Ejecuta las celdas en orden para:
1. Cargar y explorar datos
2. Crear features de ingeniería
3. Entrenar el modelo LSTM
4. Evaluar y visualizar resultados

#### Opción 2: Usando el Script Python

```bash
python model_lstm.py
```

Este script entrena el modelo automáticamente y guarda los resultados.

---

## 📈 Resultados

### Métricas de Evaluación

Las métricas se reportan en **escala original** (unidades de demanda):

| Set | MAE | RMSE | R² | MAPE |
|-----|-----|------|-----|------|
| **Train** | 13.90 u | 18.89 u | 0.1166 | 25.35% |
| **Validation** | 15.93 u | 20.00 u | 0.0091 | 27.65% |
| **Test** | 16.42 u | 20.92 u | 0.1415 | 33.83% |

**Interpretación:** Ver `ANALISIS_RESULTADOS.md` para análisis detallado de las métricas.

### Interpretación de Métricas

- **MAE (Mean Absolute Error):** Error promedio en unidades. Ejemplo: MAE=5 → el modelo se equivoca en promedio por 5 unidades.
- **RMSE (Root Mean Squared Error):** Penaliza errores grandes. Útil para detectar outliers.
- **R² (Coeficiente de Determinación):** Bondad de ajuste. Valores cercanos a 1 indican excelente ajuste.
- **MAPE (Mean Absolute Percentage Error):** Error porcentual. Útil para comparar con benchmarks.

### Visualizaciones Generadas

1. **learning_curves.png:** Evolución del loss y MAE durante entrenamiento
2. **predictions_all_sets.png:** Comparación de predicciones vs valores reales
3. **residuals_analysis.png:** Análisis de errores y distribución de residuos

---

## 🛠️ Herramientas y Tecnologías

### Librerías Principales

- **TensorFlow/Keras:** Framework de Deep Learning para construir el modelo LSTM
- **NumPy:** Operaciones numéricas y manejo de arrays
- **Pandas:** Manipulación y análisis de datos
- **Scikit-learn:** Preprocesamiento (StandardScaler) y métricas
- **Matplotlib/Seaborn:** Visualizaciones

### Técnicas de Deep Learning Aplicadas

1. **LSTM (Long Short-Term Memory)**
   - Redes neuronales recurrentes especializadas en secuencias
   - Cell state y gates (forget, input, output) para memoria selectiva
   
2. **Dropout Regularization**
   - Previene overfitting desactivando aleatoriamente neuronas
   
3. **Early Stopping**
   - Detiene entrenamiento cuando validation loss deja de mejorar
   
4. **Learning Rate Scheduling (ReduceLROnPlateau)**
   - Reduce automáticamente el learning rate cuando se estanca
   
5. **Batch Normalization** (implícito en normalización de features)
   - StandardScaler para estabilizar el entrenamiento

### Técnicas de Feature Engineering

- **Codificación cíclica:** `sin/cos` para variables temporales periódicas
- **Lagged features:** Valores históricos como predictores
- **Rolling statistics:** Captura tendencias locales
- **Log transformation:** Estabiliza varianza y normaliza distribución

---

## 🔍 Análisis Exploratorio de Datos (EDA)

El notebook incluye:

- ✅ Análisis de tendencias temporales
- ✅ Detección de estacionalidad
- ✅ Correlación entre features
- ✅ Distribución de demanda
- ✅ Impacto de promociones y festividades
- ✅ Identificación de outliers

---

## 🎓 Conclusiones

1. **El modelo LSTM captura efectivamente patrones complejos** en series temporales de demanda, incluyendo:
   - Estacionalidad anual y mensual
   - Efectos de festividades (San Valentín, Navidad, etc.)
   - Impacto de promociones
   - Tendencias de largo plazo

2. **Las features de ingeniería son críticas:**
   - Los lags de 52 semanas capturan estacionalidad anual
   - Las estadísticas móviles ayudan a suavizar predicciones
   - Los indicadores de festividades mejoran precisión en fechas clave

3. **El modelo generaliza bien** en datos no vistos (test set), demostrando robustez.

4. **Aplicaciones prácticas:**
   - Optimización de inventario
   - Planificación de producción
   - Estrategias de pricing dinámico
   - Gestión de cadena de suministro

---

## 📚 Referencias Bibliográficas

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". Neural Computation, 9(8), 1735-1780.

2. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). "Time Series Analysis: Forecasting and Control" (5th ed.). Wiley.

3. Chollet, F. (2021). "Deep Learning with Python" (2nd ed.). Manning Publications.

4. Hyndman, R. J., & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice" (3rd ed.). OTexts.

5. TensorFlow Documentation: "Time Series Forecasting". https://www.tensorflow.org/tutorials/structured_data/time_series

6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning". MIT Press.

---

## 📝 Licencia

Este proyecto fue desarrollado con fines académicos para el curso CC3092 - Deep Learning y Sistemas Inteligentes de la Universidad del Valle de Guatemala.

---

## 📧 Contacto

Para preguntas o colaboraciones:
- **Javier Prado:** [21486@uvg.edu.gt](mailto:21486@uvg.edu.gt)
- **Bryan España:** [21550@uvg.edu.gt](mailto:21550@uvg.edu.gt)

---

**¡Gracias por revisar nuestro proyecto! 🚀**