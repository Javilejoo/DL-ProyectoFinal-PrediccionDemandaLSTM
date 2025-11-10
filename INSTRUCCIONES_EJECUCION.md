# 🚀 Instrucciones de Ejecución - Proyecto LSTM

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (para clonar el repositorio)
- 4 GB RAM mínimo
- Espacio en disco: ~500 MB

---

## ⚙️ Instalación Paso a Paso

### 1️⃣ Clonar el Repositorio

```bash
git clone https://github.com/Javilejoo/DL-ProyectoFinal-PrediccionDemandaLSTM.git
cd DL-ProyectoFinal-PrediccionDemandaLSTM
```

### 2️⃣ Crear Ambiente Virtual

**En Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Si tienes error de permisos:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**En Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Verificar que el ambiente esté activado (verás `(venv)` al inicio de la línea).

### 3️⃣ Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instalará:
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter

---

## 🎯 Opciones de Ejecución

### Opción A: Notebook Jupyter (⭐ Recomendado para exploración)

**Ideal para:** Ver el análisis completo paso a paso, visualizaciones interactivas, y entender el proceso.

```bash
# Iniciar Jupyter Notebook
jupyter notebook
```

Se abrirá tu navegador automáticamente. Luego:

1. Haz clic en `predicciones.ipynb`
2. En el menú superior: `Cell` → `Run All` (o `Ctrl + A` + `Shift + Enter`)
3. Espera ~5-10 minutos para que ejecute todas las celdas

**¿Qué hace el notebook?**
- Genera datos sintéticos (366 semanas, 2018-2024)
- Realiza análisis exploratorio (EDA)
- Crea 15 features de ingeniería
- Divide datos (train/valid/test)
- Entrena modelo LSTM (2 capas, 125K parámetros)
- Evalúa con múltiples métricas (MAE, RMSE, R², MAPE)
- Genera 4 visualizaciones
- Guarda modelo y resultados

### Opción B: Script Python (⚡ Ejecución rápida)

**Ideal para:** Entrenar el modelo directamente sin interfaz gráfica.

```bash
python model_lstm.py
```

Este script:
- Genera los datos
- Entrena el modelo automáticamente
- Guarda `best_lstm_model.keras` en `models/`
- Tiempo estimado: ~5 minutos

### Opción C: Visual Studio Code (🔧 Para desarrollo)

**Ideal para:** Modificar código, experimentar con parámetros.

1. Abre VS Code
2. `File` → `Open Folder` → Selecciona la carpeta del proyecto
3. Instala la extensión "Jupyter" (Microsoft)
4. Abre `predicciones.ipynb`
5. Selecciona el kernel: `Python 3.x ('venv': venv)`
6. Ejecuta celdas con `Shift + Enter`

---

## 📊 Verificar Resultados

Después de ejecutar, verifica que se hayan creado:

```
DL-ProyectoFinal-PrediccionDemandaLSTM/
├── models/
│   ├── best_lstm_model.keras  ✅ (Modelo entrenado, ~500 KB)
│   └── scaler.pkl              ✅ (Escalador de datos)
├── results/
│   └── metrics_summary.json    ✅ (Métricas finales)
├── plots/
│   ├── learning_curves.png     ✅ (Gráfica de entrenamiento)
│   ├── predictions_all_sets.png ✅ (Predicciones vs reales)
│   ├── residuals_analysis.png  ✅ (Análisis de errores)
│   └── prediction_next_week.png ✅ (Forecast próxima semana)
└── data/
    ├── demand_weekly_chocolates_2018-2024.csv ✅
    ├── demand_weekly_chocolates_2018-2024_with_features.csv ✅
    ├── demand_weekly_chocolates_train_features.csv ✅
    ├── demand_weekly_chocolates_valid_features.csv ✅
    └── demand_weekly_chocolates_test_features.csv ✅
```

---

## 🔬 Hacer Predicciones con el Modelo Entrenado

### Desde Python:

```python
from model_lstm import LSTMDemandPredictor

# Cargar modelo entrenado
model = LSTMDemandPredictor.load('models/best_lstm_model.keras')

# Cargar datos históricos
import pandas as pd
df = pd.read_csv('data/demand_weekly_chocolates_2018-2024_with_features.csv')

# Predecir próxima semana
next_week_demand = model.predict_next_week(df)
print(f"Demanda predicha para próxima semana: {next_week_demand:.2f} unidades")

# Predecir múltiples semanas
future_predictions = model.predict_multiple_weeks(df, weeks=4)
print(f"Predicciones para próximas 4 semanas: {future_predictions}")
```

### Desde el Notebook:

Ejecuta las últimas celdas del notebook `predicciones.ipynb` que incluyen:
- Predicción de la próxima semana
- Visualización del forecast
- Intervalos de confianza

---

## 🐛 Solución de Problemas Comunes

### Error: "No module named 'tensorflow'"

**Solución:**
```bash
pip install tensorflow==2.10.0
```

### Error: "Permission Denied" al activar venv en Windows

**Solución:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: "Jupyter command not found"

**Solución:**
```bash
pip install jupyter notebook
```

### El modelo no mejora (loss muy alto)

**Causas posibles:**
- Datos sintéticos con mucho ruido (esperado)
- Learning rate muy alto/bajo
- Pocas épocas de entrenamiento

**Solución:** Modifica hiperparámetros en el notebook (celda de configuración).

### Error: "Out of Memory (OOM)"

**Solución:** Reduce el `batch_size` de 16 a 8 o 4 en la configuración.

---

## ⚡ Ejecución Rápida (TL;DR)

```bash
# Clonar, instalar y ejecutar todo en 4 comandos
git clone https://github.com/Javilejoo/DL-ProyectoFinal-PrediccionDemandaLSTM.git
cd DL-ProyectoFinal-PrediccionDemandaLSTM
python -m venv venv && .\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt && jupyter notebook predicciones.ipynb
```

---

## 📖 Archivos Importantes

| Archivo | Descripción |
|---------|-------------|
| `predicciones.ipynb` | 📔 Notebook principal con análisis completo |
| `model_lstm.py` | 🧠 Clase reutilizable del modelo LSTM |
| `dataset.py` | 📊 Generador de datos sintéticos |
| `README.md` | 📄 Documentación general del proyecto |
| `INFORME_TECNICO.md` | 📝 Reporte académico detallado |
| `ANALISIS_RESULTADOS.md` | 📈 Interpretación de métricas |
| `requirements.txt` | 📦 Dependencias de Python |

---

## 🎓 Recomendaciones para la Presentación

1. **Ejecuta el notebook completo** antes de la presentación para tener resultados frescos
2. **Toma screenshots** de las gráficas más importantes
3. **Anota las métricas finales** (MAE, R², MAPE)
4. **Prepara ejemplos** de predicción para fechas específicas (ej: Navidad)
5. **Explica el R² bajo** usando `ANALISIS_RESULTADOS.md`

---

## ⏱️ Tiempos Estimados

| Tarea | Tiempo |
|-------|--------|
| Instalación inicial | 5-10 min |
| Ejecución notebook completo | 5-10 min |
| Entrenamiento del modelo | 2-5 min |
| Generación de visualizaciones | 1 min |
| **Total** | **~15-25 min** |

---

## 📞 Soporte

Si tienes problemas durante la ejecución:

1. Revisa que Python >= 3.8: `python --version`
2. Verifica que las dependencias estén instaladas: `pip list`
3. Consulta los logs de error en la terminal
4. Revisa la sección de "Solución de Problemas" arriba

---

**¡Listo! Ahora tienes todo para ejecutar el proyecto exitosamente. 🎉**

Para más detalles técnicos, consulta:
- `README.md` - Documentación general
- `INFORME_TECNICO.md` - Teoría y metodología
- `ANALISIS_RESULTADOS.md` - Interpretación de métricas
