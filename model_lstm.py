"""
Modelo LSTM para Predicción de Demanda de Chocolates
======================================================

Este módulo contiene funciones para entrenar, evaluar y usar el modelo LSTM
para predecir la demanda semanal de chocolates.

Autores: Javier Prado (21486), Bryan España (21550)
Curso: CC3092 - Deep Learning y Sistemas Inteligentes
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


class LSTMDemandPredictor:
    """
    Clase para manejar el modelo LSTM de predicción de demanda.
    """
    
    def __init__(self, lookback=12, horizon=1):
        """
        Inicializa el predictor.
        
        Args:
            lookback (int): Número de semanas históricas a considerar
            horizon (int): Número de semanas hacia adelante a predecir
        """
        self.lookback = lookback
        self.horizon = horizon
        self.model = None
        self.scaler = None
        self.feature_cols = [
            "sin_woy", "cos_woy", "sin_month", "cos_month",
            "year_progress", "weeks_from_start",
            "holiday_flag", "holiday_lead_flag", "holiday_decay_flag",
            "demand_lag_1", "demand_lag_4", "demand_lag_12", "demand_lag_52",
            "demand_rolling_mean_8w", "demand_rolling_std_8w"
        ]
        self.target_col = "y_tr"
    
    def build_model(self, units_layer1=128, units_layer2=64, dropout_rate=0.2, learning_rate=0.001):
        """
        Construye la arquitectura del modelo LSTM.
        
        Args:
            units_layer1 (int): Unidades en la primera capa LSTM
            units_layer2 (int): Unidades en la segunda capa LSTM
            dropout_rate (float): Tasa de dropout
            learning_rate (float): Learning rate del optimizador
        """
        input_shape = (self.lookback, len(self.feature_cols))
        
        self.model = keras.Sequential([
            keras.layers.LSTM(units_layer1, activation='tanh', return_sequences=True, 
                            input_shape=input_shape),
            keras.layers.Dropout(dropout_rate),
            keras.layers.LSTM(units_layer2, activation='tanh', return_sequences=False),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"✅ Modelo construido con {units_layer1}/{units_layer2} unidades LSTM")
        return self.model
    
    def create_sequences(self, X, y=None):
        """
        Crea secuencias temporales para LSTM.
        
        Args:
            X (np.array): Features normalizadas
            y (np.array, optional): Target values
        
        Returns:
            X_seq, y_seq (si y es proporcionado) o solo X_seq
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(self.lookback, len(X) - self.horizon + 1):
            X_seq.append(X[i-self.lookback:i, :])
            if y is not None:
                y_seq.append(y[i + self.horizon - 1])
        
        if y is not None:
            return np.array(X_seq), np.array(y_seq)
        return np.array(X_seq)
    
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=100, batch_size=16, 
            patience=15, verbose=1):
        """
        Entrena el modelo.
        
        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (np.array): Target de entrenamiento
            X_valid (pd.DataFrame): Features de validación
            y_valid (np.array): Target de validación
            epochs (int): Número máximo de épocas
            batch_size (int): Tamaño del batch
            patience (int): Paciencia para early stopping
            verbose (int): Nivel de verbosidad
        
        Returns:
            history: Historia del entrenamiento
        """
        # Normalizar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        
        # Crear secuencias
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train)
        X_valid_seq, y_valid_seq = self.create_sequences(X_valid_scaled, y_valid)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, 
                restore_best_weights=True, verbose=verbose
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, 
                min_lr=1e-6, verbose=verbose
            )
        ]
        
        # Entrenar
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_valid_seq, y_valid_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """
        Realiza predicciones.
        
        Args:
            X (pd.DataFrame): Features para predecir
        
        Returns:
            predictions (np.array): Predicciones en escala log1p
        """
        X_scaled = self.scaler.transform(X)
        X_seq = self.create_sequences(X_scaled)
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        return predictions
    
    def evaluate(self, X, y):
        """
        Evalúa el modelo y retorna métricas.
        
        Args:
            X (pd.DataFrame): Features
            y (np.array): Target real
        
        Returns:
            dict: Métricas en escala log1p y original
        """
        # Predicciones en escala log1p
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        y_pred = self.model.predict(X_seq, verbose=0).flatten()
        
        # Métricas en escala log1p
        mae_log = mean_absolute_error(y_seq, y_pred)
        mse_log = mean_squared_error(y_seq, y_pred)
        rmse_log = np.sqrt(mse_log)
        r2_log = r2_score(y_seq, y_pred)
        
        # Convertir a escala original
        y_seq_orig = np.expm1(y_seq)
        y_pred_orig = np.expm1(y_pred)
        
        # Métricas en escala original
        mae_orig = mean_absolute_error(y_seq_orig, y_pred_orig)
        mse_orig = mean_squared_error(y_seq_orig, y_pred_orig)
        rmse_orig = np.sqrt(mse_orig)
        r2_orig = r2_score(y_seq_orig, y_pred_orig)
        mape_orig = np.mean(np.abs((y_seq_orig - y_pred_orig) / y_seq_orig)) * 100
        
        return {
            'log1p': {
                'mae': mae_log, 'mse': mse_log, 'rmse': rmse_log, 'r2': r2_log
            },
            'original': {
                'mae': mae_orig, 'mse': mse_orig, 'rmse': rmse_orig, 
                'r2': r2_orig, 'mape': mape_orig
            }
        }
    
    def save(self, model_path='models/lstm_model.keras', scaler_path='models/scaler.pkl'):
        """
        Guarda el modelo y el scaler.
        
        Args:
            model_path (str): Ruta para guardar el modelo (.keras formato nativo)
            scaler_path (str): Ruta para guardar el scaler
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Modelo guardado en: {model_path}")
        print(f"✅ Scaler guardado en: {scaler_path}")
    
    def load(self, model_path='models/lstm_model.keras', scaler_path='models/scaler.pkl'):
        """
        Carga el modelo y el scaler.
        
        Args:
            model_path (str): Ruta del modelo guardado (.keras formato nativo)
            scaler_path (str): Ruta del scaler guardado
        """
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Modelo cargado desde: {model_path}")
        print(f"✅ Scaler cargado desde: {scaler_path}")


def load_and_prepare_data(train_path, valid_path, test_path, feature_cols, target_col):
    """
    Carga y prepara los datasets.
    
    Args:
        train_path (str): Ruta del CSV de train
        valid_path (str): Ruta del CSV de validación
        test_path (str): Ruta del CSV de test
        feature_cols (list): Lista de columnas features
        target_col (str): Nombre de la columna target
    
    Returns:
        tuple: (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    # Cargar datos
    train_df = pd.read_csv(train_path, parse_dates=['week_start'])
    valid_df = pd.read_csv(valid_path, parse_dates=['week_start'])
    test_df = pd.read_csv(test_path, parse_dates=['week_start'])
    
    # Ordenar por fecha
    train_df = train_df.sort_values('week_start').reset_index(drop=True)
    valid_df = valid_df.sort_values('week_start').reset_index(drop=True)
    test_df = test_df.sort_values('week_start').reset_index(drop=True)
    
    # Extraer features y target
    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, target_col].values
    
    X_valid = valid_df[feature_cols].dropna()
    y_valid = valid_df.loc[X_valid.index, target_col].values
    
    X_test = test_df[feature_cols].dropna()
    y_test = test_df.loc[X_test.index, target_col].values
    
    print(f"✅ Datos cargados:")
    print(f"   Train: {X_train.shape}")
    print(f"   Valid: {X_valid.shape}")
    print(f"   Test: {X_test.shape}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":
    """
    Ejemplo de uso del modelo.
    """
    print("="*80)
    print("MODELO LSTM - PREDICCIÓN DE DEMANDA DE CHOCOLATES")
    print("="*80)
    
    # Configuración
    FEATURE_COLS = [
        "sin_woy", "cos_woy", "sin_month", "cos_month",
        "year_progress", "weeks_from_start",
        "holiday_flag", "holiday_lead_flag", "holiday_decay_flag",
        "demand_lag_1", "demand_lag_4", "demand_lag_12", "demand_lag_52",
        "demand_rolling_mean_8w", "demand_rolling_std_8w"
    ]
    TARGET_COL = "y_tr"
    
    # Cargar datos
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_prepare_data(
        'data/demand_weekly_chocolates_train_features.csv',
        'data/demand_weekly_chocolates_valid_features.csv',
        'data/demand_weekly_chocolates_test_features.csv',
        FEATURE_COLS,
        TARGET_COL
    )
    
    # Crear y entrenar modelo
    predictor = LSTMDemandPredictor(lookback=12, horizon=1)
    predictor.build_model(units_layer1=128, units_layer2=64, 
                         dropout_rate=0.2, learning_rate=0.001)
    
    print("\n🚀 Entrenando modelo...")
    history = predictor.fit(X_train, y_train, X_valid, y_valid, 
                           epochs=100, batch_size=16, patience=15)
    
    # Evaluar
    print("\n📊 Evaluando modelo...")
    test_metrics = predictor.evaluate(X_test, y_test)
    
    print("\n📈 Métricas en TEST (escala original):")
    print(f"   MAE:  {test_metrics['original']['mae']:.2f} unidades")
    print(f"   RMSE: {test_metrics['original']['rmse']:.2f} unidades")
    print(f"   R²:   {test_metrics['original']['r2']:.4f}")
    print(f"   MAPE: {test_metrics['original']['mape']:.2f}%")
    
    # Guardar modelo
    predictor.save()
    
    print("\n✅ Proceso completado!")
    print("="*80)
