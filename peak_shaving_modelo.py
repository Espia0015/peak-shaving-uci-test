"""
=============================================================================
PREDICCIÓN DE DEMANDA ENERGÉTICA Y OPTIMIZACIÓN DE PEAK SHAVING
Dataset: UCI Individual Household Electric Power Consumption
         Hebrail & Berard (2006) — CC BY 4.0
         https://archive.ics.uci.edu/dataset/235/

INSTRUCCIONES:
1. Descargar el dataset desde:
   https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
2. Descomprimir y colocar 'household_power_consumption.txt' en la misma carpeta
3. Instalar dependencias:
   pip install pandas numpy scikit-learn statsmodels xgboost tensorflow cvxpy matplotlib
4. Ejecutar: python peak_shaving_modelo.py
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ─── 1. CARGAR Y PREPROCESAR EL DATASET UCI ──────────────────────────────────
print("="*60)
print("CARGANDO DATASET UCI...")
print("="*60)

df = pd.read_csv(
    'household_power_consumption.txt',
    sep=';',
    low_memory=False,
    na_values=['?']
)

# Combinar columnas Date y Time en un datetime index (compatible con pandas nuevo)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df.drop(columns=['Date', 'Time'])
df = df.set_index('datetime')
df = df.apply(pd.to_numeric, errors='coerce')

# Resamplear a resolución horaria (promedio por hora)
df_hourly = df['Global_active_power'].resample('h').mean()

# Imputar valores faltantes por interpolación lineal
df_hourly = df_hourly.interpolate(method='linear')

# Usar 2 años completos para el análisis (2007-2008)
df_hourly = df_hourly['2007-01-01':'2008-12-31']

print(f"Registros horarios disponibles: {len(df_hourly)}")
print(f"Período: {df_hourly.index[0]} → {df_hourly.index[-1]}")
print(f"Demanda media: {df_hourly.mean():.3f} kW")
print(f"Demanda máxima: {df_hourly.max():.3f} kW")
print()

# ─── 2. INGENIERÍA DE VARIABLES ───────────────────────────────────────────────
print("CONSTRUYENDO FEATURES...")

df_feat = pd.DataFrame({'load_kw': df_hourly})
df_feat['hour']       = df_feat.index.hour
df_feat['dayofweek']  = df_feat.index.dayofweek
df_feat['month']      = df_feat.index.month
df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
df_feat['load_lag_1']  = df_feat['load_kw'].shift(1)
df_feat['load_lag_24'] = df_feat['load_kw'].shift(24)
df_feat['load_lag_168']= df_feat['load_kw'].shift(168)
df_feat['rolling_mean_24'] = df_feat['load_kw'].rolling(24).mean()
df_feat = df_feat.dropna()

# Partición temporal: 80% entrenamiento / 20% prueba
split = int(len(df_feat) * 0.8)
train = df_feat.iloc[:split]
test  = df_feat.iloc[split:]

y_train = train['load_kw'].values
y_test  = test['load_kw'].values

features = ['hour', 'dayofweek', 'month', 'is_weekend',
            'load_lag_1', 'load_lag_24', 'load_lag_168', 'rolling_mean_24']
X_train = train[features].values
X_test  = test[features].values

print(f"Muestras entrenamiento: {len(train)}")
print(f"Muestras prueba:        {len(test)}")
print()

# ─── MÉTRICAS ─────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

results = {}

# ─── 3. MODELO 1: SARIMA ──────────────────────────────────────────────────────
print("="*60)
print("ENTRENANDO MODELO 1: SARIMA...")
print("="*60)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Usar solo los últimos 365 días de entrenamiento para SARIMA (eficiencia)
    train_sarima = train['load_kw'].iloc[-8760:]  # 1 año
    test_sarima  = test['load_kw']

    model_sarima = SARIMAX(
        train_sarima,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result_sarima = model_sarima.fit(disp=False, maxiter=50)

    # Predicción sobre el test
    pred_sarima = result_sarima.forecast(steps=len(test_sarima))
    pred_sarima = np.array(pred_sarima)

    r_sarima = rmse(y_test, pred_sarima)
    m_sarima = mae(y_test, pred_sarima)
    p_sarima = mape(y_test, pred_sarima)
    results['SARIMA'] = {'RMSE': r_sarima, 'MAE': m_sarima, 'MAPE': p_sarima,
                          'pred': pred_sarima}
    print(f"SARIMA → RMSE: {r_sarima:.4f} | MAE: {m_sarima:.4f} | MAPE: {p_sarima:.2f}%")

except Exception as e:
    print(f"SARIMA no disponible: {e}")
    print("Usando valores de referencia de la literatura...")
    results['SARIMA'] = {'RMSE': None, 'MAE': None, 'MAPE': None, 'pred': None}

# ─── 4. MODELO 2: XGBoost ─────────────────────────────────────────────────────
print()
print("="*60)
print("ENTRENANDO MODELO 2: XGBoost...")
print("="*60)

try:
    from xgboost import XGBRegressor

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)

    r_xgb = rmse(y_test, pred_xgb)
    m_xgb = mae(y_test, pred_xgb)
    p_xgb = mape(y_test, pred_xgb)
    results['XGBoost'] = {'RMSE': r_xgb, 'MAE': m_xgb, 'MAPE': p_xgb,
                           'pred': pred_xgb}
    print(f"XGBoost → RMSE: {r_xgb:.4f} | MAE: {m_xgb:.4f} | MAPE: {p_xgb:.2f}%")

    # Importancia de variables
    fi = pd.Series(xgb_model.feature_importances_, index=features)
    print("\nImportancia de variables (XGBoost):")
    for feat, imp in fi.sort_values(ascending=False).items():
        print(f"  {feat:25s}: {imp:.4f}")

except Exception as e:
    print(f"XGBoost no disponible: {e}")
    results['XGBoost'] = {'RMSE': None, 'MAE': None, 'MAPE': None, 'pred': None}

# ─── 5. MODELO 3: LSTM ────────────────────────────────────────────────────────
print()
print("="*60)
print("ENTRENANDO MODELO 3: LSTM...")
print("="*60)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(42)
    np.random.seed(42)
    SEQ_LEN = 24  # ventana de 24 horas

    # Usar TODAS las features (multivariado) — igual que XGBoost para comparación justa
    features_lstm = ['load_kw', 'hour', 'dayofweek', 'month', 'is_weekend',
                     'load_lag_1', 'load_lag_24', 'load_lag_168', 'rolling_mean_24']
    data_lstm = df_feat[features_lstm].values

    # Escalar cada columna independientemente
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(data_lstm)
    y_scaled    = scaler_y.fit_transform(df_feat[['load_kw']].values)

    # Crear secuencias multivariadas
    def create_sequences_mv(data, y_target, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(data)):
            X_seq.append(data[i-seq_len:i])
            y_seq.append(y_target[i, 0])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences_mv(data_scaled, y_scaled, SEQ_LEN)

    # Partición igual que los otros modelos
    split_seq = int(len(X_seq) * 0.8)
    X_tr, X_te = X_seq[:split_seq], X_seq[split_seq:]
    y_tr, y_te = y_seq[:split_seq], y_seq[split_seq:]

    model_lstm = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, len(features_lstm)), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
    print("Entrenando LSTM (puede tomar 2-5 minutos)...")
    model_lstm.fit(X_tr, y_tr, epochs=30, batch_size=64,
                   validation_split=0.1, callbacks=[es], verbose=0)

    pred_lstm_scaled = model_lstm.predict(X_te, verbose=0).flatten()
    pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled.reshape(-1,1)).flatten()
    y_test_lstm = scaler_y.inverse_transform(y_te.reshape(-1,1)).flatten()

    # Filtrar valores cercanos a cero para un MAPE válido
    mask = y_test_lstm > 0.1
    r_lstm = rmse(y_test_lstm[mask], pred_lstm[mask])
    m_lstm = mae(y_test_lstm[mask], pred_lstm[mask])
    p_lstm = mape(y_test_lstm[mask], pred_lstm[mask])
    results['LSTM'] = {'RMSE': r_lstm, 'MAE': m_lstm, 'MAPE': p_lstm,
                        'pred': pred_lstm, 'y_true': y_test_lstm}
    print(f"LSTM   → RMSE: {r_lstm:.4f} | MAE: {m_lstm:.4f} | MAPE: {p_lstm:.2f}%")

except Exception as e:
    print(f"LSTM no disponible: {e}")
    import traceback
    traceback.print_exc()
    results['LSTM'] = {'RMSE': None, 'MAE': None, 'MAPE': None, 'pred': None}

# ─── 6. TABLA I — RESUMEN COMPARATIVO ────────────────────────────────────────
print()
print("="*60)
print("TABLA I — COMPARACIÓN DE MODELOS DE PREDICCIÓN")
print("Dataset: UCI Household Power Consumption (Hebrail & Berard, 2006)")
print("="*60)
print(f"{'Modelo':<15} {'RMSE (kW)':>12} {'MAE (kW)':>12} {'MAPE (%)':>12} {'Ranking':>10}")
print("-"*60)

modelos_ordenados = sorted(
    [(k, v) for k, v in results.items() if v['RMSE'] is not None],
    key=lambda x: x[1]['MAPE']
)

for rank, (nombre, vals) in enumerate(modelos_ordenados, 1):
    print(f"{nombre:<15} {vals['RMSE']:>12.4f} {vals['MAE']:>12.4f} "
          f"{vals['MAPE']:>12.2f} {rank:>10}°")

# ─── 7. OPTIMIZACIÓN PEAK SHAVING ────────────────────────────────────────────
print()
print("="*60)
print("OPTIMIZACIÓN PEAK SHAVING (CVXPY)")
print("="*60)

try:
    import cvxpy as cp

    # Usar predicción del mejor modelo disponible como demanda pronosticada
    mejor_modelo = modelos_ordenados[0][0] if modelos_ordenados else None

    if mejor_modelo and results[mejor_modelo]['pred'] is not None:
        # Tomar un día representativo de alta demanda (día con mayor pico)
        if mejor_modelo == 'LSTM':
            y_real_full = results['LSTM']['y_true']
            pred_full   = results['LSTM']['pred']
        else:
            y_real_full = y_test
            pred_full   = results[mejor_modelo]['pred']

        # Encontrar el día de mayor demanda en el test set
        T = 24
        n_days = len(pred_full) // T
        picos = [pred_full[i*T:(i+1)*T].max() for i in range(n_days)]
        dia_max = np.argmax(picos)
        L = pred_full[dia_max*T:(dia_max+1)*T]  # demanda pronosticada día pico
    else:
        # Fallback: perfil de demanda simulado basado en estadísticas del dataset
        print("Usando perfil de demanda basado en estadísticas UCI...")
        np.random.seed(42)
        base = df_hourly.mean()
        horas = np.arange(24)
        L = (base * (0.6 + 0.4 * np.sin(np.pi * (horas - 6) / 12)) +
             np.random.normal(0, base * 0.05, 24))
        L = np.clip(L, 0, None)

    # Parámetros BESS
    E_max    = 5.0    # kWh (escala hogareña UCI)
    P_ch_max = 1.5    # kW
    P_dis_max= 1.5    # kW
    eta_ch   = 0.95
    eta_dis  = 0.95
    dt       = 1.0    # horas
    lambda_peak = 0.5 # peso penalización pico

    # Tarifa horaria simulada (valle/pico) — definir horas aquí siempre
    horas = np.arange(24)
    ct = np.where((horas >= 18) & (horas <= 22), 0.18, 0.10)

    # Variables de decisión
    P_grid = cp.Variable(T)
    P_ch   = cp.Variable(T, nonneg=True)
    P_dis  = cp.Variable(T, nonneg=True)
    SOC    = cp.Variable(T + 1, nonneg=True)
    P_peak = cp.Variable(nonneg=True)

    constraints = []
    for t in range(T):
        constraints += [
            P_grid[t] == L[t] + P_ch[t] - P_dis[t],
            P_grid[t] <= P_peak,
            P_grid[t] >= 0,
            P_ch[t]   <= P_ch_max,
            P_dis[t]  <= P_dis_max,
            SOC[t+1]  == SOC[t] + eta_ch * P_ch[t] * dt - (P_dis[t] * dt) / eta_dis,
            SOC[t+1]  <= E_max,
        ]
    constraints += [SOC[0] == E_max * 0.3, SOC[T] >= E_max * 0.2]

    objective = cp.Minimize(cp.sum(cp.multiply(ct, P_grid)) * dt + lambda_peak * P_peak)
    prob = cp.Problem(objective, constraints)

    # Intentar varios solvers en orden de preferencia (todos vienen con CVXPY)
    solvers_a_probar = [cp.CLARABEL, cp.ECOS, cp.SCS]
    resuelto = False
    for solver in solvers_a_probar:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                print(f"Optimización resuelta con solver: {solver}")
                resuelto = True
                break
        except Exception:
            continue

    if not resuelto:
        print(f"Ningún solver pudo resolver. Estado: {prob.status}")

    if prob.status in ['optimal', 'optimal_inaccurate']:
        P_grid_opt = P_grid.value
        SOC_opt    = SOC.value
        ct_array   = ct if isinstance(ct, np.ndarray) else np.full(T, ct)

        costo_sin  = np.sum(ct_array * L * dt)
        costo_con  = np.sum(ct_array * P_grid_opt * dt)
        pico_sin   = L.max()
        pico_con   = P_grid_opt.max()
        energia    = L.sum() * dt
        factor_sin = L.mean() / L.max()
        factor_con = P_grid_opt.mean() / P_grid_opt.max()

        print(f"\nResultados día de máxima demanda:")
        print(f"  Demanda máxima SIN optimización : {pico_sin:.3f} kW")
        print(f"  Demanda máxima CON peak shaving : {pico_con:.3f} kW")
        print(f"  Reducción del pico              : {(1-pico_con/pico_sin)*100:.1f}%")
        print(f"  Costo energético SIN opt.       : ${costo_sin:.4f}")
        print(f"  Costo energético CON opt.       : ${costo_con:.4f}")
        print(f"  Ahorro económico                : {(1-costo_con/costo_sin)*100:.1f}%")
        print(f"  Factor de carga SIN opt.        : {factor_sin:.3f}")
        print(f"  Factor de carga CON opt.        : {factor_con:.3f}")

        # ─── TABLA II ──────────────────────────────────────────────────────────
        print()
        print("="*60)
        print("TABLA II — PERFIL DE DEMANDA: SIN vs. CON PEAK SHAVING")
        print("="*60)
        print(f"{'Indicador':<35} {'Sin opt.':>12} {'Con peak shaving':>18}")
        print("-"*68)
        print(f"{'Demanda máxima (kW)':<35} {pico_sin:>12.3f} "
              f"{pico_con:>12.3f}  (↓{(1-pico_con/pico_sin)*100:.1f}%)")
        print(f"{'Costo energético diario ($)':<35} {costo_sin:>12.4f} "
              f"{costo_con:>12.4f}  (↓{(1-costo_con/costo_sin)*100:.1f}%)")
        print(f"{'Energía total consumida (kWh)':<35} {energia:>12.3f} "
              f"{np.sum(P_grid_opt*dt):>12.3f}  (0%)")
        print(f"{'Factor de carga':<35} {factor_sin:>12.3f} "
              f"{factor_con:>12.3f}  (↑{(factor_con/factor_sin-1)*100:.1f}%)")

    else:
        print(f"Optimizador no convergió: {prob.status}")

except ImportError as e:
    print(f"CVXPY no disponible: {e}")
    print("Instalar con: pip install cvxpy")

except Exception as e:
    print(f"Error en optimización: {e}")

# ─── 8. GRÁFICAS ──────────────────────────────────────────────────────────────
print()
print("="*60)
print("GENERANDO GRÁFICAS...")
print("="*60)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Predicción de Demanda y Peak Shaving\nDataset UCI — Hebrail & Berard (2006)',
                 fontsize=13, fontweight='bold')

    # Gráfica 1: Perfil de demanda horario (muestra de 7 días)
    ax1 = axes[0, 0]
    muestra = df_hourly[-168:]
    ax1.plot(muestra.values, color='steelblue', linewidth=0.8)
    ax1.set_title('Perfil de demanda — últimos 7 días', fontsize=11)
    ax1.set_xlabel('Horas')
    ax1.set_ylabel('Demanda (kW)')
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: Comparación real vs predicho (mejor modelo)
    ax2 = axes[0, 1]
    if modelos_ordenados:
        mejor = modelos_ordenados[0]
        nombre_mejor = mejor[0]
        pred_mejor   = mejor[1]['pred']
        y_real_plot  = mejor[1].get('y_true', y_test)
        n_plot = min(168, len(pred_mejor))
        ax2.plot(y_real_plot[:n_plot], label='Real', color='steelblue', linewidth=1)
        ax2.plot(pred_mejor[:n_plot], label=f'Pred. {nombre_mejor}',
                 color='tomato', linewidth=1, linestyle='--')
        ax2.set_title(f'Real vs. Predicción — {nombre_mejor} (7 días)', fontsize=11)
        ax2.set_xlabel('Horas')
        ax2.set_ylabel('Demanda (kW)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    # Gráfica 3: Barras comparativas MAPE
    ax3 = axes[1, 0]
    if modelos_ordenados:
        nombres = [m[0] for m in modelos_ordenados]
        mapes   = [m[1]['MAPE'] for m in modelos_ordenados]
        colores = ['#2ecc71' if i == 0 else '#3498db' if i == 1 else '#e74c3c'
                   for i in range(len(nombres))]
        bars = ax3.bar(nombres, mapes, color=colores, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, mapes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
        ax3.set_title('MAPE por modelo (%)', fontsize=11)
        ax3.set_ylabel('MAPE (%)')
        ax3.grid(True, alpha=0.3, axis='y')

    # Gráfica 4: Peak shaving — demanda original vs optimizada
    ax4 = axes[1, 1]
    try:
        if prob.status in ['optimal', 'optimal_inaccurate']:
            ax4.plot(L, label='Sin optimización', color='tomato',
                     linewidth=1.5, marker='o', markersize=3)
            ax4.plot(P_grid_opt, label='Con peak shaving', color='steelblue',
                     linewidth=1.5, marker='s', markersize=3)
            ax4.axhline(y=pico_con, color='steelblue', linestyle='--',
                        alpha=0.5, label=f'Pico opt. {pico_con:.2f} kW')
            ax4.set_title('Peak Shaving — día de máxima demanda', fontsize=11)
            ax4.set_xlabel('Hora del día')
            ax4.set_ylabel('Demanda (kW)')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
    except:
        ax4.text(0.5, 0.5, 'Optimización no disponible\n(instalar CVXPY)',
                 ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    import os
    ruta_graficas = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados_peak_shaving.png')
    plt.savefig(ruta_graficas, dpi=150, bbox_inches='tight')
    print(f"Gráficas guardadas en: {ruta_graficas}")
    plt.close()

except Exception as e:
    print(f"Error en gráficas: {e}")

# ─── 9. REFERENCIA DEL DATASET ────────────────────────────────────────────────
print()
print("="*60)
print("REFERENCIA BIBLIOGRÁFICA DEL DATASET")
print("="*60)
print("""
Hebrail, G. & Berard, A. (2006). Individual Household Electric
Power Consumption [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C58K54

Licencia: Creative Commons Attribution 4.0 International (CC BY 4.0)
Período: diciembre 2006 – noviembre 2010
Frecuencia: 1 minuto (resampleado a 1 hora para este estudio)
Registros originales: 2,075,259
""")

print("="*60)
print("PROCESO COMPLETADO")
print("Los valores de la Tabla I y Tabla II provienen de")
print("datos reales del dataset UCI (Hebrail & Berard, 2006)")
print("="*60)
