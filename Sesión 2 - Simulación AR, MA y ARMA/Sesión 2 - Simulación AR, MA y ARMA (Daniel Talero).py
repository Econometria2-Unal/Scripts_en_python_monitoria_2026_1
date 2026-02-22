#=====================================================================================#
#                           UNIVERSIDAD NACIONAL DE COLOMBIA
#                   Facultad de Ciencias Económicas | 2026 - 01
#                            Econometría II | Monitoría 
#
#                                     Sesión 2:
#                              Simulación AR, MA y ARMA
#=====================================================================================#

#%%========= Importación de librerías==============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#Configuración para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

#%%=========== Comentarios sobre el manejo de series de tiempo==========
#En python uno puede trabajar series de tiempo con 3 tipos de objetos:
###1. Series de Pandas: Permiten indexar por fechas, realizar operaciones de ventana, etc.
###2. DataFrames de Pandas: Útiles para manejar múltiples series de tiempo en un mismo objeto, facilitando análisis comparativos.
###3. Arrays de NumPy: Más eficientes para cálculos numéricos
# Número de observaciones para todas las series
obs = 1000
# Existe un soporte teórico que justifica el uso de un amplio número de observaciones
# Reducción del error muestral y mayor precisión de los estimadores (betas consistente)
#Teorema del limite central: A medida que aumentan los datos, estos se pueden a distribuir de forma normal

#%% ==========Características de las simulaciones ============
# En la práctica, es muy común que los errores de varias series de tiempo no sean
# normales y por el contrario tengan una distribución diferente a la normal. 
# En caso tal de que los errores no distribuyan de manera normal, va a ser necesario
# emplear métodos de simulación (como bootstrapping). sin embargo, para fines didácticos, se asume que los errores son normales.

#%%======== Sobre el análisis gráfico de las series de tiempo ========
# El análisis gráfico es fundamental para identificar patrones, tendencias, estacionalidades y posibles anomalías en los datos.
#  Sin embargo, es importante complementar este análisis con pruebas formales de estacionariedad (
# como la prueba de Dickey-Fuller) y otras técnicas estadísticas para confirmar las conclusiones obtenidas visualmente.    

#---- Simulación de un proceso de ruido blanco ---- ARIMA(0,0,0)
print("1. PROCESO DE RUIDO BLANCO - ARIMA(0,0,0)")
np.random.seed(1)
innov = np.random.normal(0, 1, obs)
yt_rb = pd.Series(innov, index=pd.date_range('2000-01-01', periods=obs, freq='D'))
# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
yt_rb.plot(ax=axes[0,0], title='Serie y_t ARIMA(0,0,0) - Ruido Blanco', 
           color='lightblue', linewidth=0.5)
plot_acf(yt_rb, lags=20, ax=axes[0,1], title='FAC ARIMA(0,0,0)')
plot_pacf(yt_rb, lags=20, ax=axes[1,0], title='FACP ARIMA(0,0,0)')
plt.tight_layout()
plt.show()

#%% Simulación de un proceso AR(1) estacionario ARIMA(1,0,0)
print("2. PROCESO AR(1) ESTACIONARIO - ARIMA(1,0,0)")
np.random.seed(1)
ar1 = np.array([1, -0.6])  # φ₁=0.6 (estacionariedad)
ma1 = np.array([1])
arma_process1s = ArmaProcess(ar1, ma1)
yt1s = arma_process1s.generate_sample(nsample=obs)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pd.Series(yt1s).plot(ax=axes[0], title='Serie AR(1) Estacionario φ=0.6', 
                     color='turquoise', linewidth=0.5)
plot_acf(yt1s, lags=20, ax=axes[1], title='ACF AR(1)')
plot_pacf(yt1s, lags=20, ax=axes[2], title='PACF AR(1)')
plt.tight_layout()
plt.show()

#%% Simulación de un proceso AR(1) no estacionario ---- ARIMA(1,1,0)
print("3. PROCESO AR(1) NO ESTACIONARIO - ARIMA(1,1,0)")
np.random.seed(1)
ar1n = np.array([1, -0.5])
ma1n = np.array([1])
arma_process1n = ArmaProcess(ar1n, ma1n)
yt1n = arma_process1n.generate_sample(nsample=obs).cumsum()  # Integración d=1

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
pd.Series(yt1n).plot(ax=axes[0,0], title='Serie ARIMA(1,1,0) - No Estacionaria', 
                     color='darkblue', linewidth=0.5)
plot_acf(yt1n, lags=20, ax=axes[0,1], title='ACF ARIMA(1,1,0)')
plot_pacf(yt1n, lags=20, ax=axes[1,0], title='PACF ARIMA(1,1,0)')

# Diferenciación
diff_yt1n = pd.Series(np.diff(yt1n))
diff_yt1n.plot(ax=axes[1,1], title='1ra Diferencia - Estacionaria', 
               color='turquoise', linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Simulación de un proceso AR(2) estacionario ARIMA(2,0,0)
print("4. PROCESO AR(2) ESTACIONARIO - ARIMA(2,0,0)")
np.random.seed(2929)
ar2s = np.array([1, -0.6, -0.3])  # φ₁=0.6, φ₂=0.3
ma2s = np.array([1])
arma_process2s = ArmaProcess(ar2s, ma2s)
yt2s = arma_process2s.generate_sample(nsample=obs)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pd.Series(yt2s).plot(ax=axes[0], title='Serie AR(2) Estacionario', color='green', linewidth=0.5)
plot_acf(yt2s, lags=20, ax=axes[1], title='ACF AR(2)')
plot_pacf(yt2s, lags=20, ax=axes[2], title='PACF AR(2)')
plt.tight_layout()
plt.show()

#%% Simulación AR(2) no estacionario ---- ARIMA(2,1,0)
print("5. PROCESO AR(2) NO ESTACIONARIO - ARIMA(2,1,0)")
np.random.seed(202)
yt2n = arma_process2s.generate_sample(nsample=obs).cumsum()  # d=1

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
pd.Series(yt2n).plot(ax=axes[0,0], title='Serie ARIMA(2,1,0)', color='green', linewidth=0.8)
plot_acf(yt2n, lags=20, ax=axes[0,1], title='ACF ARIMA(2,1,0)')

diff_yt2n = pd.Series(np.diff(yt2n))
diff_yt2n.plot(ax=axes[1,1], title='1ra Diferencia ARIMA(2,1,0)', color='turquoise')
plot_acf(diff_yt2n, lags=20, ax=axes[1,0], title='ACF 1ra Diferencia')
plt.tight_layout()
plt.show()

#%% Simulación MA(1)  ARIMA(0,0,1)
print("6. PROCESO MA(1) - ARIMA(0,0,1)")
np.random.seed(2020)
ar_ma1 = np.array([1])
ma_ma1 = np.array([1, 0.7])  # θ₁=0.7
arma_process_ma1 = ArmaProcess(ar_ma1, ma_ma1)
yt3s = arma_process_ma1.generate_sample(nsample=obs)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pd.Series(yt3s).plot(ax=axes[0], title='Serie MA(1) θ=0.7', color='blue', linewidth=0.8)
plot_acf(yt3s, lags=20, ax=axes[1], title='ACF MA(1)')
plot_pacf(yt3s, lags=20, ax=axes[2], title='PACF MA(1)')
plt.tight_layout()
plt.show()

#%% Simulación MA(2) ARIMA(0,0,2)
print("7. PROCESO MA(2) - ARIMA(0,0,2)")
np.random.seed(10181)
ma_ma2 = np.array([1, 0.7, 0.25])  # θ₁=0.7, θ₂=0.25
arma_process_ma2 = ArmaProcess(ar_ma1, ma_ma2)
yt4s = arma_process_ma2.generate_sample(nsample=obs)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pd.Series(yt4s).plot(ax=axes[0], title='Serie MA(2)', color='turquoise', linewidth=0.5)
plot_acf(yt4s, lags=20, ax=axes[1], title='ACF MA(2)')
plot_pacf(yt4s, lags=20, ax=axes[2], title='PACF MA(2)')
plt.tight_layout()
plt.show()

#---- Simulación ARMA(1,1) estacionario ---- ARIMA(1,0,1)
print("8. PROCESO ARMA(1,1) ESTACIONARIO - ARIMA(1,0,1)")
np.random.seed(220422)
ar_arma = np.array([1, -0.7])  # φ₁=0.7
ma_arma = np.array([1, 0.3])   # θ₁=0.3
arma_process_arma = ArmaProcess(ar_arma, ma_arma)
yt5s = arma_process_arma.generate_sample(nsample=obs)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pd.Series(yt5s).plot(ax=axes[0], title='Serie ARMA(1,1)', color='orange', linewidth=0.7)
plot_acf(yt5s, lags=20, ax=axes[1], title='ACF ARMA(1,1)')
plot_pacf(yt5s, lags=20, ax=axes[2], title='PACF ARMA(1,1)')
plt.tight_layout()
plt.show()

#%% Simulación ARMA(1,1) no estacionario ARIMA(1,1,1)
print("9. PROCESO ARMA(1,1) NO ESTACIONARIO - ARIMA(1,1,1)")
np.random.seed(81711)
arma_process_arma_n = ArmaProcess(np.array([1, -0.6]), np.array([1, 0.7]))
yt5n = arma_process_arma_n.generate_sample(nsample=obs).cumsum()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
pd.Series(yt5n).plot(ax=axes[0,0], title='Serie ARIMA(1,1,1)', color='red', linewidth=0.7)
plot_acf(yt5n, lags=20, ax=axes[0,1], title='ACF ARIMA(1,1,1)')

# Diferenciación
dif_yt5n = pd.Series(np.diff(yt5n))
dif_yt5n.plot(ax=axes[1,1], title='1ra Diferencia ARIMA(1,1,1)', color='red', linewidth=0.7)
plot_acf(dif_yt5n, lags=20, ax=axes[1,0], title='ACF 1ra Diferencia')
plt.tight_layout()
plt.show()

#%%  conclusiones finales
print("\n=== RESUMEN DE PATRONES IDENTIFICADOS ===")
print("""
• AR(p) estacionario: ACF cae geométricamente, PACF se corta después de p rezagos
• MA(q) estacionario: PACF cae geométricamente, ACF se corta después de q rezagos  
• No estacionario (d=1): ACF cae lentamente (lineal), alta persistencia
• Diferenciación: Transforma I(1) → I(0) estacionario
• ARMA(p,q): Patrones mixtos, más complejos de identificar
• SIEMPRE complementar gráficos con pruebas formales (Dickey-Fuller, etc.)
""")


# %%
