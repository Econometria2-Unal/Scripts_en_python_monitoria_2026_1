# %% =========================
# Importación de librerías
# ==========================

import numpy as np  # Cálculo numérico y generación de datos aleatorios
import pandas as pd  # Manipulación de estructuras tipo DataFrame
import matplotlib.pyplot as plt  # Visualización básica
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX


# %% =========================
# Simulación de ruido blanco
# ==========================

# Fijamos semilla para garantizar reproducibilidad
np.random.seed(10)

# Número de observaciones
n = 10000

# Simulamos realizaciones de una variable aleatoria normal
# loc = Media = 0, scale= Desviación estándar = 1
yt_rb = np.random.normal(loc=0, scale=1, size=n)

yt_rb = pd.Series(yt_rb)


# %% =========================
# Gráfico de la serie simulada
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_rb)
plt.title("Simulación serie y_t ARIMA(0,0,0) - Ruido Blanco")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del proceso
# ==========================

lags = 20

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Función de Autocorrelación (FAC)
plot_acf(yt_rb, lags=lags, ax=ax1)
ax1.set_title("FAC - ARIMA(0,0,0)")

# Función de Autocorrelación Parcial (FACP)
plot_pacf(yt_rb, lags=lags, ax=ax2)
ax2.set_title("FACP - ARIMA(0,0,0)")

plt.tight_layout()
plt.show()


# %% =========================
# Simulación de un proceso AR(1) estacionario
# ==========================

np.random.seed(11)  # Fijar semilla para reproducibilidad

# En statsmodels, los coeficientes AR se pasan con signo invertido.
# Si el modelo es: y_t = 0.5 y_{t-1} + e_t
# Entonces debemos escribir: [1, -0.5]
ar_params = np.array([1, -0.5])

# No hay componente MA
ma_params = np.array([1])

# Crear objeto ARMA (en este caso AR(1))
ar1_process = ArmaProcess(ar_params, ma_params)

# Generar muestra de tamaño n
yt_ar1 = ar1_process.generate_sample(nsample=n)


# %% =========================
# Gráfico del proceso AR(1)
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_ar1)
plt.title("Simulación serie y_t ARIMA(1,0,0) - AR(1) estacionario")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del AR(1)
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_ar1, lags=lags, ax=ax1)
ax1.set_title("FAC - ARIMA(1,0,0)")

plot_pacf(yt_ar1, lags=lags, ax=ax2)
ax2.set_title("FACP - ARIMA(1,0,0)")

plt.tight_layout()
plt.show()


# %% =========================
# Simulación de un AR(1) no estacionario (ARIMA(1,1,0))
# ==========================

np.random.seed(13) 

# Modelo ARIMA(1,1,0):
# d = 1 implica que el proceso en niveles es no estacionario.
# A diferencia de ArmaProcess, aquí NO se invierte el signo
# de los coeficientes AR.
#
# Si el modelo es:
#     (1 - phi_1 L)(1 - L) y_t = e_t
# entonces simplemente pasamos phi_1 directamente.

modelo_ar1_ns = SARIMAX(endog=[0], order=(1, 1, 0), trend="n")

# params = [phi_1, sigma2]
# phi_1 = 0.5
# sigma2 = 1
yt_ar1_ns = modelo_ar1_ns.simulate(params=[0.5, 1], nsimulations=n)


# %% =========================
# Gráfico del AR(1) no estacionario
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_ar1_ns)
plt.title("Simulación serie y_t ARIMA(1,1,0) - No estacionario")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()
# %% =========================
# FAC y FACP del AR(1) no estacionario
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_ar1_ns, lags=lags, ax=ax1)
ax1.set_title("FAC - ARIMA(1,1,0)")

plot_pacf(yt_ar1_ns, lags=lags, ax=ax2)
ax2.set_title("FACP - ARIMA(1,1,0)")

plt.tight_layout()
plt.show()


# %% =========================
# Comparación: AR(1) estacionario vs no estacionario
# ==========================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(yt_ar1)
ax1.set_title("AR(1) estacionario")
ax1.set_xlabel("Tiempo")
ax1.set_ylabel("y_t")

ax2.plot(yt_ar1_ns)
ax2.set_title("AR(1) no estacionario (ARIMA(1,1,0))")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("y_t")

plt.tight_layout()
plt.show()


# %% =========================
# Diferenciación del proceso no estacionario
# ==========================

# np.diff calcula: y_t - y_{t-1}
diff_yt_ar1_ns = np.diff(yt_ar1_ns)

plt.figure(figsize=(12, 6))
plt.plot(diff_yt_ar1_ns)
plt.title("Primera diferencia del ARIMA(1,1,0)")
plt.xlabel("Tiempo")
plt.ylabel("Δy_t")
plt.show()


# %% =========================
# FAC y FACP de la serie diferenciada
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(diff_yt_ar1_ns, lags=lags, ax=ax1)
ax1.set_title("FAC - Serie diferenciada")

plot_pacf(diff_yt_ar1_ns, lags=lags, ax=ax2)
ax2.set_title("FACP - Serie diferenciada")

plt.tight_layout()
plt.show()

# %% =========================
# Simulación de un proceso AR(2) estacionario
# ==========================

np.random.seed(14)  

# Modelo teórico:
# y_t = 0.5 y_{t-1} - 0.25 y_{t-2} + e_t
#
# En ArmaProcess los coeficientes AR se pasan con signo invertido:
# [1, -phi_1, -phi_2]

ar_params = np.array([1, -0.5, 0.25])
ma_params = np.array([1])  # Sin componente MA

ar2_process = ArmaProcess(ar_params, ma_params)
yt_ar2 = ar2_process.generate_sample(nsample=n)


# %% =========================
# Gráfico del AR(2) estacionario
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_ar2)
plt.title("Simulación serie y_t ARIMA(2,0,0) - AR(2)")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del AR(2)
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_ar2, lags=lags, ax=ax1)
ax1.set_title("FAC - AR(2)")

plot_pacf(yt_ar2, lags=lags, ax=ax2)
ax2.set_title("FACP - AR(2)")

plt.tight_layout()
plt.show()


# %% =========================
# Simulación de AR(2) no estacionario (ARIMA(2,1,0))
# ==========================

np.random.seed(15)  # Reproducibilidad

# En SARIMAX NO se invierten signos.
# params = [phi_1, phi_2, sigma2]

modelo_ar2_ns = SARIMAX(endog=[0], order=(2, 1, 0), trend="n")

yt_ar2_ns = modelo_ar2_ns.simulate(
    params=[0.5, -0.25, 1],
    nsimulations=n,
)


# %% =========================
# Gráfico del AR(2) no estacionario
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_ar2_ns)
plt.title("Simulación serie y_t ARIMA(2,1,0) - No estacionario")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del AR(2) no estacionario
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_ar2_ns, lags=lags, ax=ax1)
ax1.set_title("FAC - ARIMA(2,1,0)")

plot_pacf(yt_ar2_ns, lags=lags, ax=ax2)
ax2.set_title("FACP - ARIMA(2,1,0)")

plt.tight_layout()
plt.show()

# %% =========================
# Diferenciación del AR(2) no estacionario
# ==========================

# Primera diferencia: Δy_t = y_t - y_{t-1}
diff_yt_ar2_ns = np.diff(yt_ar2_ns)

plt.figure(figsize=(12, 6))
plt.plot(diff_yt_ar2_ns)
plt.title("Primera diferencia del ARIMA(2,1,0)")
plt.xlabel("Tiempo")
plt.ylabel("diff_y_t")
plt.show()


# %% =========================
# FAC y FACP de la serie diferenciada
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(diff_yt_ar2_ns, lags=lags, ax=ax1)
ax1.set_title("FAC - Serie diferenciada")

plot_pacf(diff_yt_ar2_ns, lags=lags, ax=ax2)
ax2.set_title("FACP - Serie diferenciada")

plt.tight_layout()
plt.show()


# %% =========================
# Simulación de un proceso MA(1)
# ==========================

np.random.seed(16)  

# Modelo teórico:
# y_t = e_t + 0.5 e_{t-1}

# No hay componente AR
ar_params = np.array([1])

# En MA no se invierten signos en ArmaProcess.
# ma = [1, theta_1]
ma_params = np.array([1, 0.5])

ma1_process = ArmaProcess(ar_params, ma_params)
yt_ma1 = ma1_process.generate_sample(nsample=n)


# %% =========================
# Gráfico del MA(1)
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_ma1)
plt.title("Simulación serie y_t MA(1)")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del MA(1)
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_ma1, lags=lags, ax=ax1)
ax1.set_title("FAC - MA(1)")

plot_pacf(yt_ma1, lags=lags, ax=ax2)
ax2.set_title("FACP - MA(1)")

plt.tight_layout()
plt.show()


# %% =========================
# Simulación de un proceso MA(2)
# ==========================

np.random.seed(17)  # Reproducibilidad

# Modelo teórico:
# y_t = e_t + 0.5 e_{t-1} - 0.25 e_{t-2}

ar_params = np.array([1])  # Sin componente AR
ma_params = np.array([1, 0.5, -0.25])

ma2_process = ArmaProcess(ar_params, ma_params)
yt_ma2 = ma2_process.generate_sample(nsample=n)


# %% =========================
# Gráfico del MA(2)
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_ma2)
plt.title("Simulación serie y_t MA(2)")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del MA(2)
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_ma2, lags=lags, ax=ax1)
ax1.set_title("FAC - MA(2)")

plot_pacf(yt_ma2, lags=lags, ax=ax2)
ax2.set_title("FACP - MA(2)")

plt.tight_layout()
plt.show()


# %% =========================
# Simulación de un proceso ARMA(1,1) estacionario
# ==========================

np.random.seed(18)  

# Modelo teórico:
# y_t = 0.5 y_{t-1} + e_t + 0.5 e_{t-1}
#
# En ArmaProcess:
# - AR se pasa con signo invertido
# - MA se pasa sin invertir signo

ar_params = np.array([1, -0.5])
ma_params = np.array([1, 0.5])

arma11_process = ArmaProcess(ar_params, ma_params)
yt_arma11 = arma11_process.generate_sample(nsample=n)


# %% =========================
# Gráfico del ARMA(1,1)
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_arma11)
plt.title("Simulación serie y_t ARMA(1,1)")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del ARMA(1,1)
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_arma11, lags=lags, ax=ax1)
ax1.set_title("FAC - ARMA(1,1)")

plot_pacf(yt_arma11, lags=lags, ax=ax2)
ax2.set_title("FACP - ARMA(1,1)")

plt.tight_layout()
plt.show()


# En procesos ARMA(p,q) el análisis visual es más complejo:
# - La FAC no se corta abruptamente como en MA(q)
# - La FACP no se corta abruptamente como en AR(p)
# Ambas suelen decaer gradualmente.


# %% =========================
# Simulación de ARIMA(1,1,1) (ARMA (1,1) no estacionario en niveles)
# ==========================

np.random.seed(19)  

# En SARIMAX NO se invierten signos.
# Orden de params:
# [AR params, MA params, sigma2]

modelo_arma11_ns = SARIMAX(endog=[0], order=(1, 1, 1), trend="n")

yt_arma11_ns = modelo_arma11_ns.simulate(
    params=[0.5, 0.5, 1],
    nsimulations=n,
)


# %% =========================
# Gráfico del ARIMA(1,1,1)
# ==========================

plt.figure(figsize=(12, 6))
plt.plot(yt_arma11_ns)
plt.title("Simulación serie y_t ARIMA(1,1,1)")
plt.xlabel("Tiempo")
plt.ylabel("y_t")
plt.show()


# %% =========================
# FAC y FACP del ARIMA(1,1,1)
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(yt_arma11_ns, lags=lags, ax=ax1)
ax1.set_title("FAC - ARIMA(1,1,1)")

plot_pacf(yt_arma11_ns, lags=lags, ax=ax2)
ax2.set_title("FACP - ARIMA(1,1,1)")

plt.tight_layout()
plt.show()


# %% =========================
# Diferenciación del ARIMA(1,1,1)
# ==========================

# Primera diferencia: Δy_t = y_t - y_{t-1}
diff_yt_arma11_ns = np.diff(yt_arma11_ns)

plt.figure(figsize=(12, 6))
plt.plot(diff_yt_arma11_ns)
plt.title("Primera diferencia del ARIMA(1,1,1)")
plt.xlabel("Tiempo")
plt.ylabel("Δy_t")
plt.show()


# %% =========================
# FAC y FACP de la serie diferenciada
# ==========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(diff_yt_arma11_ns, lags=lags, ax=ax1)
ax1.set_title("FAC - Serie diferenciada")

plot_pacf(diff_yt_arma11_ns, lags=lags, ax=ax2)
ax2.set_title("FACP - Serie diferenciada")

plt.tight_layout()
plt.show()


# =========================
# FIN DEL CÓDIGO
# ==========================
# %%
