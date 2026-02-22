# %% Importar librerías
import pandas as pd #Trabaja todo lo de DataFrames
import matplotlib.pyplot as plt #Trabaja todo lo de gráficos
import seaborn as sns #Trabaja todo lo de gráficos, pero con un estilo más bonito (en este caso no lo usamos jaja)
from statsmodels.tsa.statespace.sarimax import SARIMAX # Estimar ARIMA(p,d,q), equivalente a arima.sim() en R
import numpy as np #Trabaja todo lo de cálculos numéricos, como generar números aleatorios, realizar operaciones matemáticas, etc.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Trabaja todo lo de gráficos de la FAC y la FACP).
from statsmodels.tsa.arima_process import ArmaProcess  # Trabaja todo lo de simular procesos ARMA ESTACIONARIOS

# %% Simular un proceso de ruido blanco
np.random.seed(10)  # Para reproducibilidad
n = 1000  # Número de observaciones 
yt_rb = np.random.normal(loc=0, scale=1, size=n) #loc es la media, scale es la desviación estándar

#Estamos simulando realizaciones de una variable aleatoria que sigue una distribución normal 
#con media 0 y desviación estándar 1.
yt_rb = pd.Series(yt_rb)
# %% Graficar el proceso de ruido blanco
plt.figure(figsize=(12, 6)) #Tamaño de la figura (ancho, alto) en pulgadas
plt.plot(yt_rb) #Graficar la serie de tiempo del proceso de ruido blanco
plt.title("Simulación serie y_t ARIMA (0,0,0) (Ruido Blanco)") #Título del gráfico
plt.show() #Mostrar el gráfico generado

# %% FAC y FACP del proceso de ruido blanco
lags = 20 

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,4)) #Crear una figura con 1 fila y 2 columnas de subgráficos, con un tamaño total de 12x4 pulgadas

plot_acf(yt_rb, lags=lags, ax=ax1) #Graficar la función de autocorrelación (FAC) del proceso de ruido blanco en el primer subgráfico (ax1), utilizando los primeros 20 rezagos (lags)
ax1.set_title("FAC del PGD ARIMA(0,0,0) (Ruido Blanco)") #Establecer el título del primer subgráfico (ax1) como "FAC del PGD ARIMA(0,0,0) (Ruido Blanco)"

plot_pacf(yt_rb, lags=lags, ax=ax2)
ax2.set_title("FACP del PGD ARIMA(0,0,0) (Ruido Blanco)")

plt.tight_layout() #Ajustar el diseño de los subgráficos para que no se superpongan y se vean bien en la figura
plt.show()
# %% Simular un proceso AR(1) estacionario
np.random.seed(11)  # Para reproducibilidad}

ar = np.array([1, -0.5])  # Coeficientes del proceso AR (signos invertidos para la función ArmaProcess)
ma = np.array([1])  # No hay componente MA
ar1 = ArmaProcess(ar, ma) #Crear un objeto ArmaProcess con los coeficientes AR y MA especificados. En este caso, el proceso AR(1) tiene un coeficiente de 0.5 y no tiene componente MA (coeficiente de 1).
yt_ar1 = ar1.generate_sample(nsample=n) #Generar una muestra de tamaño n (1000 observaciones) del proceso AR(1).

# %% Graficar el proceso AR(1)
plt.figure(figsize=(12, 6))
plt.plot(yt_ar1)
plt.title("Simulación serie y_t ARIMA (1,0,0) (AR(1))")
plt.show()

# %% FAC y FACP del proceso AR(1)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,4))
plot_acf(yt_ar1, lags=lags, ax=ax1)
ax1.set_title("FAC del PGD ARIMA(1,0,0) (AR(1))")
plot_pacf(yt_ar1, lags=lags, ax=ax2)
ax2.set_title("FACP del PGD ARIMA(1,0,0) (AR(1))")
plt.tight_layout()  
plt.show()

# %% Simulacion de un proceso AR(1) no estacionario
np.random.seed(13)  # Para reproducibilidad
  
modelo1 = SARIMAX(endog=[0], order=(1, 1, 0), trend='n') 
ytn1 = modelo1.simulate(params=[0.5, 1], nsimulations=n) 
#Usamos .simulate() y no .generate_sample() porque el proceso no es estacionario.
#El orden (1, 1, 0) indica que es un proceso AR(1) con una diferenciación (d=1) y sin componente MA (q=0).

plt.plot(ytn1)
plt.title("Simulación serie y_t ARIMA (1,1,0) (AR(1) no estacionario)")
plt.show()
# %% FAC y FACP del proceso AR(1) no estacionario
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,4))
plot_acf(ytn1, lags=lags, ax=ax1)
ax1.set_title("FAC del PGD ARIMA(1,1,0) (AR(1) no estacionario)")
plot_pacf(ytn1, lags=lags, ax=ax2)
ax2.set_title("FACP del PGD ARIMA(1,1,0) (AR(1) no estacionario)")
plt.tight_layout()
plt.show()

# %% Comparación de las gráficas de los procesos AR(1) estacionario y no estacionario
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(yt_ar1)
ax1.set_title("Proceso AR(1) estacionario")
ax2.plot(ytn1)
ax2.set_title("Proceso AR(1) no estacionario")
plt.tight_layout()
plt.show()

# %% Diferenciación del proceso AR(1) no estacionario
diff_ytn1 = np.diff(ytn1) 
plt.plot(diff_ytn1)
plt.title("Diferenciación del proceso AR(1) no estacionario")
plt.show()

# %% FAC y FACP de la diferenciación del proceso AR(1) no estacionario
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(diff_ytn1, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso diferenciado ARIMA(1,1,0)")
plot_pacf(diff_ytn1, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso diferenciado ARIMA(1,1,0)")
plt.tight_layout()
plt.show()

# %% Simulación de un proceso AR(2) estacionario
np.random.seed(14)  # Para reproducibilidad
ar = np.array([1, -0.5, 0.25])  # Coeficientes del proceso AR(2)
ma = np.array([1])  # No hay componente MA
ar2 = ArmaProcess(ar, ma)
yt_ar2 = ar2.generate_sample(nsample=n)

# %% Graficar el proceso AR(2)
plt.plot(yt_ar2)
plt.title("Simulación serie y_t ARIMA (2,0,0) (AR(2))")
plt.show()

# %% FAC y FACP del proceso AR(2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(yt_ar2, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso AR(2)")
plot_pacf(yt_ar2, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso AR(2)")
plt.tight_layout()
plt.show()

# %% Simulación de un proceso AR(2) no estacionario (ARIMA(2,1,0))
np.random.seed(15)  # Para reproducibilidad
modelo_ar2_no_est = SARIMAX(endog=[0], order=(2, 1, 0), trend='n')
ytn2 = modelo_ar2_no_est.simulate(params=[0.5, -0.25, 1], nsimulations=n) #No hay cambio de signos

# %% Graficar el proceso AR(2) no estacionario
plt.figure(figsize=(12, 6))
plt.plot(ytn2)
plt.title("Simulación serie y_t ARIMA (2,1,0) (AR(2) no estacionario)")
plt.show()

# %% FAC y FACP del proceso AR(2) no estacionario
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ytn2, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso AR(2) no estacionario")
plot_pacf(ytn2, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso AR(2) no estacionario")
plt.tight_layout()
plt.show()

# %% Diferenciación del proceso AR(2) no estacionario
diff_ytn2 = np.diff(ytn2)
plt.plot(diff_ytn2)
plt.title("Diferenciación del proceso AR(2) no estacionario")
plt.show()
#  AC y FACP de la diferenciación del proceso AR(2) no estacionario
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(diff_ytn2, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso diferenciado ARIMA(2,1,0)")
plot_pacf(diff_ytn2, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso diferenciado ARIMA(2,1,0)")
plt.tight_layout()
plt.show()

# %% Simulación de un proceso MA(1) 
np.random.seed(16)  # Para reproducibilidad
ar = np.array([1])  # No hay componente AR
ma = np.array([1, 0.5])  # Coeficientes del proceso MA(1)
ma1 = ArmaProcess(ar, ma)
yt_ma1 = ma1.generate_sample(nsample=n)

# %% Graficar el proceso MA(1)
plt.plot(yt_ma1)
plt.title("Simulación serie y_t MA(1)")
plt.show()

# FAC y FACP del proceso MA(1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(yt_ma1, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso MA(1)")
plot_pacf(yt_ma1, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso MA(1)")
plt.tight_layout()
plt.show()

# %% Simulación de un proceso MA(2)
np.random.seed(17)  # Para reproducibilidad
ar = np.array([1])  # No hay componente AR
ma = np.array([1, 0.5, -0.25])  # Coeficientes del proceso MA(2)
ma2 = ArmaProcess(ar, ma)
yt_ma2 = ma2.generate_sample(nsample=n)

# %% Graficar el proceso MA(2)
plt.plot(yt_ma2)
plt.title("Simulación serie y_t MA(2)")
plt.show()
# FAC y FACP del proceso MA(2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(yt_ma2, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso MA(2)")
plot_pacf(yt_ma2, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso MA(2)")
plt.tight_layout()
plt.show()

# %% Simulación de un proceso ARMA(1,1) estacionario
np.random.seed(18)  # Para reproducibilidad
ar = np.array([1, -0.5])  # Coeficientes del proceso AR(1)
ma = np.array([1, 0.5])  # Coeficientes del proceso MA(1)
arma11 = ArmaProcess(ar, ma)
yt_arma11 = arma11.generate_sample(nsample=n)

# %% Graficar el proceso ARMA(1,1)
plt.plot(yt_arma11)
plt.title("Simulación serie y_t ARMA(1,1)") 
plt.show()

# %%FAC y FACP del proceso ARMA(1,1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(yt_arma11, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso ARMA(1,1)")
plot_pacf(yt_arma11, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso ARMA(1,1)")
plt.tight_layout()
plt.show()

# Para un proceso ARMA(p,q) es más complejo el análisis de ACF y PACF. 
#Esto se debe a que a diferencia de una AR puro o un MA puro puede que las ACF 
# y la PAFC del ARMA(p,q) no se corten abruptamente como si pasa en los procesos "puros".

# %% Simulación de un ARMA(1,1) no estacionario (ARIMA(1,1,1))
np.random.seed(19)  # Para reproducibilidad
modelo_arma11_no_est = SARIMAX(endog=[0], order=(1, 1, 1), trend='n') 
ytn_arma11 = modelo_arma11_no_est.simulate(params=[0.5, 0.5, 1], nsimulations=n)
#Orden de los parametros en params sigue: [AR params, MA params, (seasonal AR), (seasonal MA), trend, variance]

# %% Graficar el proceso ARMA(1,1) no estacionario
plt.plot(ytn_arma11)
plt.title("Simulación serie y_t ARIMA (1,1,1) (ARMA(1,1) no estacionario)")
plt.show()

# %% FAC y FACP del proceso ARMA(1,1) no estacionario
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ytn_arma11, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso ARIMA(1,1,1)")
plot_pacf(ytn_arma11, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso ARIMA(1,1,1)")
plt.tight_layout()
plt.show()
# %% Diferenciación del proceso ARMA(1,1) no estacionario
diff_ytn_arma11 = np.diff(ytn_arma11)
plt.plot(diff_ytn_arma11)
plt.title("Diferenciación del proceso ARIMA(1,1,1)")
plt.show()

# %% FAC y FACP de la diferenciación del proceso ARMA(1,1) no estacionario
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(diff_ytn_arma11, lags=lags, ax=ax1)
ax1.set_title("FAC del proceso diferenciado ARIMA(1,1,1)")
plot_pacf(diff_ytn_arma11, lags=lags, ax=ax2)
ax2.set_title("FACP del proceso diferenciado ARIMA(1,1,1)")
plt.tight_layout()
plt.show()

# FIN DEL CÓDIGO
