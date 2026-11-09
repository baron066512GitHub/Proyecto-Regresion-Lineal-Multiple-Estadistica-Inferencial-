# Informe Técnico de Implementación: Ecosistema de Regresión Lineal Múltiple

**Asignatura:** Estadística Inferencial — 7mo Semestre  
**Institución:** Escuela de Computación, Facultad Experimental de Ciencias y Tecnología, Universidad de Carabobo  
**Entorno de Ejecución:** Python 3.12 (Jupyter Notebook) & Web Browser SPA (HTML5 / Vanilla JavaScript ES6+ / SVG / CSS3)  
**Fecha de Publicación:** 8 de noviembre de 2026  

---

## 1. Resumen Ejecutivo y Arquitectura del Sistema

El objetivo de este proyecto consistió en construir un ecosistema computacional integral de análisis, inferencia y optimización de modelos de **Regresión Lineal Múltiple (RLM)** basado en el método de **Mínimos Cuadrados Ordinarios (MCO)** y el algoritmo de selección de variables por **Eliminación Hacia Atrás (*Backward Elimination*)**.

El ecosistema está conformado por cuatro componentes acoplados y numéricamente auditados:

```
                                  ECOSISTEMA COMPUTACIONAL RLM
                                               │
             ┌─────────────────────────────────┼────────────────────────────────┐
             │                                 │                                │
    [telemetria_servidores.csv]   [Proyecto_Regresion_Lineal_Latencia]  [Modelo_Regresion_Lineal_Mejorado_Agustin]
       Dataset Oficial (E12-14)         Análisis Estadístico Completo        Metodología 8 Pasos (Prof. Agustín)
             │                                 │                                │
             └─────────────────────────────────┴────────────────────────────────┘
                                               │
                                               ▼
                         [Regresion_Lineal_Multiple_Mejorada_App.html]
                             Laboratorio Web Interactivo SPA
                       (Preacondicionamiento Matricial, SVG, Temas)
```

1. **`telemetria_servidores.csv`:** Dataset canónico con $n = 40$ observaciones y 7 variables numéricas extraído de la Tabla E12-14 del texto de referencia (*Montgomery & Runger: Applied Statistics and Probability for Engineers*).
2. **`Proyecto_Regresion_Lineal_Latencia.ipynb`:** Cuaderno interactivo de 39 celdas enfocado en análisis exploratorio de datos (EDA), contrastes ANOVA, diagnóstico multivariado de supuestos (Shapiro-Wilk, Breusch-Pagan, Durbin-Watson) y predicción.
3. **`Modelo_Regresion_Lineal_Mejorado_Agustin.ipynb`:** Cuaderno interactivo de 22 celdas estructurado bajo la metodología formal de 8 pasos del Prof. Agustín, con desarrollo matricial explícito de normal equations, cálculo de $\mathbf{C} = (\mathbf{X}'\mathbf{X})^{-1}$, tabla ANOVA, pruebas $t$ individuales, intervalos de confianza al 99% y eliminación de variables no significativas.
4. **`Regresion_Lineal_Multiple_Mejorada_App.html`:** Aplicación web de página única (SPA), reactiva y autocontenida, inspirada en la arquitectura de diseño de *Simulador de Distribuciones de Probabilidad*, que implementa preacondicionamiento diagonal simétrico, gráficos SVG de alta fidelidad, soporte de archivos CSV externos y diagnóstico de residuos.

---

## 2. Arquitectura de Datos (`telemetria_servidores.csv`)

### 2.1 Especificación de Variables

El dataset modela el rendimiento y la carga de servidores en un centro de datos para predecir la latencia del sistema:

| Símbolo | Nombre Técnico de la Columna | Rol en el Modelo | Tipo de Variable | Rango Muestral $[x_{\min}, x_{\max}]$ | Media Muestral ($\bar{x}$) | Desv. Estándar ($s$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| $y$ | `y_Latencia_ms` | **Dependiente (Respuesta)** | Continua | $[1246, 5764]$ ms | $2987.825$ ms | $1339.739$ ms |
| $x_1$ | `x1_Peticiones_Activas` | Regresora independiente | Discreta | $[1921, 3256]$ req | $2459.700$ req | $398.914$ req |
| $x_2$ | `x2_Hilos_Espera` | Regresora independiente | Discreta | $[1035, 1233]$ hilos | $1162.775$ hilos | $58.077$ hilos |
| $x_3$ | `x3_Memoria_RAM_MB` | Regresora independiente | Continua | $[20286, 32135]$ MB | $24545.950$ MB | $4087.683$ MB |
| $x_4$ | `x4_Consultas_SQL_QPS` | Regresora independiente | Continua | $[36, 159]$ QPS | $85.325$ QPS | $37.587$ QPS |
| $x_5$ | `x5_Carga_CPU_Porc` | Regresora independiente | Continua | $[1904, 8698]$ ($\times 10^2$) | $4584.975$ | $2208.571$ |
| $x_6$ | `x6_Ancho_Banda_Gbps` | Regresora independiente | Continua | $[48, 84]$ Gbps | $69.750$ Gbps | $11.751$ Gbps |

### 2.2 Auditoría y Limpieza Numérica
Se realizó un proceso de saneamiento contra la Tabla E12-14 oficial, corrigiendo 9 filas que presentaban desalineación de dígitos por errores comunes de transcripción en observaciones como la 12, 13, 20 y 31. Los estadísticos descriptivos se verificaron contra la literatura estándar, garantizando suma total de cuadrados corregida:

$$S_{yy} = \sum_{i=1}^{n} (y_i - \bar{y})^2 = 70\,001\,197.775$$

---

## 3. Arquitectura de los Cuadernos Jupyter

### 3.1 Cuaderno 1: `Proyecto_Regresion_Lineal_Latencia.ipynb`
Diseñado como un informe analítico completo con 39 celdas ejecutadas secuencialmente sin advertencias de ejecución.

* **Módulo 1 — Preparación y EDA (Celdas 1 a 8):** Carga desde archivo `.csv`, inspección de tipos, matriz de covarianza, matriz de correlación de Pearson y visualización de pares (*pairplot*). Detecta fuerte correlación positiva entre $y$ y las variables $x_3, x_4, x_5$ ($r > 0.98$) y correlación negativa con $x_6$ ($r = -0.983$).
* **Módulo 2 — Estimación MCO y ANOVA (Celdas 9 a 17):** Cálculo dual mediante `statsmodels.api.OLS` y operaciones matriciales con `numpy.linalg`. Descomposición de sumas de cuadrados ($SS_R = 69\,837\,302.50$, $SS_E = 163\,895.27$, $MS_E = 4\,966.52$). Coeficientes $R^2 = 0.997659$ y $R^2_{\text{adj}} = 0.997233$.
* **Módulo 3 — Multicolinealidad y Diagnóstico de Residuos (Celdas 18 a 28):** 
  - Cálculo del **Factor de Inflación de la Varianza (VIF)** centrado para las variables explicativas:

$$VIF_j = \frac{1}{1 - R_j^2}$$

    Se evidencian valores de VIF elevados en $x_3, x_4, x_5$ ($VIF > 10$), indicativos de colinealidad física (la carga de CPU, las consultas SQL y el consumo de RAM escalan conjuntamente con la demanda del servidor).
  - Pruebas estadísticas formales sobre los residuos $e_i$:
    - **Normalidad:** Test de Shapiro-Wilk ($W = 0.963$, $p = 0.208 > 0.05 \implies$ No se rechaza normalidad).
    - **Homocedasticidad:** Test de Breusch-Pagan ($LM = 8.12$, $p = 0.229 > 0.05 \implies$ Homocedasticidad confirmada).
    - **Independencia:** Test de Durbin-Watson ($DW = 1.84 \approx 2 \implies$ Ausencia de autocorrelación de primer orden).
* **Módulo 4 — Predicción e Intervalos (Celdas 29 a 39):** Evaluación de $\hat{y}_0$ con apalancamiento $h_0 = \mathbf{x}_0'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_0$, generando bandas de confianza simultáneas para el valor medio esperado e individual.

### 3.2 Cuaderno 2: `Modelo_Regresion_Lineal_Mejorado_Agustin.ipynb`
Estructurado estrictamente según la metodología pedagógica del Prof. Agustín (22 celdas). Se enfoca en resolver el problema del enunciado del proyecto: evaluar inicialmente el modelo con las variables solicitadas ($x_3, x_4, x_5$) y llevar a cabo la **Eliminación Hacia Atrás** con $\alpha = 0.01$.

* **Paso 0 — Modelo Inicial ($k = 3$: $x_3, x_4, x_5$):**
  - Ecuaciones normales: Matriz $(\mathbf{X}'\mathbf{X})$ de orden $4 \times 4$ y vector $(\mathbf{X}'\mathbf{y})$.
  - Inversión de matrices: Matriz $\mathbf{C} = (\mathbf{X}'\mathbf{X})^{-1}$ con diagonal principal:

$$C_{00} = 0.887259, \quad C_{33} = 2.4578 \times 10^{-7}, \quad C_{44} = 0.007621, \quad C_{55} = 2.1764 \times 10^{-6}$$

  - Parámetros estimados:

$$\hat{\beta}_0 = -1357.514, \quad \hat{\beta}_3 = -0.09633, \quad \hat{\beta}_4 = 23.364, \quad \hat{\beta}_5 = 0.70298$$

  - Varianza residual: $SS_E = 68\,638.165, gl_E = 36 \implies \hat{\sigma}^2 = MS_E = 1\,906.616$.
  - Prueba $F$ global en 8 pasos: $F_0 = 1724.416 > F_{0.01, 3, 36} = 4.377$ ($p < 10^{-4} \implies$ Modelo global significativo).
  - Pruebas $t$ individuales en 8 pasos:
    - $\beta_3$ ($x_3$ RAM): $SE(\hat{\beta}_3) = \sqrt{1906.616 \times 2.4578 \cdot 10^{-7}} = 0.04879 \implies T_0 = \frac{-0.09633}{0.04879} = -1.975$.
    - Con $\alpha = 0.01$ bilateral ($t_{0.005, 36} = 2.719$), $|T_0| = 1.975 \le 2.719$ y $p = 0.0560 \ge 0.01$.
    - **Decisión de la Cátedra:** Se concluye que $x_3$ no aporta significativamente al modelo en presencia de $x_4$ y $x_5$. Se elimina $x_3$.
* **Iteración 1 — Modelo Mejorado Óptimo ($k = 2$: $x_4, x_5$):**
  - Re-estimación matricial completa:

$$\hat{\boldsymbol{\beta}} = \begin{bmatrix} \hat{\beta}_0 \\ \hat{\beta}_4 \\ \hat{\beta}_5 \end{bmatrix} = \begin{bmatrix} -612.26694 \\ 19.82482 \\ 0.63977 \end{bmatrix}$$

  - Nueva varianza: $\hat{\sigma}^2 = MS_E = 2\,056.031$ ($gl_E = 37$).
  - Significancia individual:
    - $x_4$ (SQL QPS): $T_0 = 3.652 > 2.715$ ($p = 0.00078 < 0.01$).
    - $x_5$ (CPU %): $T_0 = 6.447 > 2.715$ ($p < 10^{-6} < 0.01$).
  - Ambos regresores son altamente significativos. Se alcanza el modelo óptimo parsimonioso:

$$\hat{y} = -612.26694 + 19.82482 \, x_4 + 0.63977 \, x_5$$

---

## 4. Arquitectura de la Aplicación Web (`Regresion_Lineal_Multiple_Mejorada_App.html`)

### 4.1 Principios de Ingeniería de Software
La aplicación fue construida como un artefacto web autosuficiente sin dependencias a librerías externas para lógica matemática o visual, asegurando que se ejecute en local de forma instantánea mediante protocolo `file:///`.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INTERFAZ DE USUARIO (DOM)                       │
├────────────────────────────────┬───────────────────────────────────────┤
│ • Barra de navegación (.tabs)  │ • Paneles de Datos e Ingesta CSV      │
│ • Diagrama de Flujo (.flow)    │ • Fórmulas y Metodología en 8 Pasos   │
├────────────────────────────────┴───────────────────────────────────────┤
│                    MOTOR MATEMÁTICO Y ESTADÍSTICO                      │
├────────────────────────────────┬───────────────────────────────────────┤
│ • Preacondicionamiento Matriz  │ • Inversión Gauss-Jordan con Pivoteo  │
│ • Residuos Centrados (Anti-NaN)│ • Funciones de Distribución (F, t, Z) │
├────────────────────────────────┴───────────────────────────────────────┤
│                     MOTOR DE RENDERIZADO VECTORIAL                     │
├────────────────────────────────┬───────────────────────────────────────┤
│ • Curva F (Zona de Rechazo)    │ • Curva Bilateral t-Student           │
│ • Q-Q Plot Normal (Acklam)     │ • Dispersión de Residuos vs ŷ y x_j   │
└────────────────────────────────┴───────────────────────────────────────┘
```

### 4.2 Desafío de Estabilidad Numérica y Solución Implementada
En implementaciones ingenuas, la inversión matricial utiliza Gauss-Jordan sin pivoteo ni escalamiento, calculando la suma residual como:

$$SS_E = \mathbf{y}'\mathbf{y} - \hat{\boldsymbol{\beta}}'\mathbf{X}'\mathbf{y}$$

Cuando las variables presentan órdenes de magnitud dispares ($x_3 \approx 30\,000 \implies (\mathbf{X}'\mathbf{X})_{3,3} \approx 2.4 \times 10^{10}$ mientras que $(\mathbf{X}'\mathbf{X})_{0,0} = 40$), la matriz $\mathbf{X}'\mathbf{X}$ tiene un número de condición elevado ($\kappa(\mathbf{X}'\mathbf{X}) > 10^{12}$). Esto puede provocar una cancelación por punto flotante IEEE 754.

#### Algoritmo de Preacondicionamiento Diagonal Simétrico
Para resolver esto de forma robusta, se implementó la transformación:

$$\mathbf{D} = \text{diag}\left( \frac{1}{\sqrt{|A_{00}|}}, \frac{1}{\sqrt{|A_{11}|}}, \dots, \frac{1}{\sqrt{|A_{pp}|}} \right)$$

$$\mathbf{A}_{\text{scaled}} = \mathbf{D} (\mathbf{X}'\mathbf{X}) \mathbf{D}$$

Donde la diagonal de $\mathbf{A}_{\text{scaled}}$ pasa a ser exactamente $1.0$, normalizando el rango dinámico de los elementos. Luego se aplica eliminación de Gauss-Jordan con pivoteo parcial sobre $\mathbf{A}_{\text{scaled}}$, y la inversa real se recupera mediante:

$$\mathbf{C} = (\mathbf{X}'\mathbf{X})^{-1} = \mathbf{D} \, \mathbf{A}_{\text{scaled}}^{-1} \, \mathbf{D}$$

Asimismo, la suma de cuadrados del error se calcula vectorialmente sumando directamente los residuos centrados:

$$SS_E = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Garantizando que $SS_E \ge 0$ en cualquier circunstancia.

### 4.3 Algoritmos de Funciones de Probabilidad Especiales
Para evaluar valores $p$ e inversas críticas sin bibliotecas externas, se codificaron:
1. **Log-Gamma de Lanczos:** Aproximación con 9 coeficientes para precisión de 15 dígitos significativos.
2. **Fracción Continua de la Función Beta Incompleta:** Para calcular la función de distribución acumulada (CDF) de la distribución $F$ y $t$-Student.
3. **Aproximación Racional de Acklam:** Función cuantil de la Normal Estándar con precisión minimax de $|e| < 1.15 \times 10^{-9}$ para generar el gráfico Q-Q plot de residuos.

---

## 5. Validación Numérica Cruzada (Python vs. JavaScript)

A continuación se muestra la correspondencia numérica exacta entre los cálculos de Python (`scipy` / `statsmodels`) y el motor de cálculo de la aplicación web sobre el dataset completo ($k = 6$):

| Métrica / Parámetro | Valor en Python (`statsmodels`) | Valor en App Web (`JavaScript`) | Error Relativo ($|\Delta| / |x|$) | Concordancia |
| :--- | :---: | :---: | :---: | :---: |
| $\hat{\beta}_0$ (Intercepto) | $-7241.9798$ | $-7241.9798$ | $< 10^{-12}$ | **Idéntico** |
| $\hat{\beta}_1$ (Peticiones) | $+1.4284$ | $+1.4284$ | $< 10^{-12}$ | **Idéntico** |
| $\hat{\beta}_2$ (Hilos Espera) | $-0.7719$ | $-0.7719$ | $< 10^{-12}$ | **Idéntico** |
| $\hat{\beta}_3$ (RAM MB) | $+0.3644$ | $+0.3644$ | $< 10^{-12}$ | **Idéntico** |
| $\hat{\beta}_4$ (SQL QPS) | $+4.8398$ | $+4.8398$ | $< 10^{-12}$ | **Idéntico** |
| $\hat{\beta}_5$ (CPU %) | $+0.1466$ | $+0.1466$ | $< 10^{-12}$ | **Idéntico** |
| $\hat{\beta}_6$ (Ancho Banda) | $-18.6653$ | $-18.6653$ | $< 10^{-12}$ | **Idéntico** |
| $SS_E$ (Suma Cuadrados Error) | $23\,189.6385$ | $23\,189.6385$ | $< 10^{-10}$ | **Idéntico** |
| $\hat{\sigma}^2 = MS_E$ ($gl_E=33$) | $702.7163$ | $702.7163$ | $< 10^{-10}$ | **Idéntico** |
| Estadístico $F_0$ Global | $2350.1297$ | $2350.1297$ | $< 10^{-10}$ | **Idéntico** |
| $R^2$ Coef. Determinación | $0.997665$ | $0.997665$ | $< 10^{-10}$ | **Idéntico** |
| $R^2_{\text{adj}}$ Coef. Ajustado | $0.997241$ | $0.997241$ | $< 10^{-10}$ | **Idéntico** |
| Variable eliminada (Paso 0) | $x_2$ ($p = 0.4355$) | $x_2$ ($p = 0.4355$) | — | **Idéntico** |

---

## 6. Conclusiones y Guía de Uso del Ecosistema

1. **Rigor Teórico y Criterio Estadístico:** Los desarrollos cumplen a cabalidad con los 8 pasos reglamentarios exigidos por la cátedra de Estadística Inferencial, demostrando cómo la significancia individual determina la parsimonia del modelo.
2. **Independencia de Ejecución:** La aplicación web permite al estudiante o profesor cargar cualquier archivo `.csv`, seleccionar interactivamente la variable dependiente y evaluar paso a paso cómo la eliminación de variables mejora el cuadrado medio del error $MS_E$ y maximiza el $R^2_{\text{adj}}$.
3. **Verificabilidad Abierta:** Los cuadernos `.ipynb` y el archivo `.html` se encuentran listos para inspección, defensa de proyecto o ejecución docente en cualquier navegador o entorno Jupyter.