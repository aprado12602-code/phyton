import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

archivo = "datos/Base_ventas_lacteos.xlsx"

df = pd.read_excel(
    archivo,
    sheet_name="Base de ventas"
)

# DIMENSIONES DE LA BASE

print("\n DIMENSIONES")

print(f"Cantidad de filas: {df.shape[0]}")
print(f"Cantidad de columnas: {df.shape[1]}")

# INFORMACIÓN Y TIPOS DE DATOS

print("\n INFORMACIÓN DE LAS COLUMNAS")

df.info()

print("\n TIPOS DE DATOS")

print(df.dtypes)


# VERIFICACION VACIOS

print("\n VALORES VACÍOS")

print(df.isnull().sum())

print(f"\n Total de valores vacíos: {df.isnull().sum().sum()}")


# DUPLICADOS

print("\n FILAS DUPLICADAS")

print(f"Total de filas duplicadas: {df.duplicated().sum()}")


# VARIABLE PRECIO POR KILOGRAMO

# Evitamos dividir entre cero.
df["Precio_Kg"] = (
    df["Ventas en soles"]
    .div(df["Ventas en kilogramos"].where(
        df["Ventas en kilogramos"] != 0
    ))
)

print("\n NUEVA VARIABLE: PRECIO POR KILOGRAMO")

print(
    df[
        [
            "Producto",
            "Ventas en soles",
            "Ventas en kilogramos",
            "Precio_Kg"
        ]
    ].head()
)



# DESCRIPCIÓN ESTADÍSTICA DEL PRECIO POR KILOGRAMO

print("\n ESTADÍSTICAS DEL PRECIO POR KILOGRAMO")

print(
    df[["Precio_Kg"]]
    .describe()
    .round(2)
)


# 2.2. GRÁFICOS DE COMPARACIÓN: PRECIO POR KG VS. VENTAS EN KG


plt.figure(figsize=(10, 6))

plt.scatter(
    df["Precio_Kg"],
    df["Ventas en kilogramos"],
    alpha=0.5
)

plt.title("Relación entre precio por kg y ventas en kg")
plt.xlabel("Precio por kilogramo (S/)")
plt.ylabel("Ventas en kilogramos")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 2.3. GRÁFICOS CON DATOS SEGMENTADOS VENTAS EN KG POR CATEGORÍA

ventas_categoria = (
    df.groupby("Categoría")["Ventas en kilogramos"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 6))

ventas_categoria.plot(
    kind="bar",
    width=0.75
)

plt.title("Ventas en kilogramos por categoría")
plt.xlabel("Categoría")
plt.ylabel("Ventas acumuladas en kilogramos")

plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# RESUMEN DE VENTAS POR CATEGORÍA
# ============================================================

print("\nVENTAS EN KILOGRAMOS POR CATEGORÍA")

print(
    ventas_categoria
    .round(2)
    .to_string()
)

print("\nPARTICIPACIÓN PORCENTUAL DE CADA CATEGORÍA")

participacion_categoria = (
    ventas_categoria / ventas_categoria.sum()
) * 100

print(
    participacion_categoria
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)


# ANÁLISIS DE ESTACIONALIDAD


# 1. Ventas totales por periodo
ventas_mensuales = (
    df.groupby("Periodo", as_index=False)
    .agg(Ventas_Kg=("Ventas en kilogramos", "sum"))
    .sort_values("Periodo")
)

# Convertir el periodo YYYYMM en una fecha
ventas_mensuales["Fecha"] = pd.to_datetime(
    ventas_mensuales["Periodo"].astype(str),
    format="%Y%m"
)

# Obtener el año y el mes
ventas_mensuales["Año"] = ventas_mensuales["Fecha"].dt.year
ventas_mensuales["Mes"] = ventas_mensuales["Fecha"].dt.month

# 2. Ventas promedio para cada mes del año
promedio_por_mes = (
    ventas_mensuales.groupby("Mes")["Ventas_Kg"]
    .mean()
)

# 3. Promedio general de los meses disponibles
promedio_general = ventas_mensuales["Ventas_Kg"].mean()

# 4. Calcular el índice de estacionalidad
indice_estacional = (
    promedio_por_mes / promedio_general
) * 100

# 5. Crear una tabla consolidada
nombres_meses = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre"
}

tabla_estacionalidad = pd.DataFrame({
    "Mes": range(1, 13),
    "Ventas promedio (kg)": [
        promedio_por_mes.get(mes, float("nan"))
        for mes in range(1, 13)
    ],
    "Índice de estacionalidad (%)": [
        indice_estacional.get(mes, float("nan"))
        for mes in range(1, 13)
    ]
})

tabla_estacionalidad["Mes"] = (
    tabla_estacionalidad["Mes"].map(nombres_meses)
)

# Agregar una interpretación
def interpretar_indice(indice):
    if pd.isna(indice):
        return "Sin datos"
    elif indice > 100:
        return "Demanda superior al promedio"
    elif indice < 100:
        return "Demanda inferior al promedio"
    else:
        return "Demanda igual al promedio"

tabla_estacionalidad["Interpretación"] = (
    tabla_estacionalidad["Índice de estacionalidad (%)"]
    .apply(interpretar_indice)
)

# 6. Mostrar la tabla
print("\n TABLA DE ESTACIONALIDAD")

print(
    tabla_estacionalidad.to_string(
        index=False,
        formatters={
            "Ventas promedio (kg)": lambda x: (
                f"{x:,.2f}" if pd.notna(x) else "Sin datos"
            ),
            "Índice de estacionalidad (%)": lambda x: (
                f"{x:.2f}%" if pd.notna(x) else "Sin datos"
            )
        }
    )
)

# 7. Graficar únicamente el índice de estacionalidad
plt.figure(figsize=(12, 6))

plt.plot(
    tabla_estacionalidad["Mes"],
    tabla_estacionalidad["Índice de estacionalidad (%)"],
    marker="o"
)

plt.axhline(
    y=100,
    linestyle="--",
    label="Promedio de referencia (100%)"
)

plt.title("Índice de estacionalidad de las ventas")
plt.xlabel("Mes")
plt.ylabel("Índice de estacionalidad (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 3.MODELO DE MACHINE LEARNING: FORECAST DE VENTAS

print("\n MACHINE LEARNING")

# 3.1.AGRUPAR LAS VENTAS POR MES, CANAL, CATEGORÍA Y PRODUCTO

ventas_ml = (
    df.groupby(
        ["Periodo", "Canal", "Categoría", "Producto"],
        as_index=False
    )
    .agg(
        Ventas_Kg=("Ventas en kilogramos", "sum"),
        Ventas_Soles=("Ventas en soles", "sum")
    )
)

# Convertir el periodo en fecha
ventas_ml["Fecha"] = pd.to_datetime(
    ventas_ml["Periodo"].astype(str),
    format="%Y%m"
)

# Obtener el mes
ventas_ml["Mes"] = ventas_ml["Fecha"].dt.month


# 3.2.CALCULAR EL PRECIO POR KILOGRAMO

ventas_ml["Precio_Kg"] = (
    ventas_ml["Ventas_Soles"] /
    ventas_ml["Ventas_Kg"].replace(0, np.nan)
)


# 3.3.INCORPORAR EL ÍNDICE DE ESTACIONALIDAD

ventas_ml["Indice_Estacional"] = (
    ventas_ml["Mes"].map(indice_estacional) / 100
)


# 3.4.CREAR LA VARIABLE DE VENTAS DEL MES ANTERIOR

grupo = ["Canal", "Categoría", "Producto"]

ventas_ml = ventas_ml.sort_values(
    ["Canal", "Categoría", "Producto", "Fecha"]
)

ventas_ml["Venta_Mes_Anterior"] = (
    ventas_ml.groupby(grupo)["Ventas_Kg"].shift(1)
)


# 3.5.PREPARAR LOS DATOS PARA EL MODELO

datos_ml = ventas_ml.dropna(
    subset=[
        "Precio_Kg",
        "Venta_Mes_Anterior",
        "Indice_Estacional"
    ]
).copy()

# Variables predictoras
X = datos_ml[
    [
        "Canal",
        "Categoría",
        "Producto",
        "Mes",
        "Precio_Kg",
        "Indice_Estacional",
        "Venta_Mes_Anterior"
    ]
]

# Convertir las variables categóricas en variables numéricas
X = pd.get_dummies(
    X,
    columns=["Canal", "Categoría", "Producto"],
    dtype=int
)

# Variable que queremos pronosticar
y = datos_ml["Ventas_Kg"]


# 3.6.SEPARAR LOS DATOS PARA ENTRENAR Y EVALUAR

# Reservar los últimos tres meses como prueba
ultima_fecha = datos_ml["Fecha"].max()

fecha_prueba = ultima_fecha - pd.DateOffset(months=2)

entrenamiento = datos_ml["Fecha"] < fecha_prueba
prueba = datos_ml["Fecha"] >= fecha_prueba

X_train = X.loc[entrenamiento]
X_test = X.loc[prueba]

y_train = y.loc[entrenamiento]
y_test = y.loc[prueba]

if X_train.empty or X_test.empty:
    raise ValueError(
        "No hay suficientes datos para entrenar y evaluar el modelo."
    )


# 3.7.ENTRENAR Y EVALUAR EL MODELO

modelo = LinearRegression()

# Entrenar usando el logaritmo de las ventas
modelo.fit(
    X_train,
    np.log1p(y_train)
)

# Predecir los meses de prueba
predicciones_log = modelo.predict(X_test)

# Convertir las predicciones a kilogramos
predicciones = np.expm1(predicciones_log)

# Evitar predicciones negativas
predicciones = np.maximum(predicciones, 0)

# Calcular el error absoluto medio
error = mean_absolute_error(
    y_test,
    predicciones
)

print("\n EVALUACIÓN DEL MODELO")
print(f"Error absoluto medio: {error:.2f} kg")
print(f"Registros de entrenamiento: {len(X_train)}")
print(f"Registros de prueba: {len(X_test)}")


# 3.8. ENTRENAR CON TODO EL HISTÓRICO DISPONIBLE

modelo.fit(
    X,
    np.log1p(y)
)



# 3.9. GENERAR LOS MESES FUTUROS

# Definir el año que queremos pronosticar
anio_pronostico = 2027

# Obtener la última fecha del histórico
ultima_fecha = ventas_ml["Fecha"].max()

# Generar los 12 meses de 2027
fechas_futuras = pd.date_range(
    start=f"{anio_pronostico}-01-01",
    periods=12,
    freq="MS"
)

print("\n========== PERIODOS A PRONOSTICAR ==========")

for fecha in fechas_futuras:
    print(fecha.strftime("%Y%m"))


# 3.10.PREPARAR LAS COMBINACIONES DE PRODUCTOS

combinaciones = (
    ventas_ml[grupo]
    .drop_duplicates()
    .reset_index(drop=True)
)

historico = ventas_ml.copy()

resultados = []


# 3.11.PRONOSTICAR CADA CANAL, CATEGORÍA Y PRODUCTO

for fecha in fechas_futuras:

    for _, combinacion in combinaciones.iterrows():

        canal = combinacion["Canal"]
        categoria = combinacion["Categoría"]
        producto = combinacion["Producto"]

        # Obtener el historial de la combinación
        serie = historico[
            (historico["Canal"] == canal) &
            (historico["Categoría"] == categoria) &
            (historico["Producto"] == producto)
        ].sort_values("Fecha")

        if serie.empty:
            continue

        # Venta del periodo anterior
        venta_anterior = serie.iloc[-1]["Ventas_Kg"]

        # Precio promedio de los últimos tres meses
        # que registraron ventas positivas
        precios = (
            serie.loc[
                serie["Ventas_Kg"] > 0,
                "Precio_Kg"
            ]
            .dropna()
            .tail(3)
        )

        if precios.empty:
            continue

        precio = precios.mean()

        mes = fecha.month

        # Obtener el índice estacional del mes
        indice = indice_estacional.get(mes, np.nan)

        # Si no hay índice calculado, usar el promedio
        if pd.isna(indice):
            indice = 100

        # Preparar las variables del nuevo periodo
        nuevo = pd.DataFrame([{
            "Canal": canal,
            "Categoría": categoria,
            "Producto": producto,
            "Mes": mes,
            "Precio_Kg": precio,
            "Indice_Estacional": indice / 100,
            "Venta_Mes_Anterior": venta_anterior
        }])

        # Convertir las categorías en variables numéricas
        nuevo = pd.get_dummies(
            nuevo,
            columns=["Canal", "Categoría", "Producto"],
            dtype=int
        )

        # Igualar las columnas a las utilizadas durante el entrenamiento
        nuevo = nuevo.reindex(
            columns=X.columns,
            fill_value=0
        )

        # Obtener el pronóstico en escala logarítmica
        pronostico_log = modelo.predict(nuevo)[0]

        # Convertir el resultado a kilogramos
        pronostico = max(
            0,
            np.expm1(pronostico_log)
        )

        # Guardar el resultado
        resultados.append({
            "Periodo": fecha.strftime("%Y%m"),
            "Canal": canal,
            "Categoría": categoria,
            "Producto": producto,
            "Precio_Kg": round(precio, 2),
            "Indice_Estacional": round(indice, 2),
            "Pronostico_Kg": round(pronostico, 2)
        })

        # Agregar la predicción al histórico para usarla
        # como referencia en el siguiente mes
        historico = pd.concat(
            [
                historico,
                pd.DataFrame([{
                    "Fecha": fecha,
                    "Periodo": int(fecha.strftime("%Y%m")),
                    "Canal": canal,
                    "Categoría": categoria,
                    "Producto": producto,
                    "Ventas_Kg": pronostico,
                    "Ventas_Soles": pronostico * precio,
                    "Precio_Kg": precio,
                    "Mes": mes,
                    "Indice_Estacional": indice / 100,
                    "Venta_Mes_Anterior": venta_anterior
                }])
            ],
            ignore_index=True
        )


# 3.12.GUARDAR Y MOSTRAR EL FORECAST

pronostico = pd.DataFrame(resultados)

if pronostico.empty:
    raise ValueError(
        "El forecast está vacío. Revisa los datos históricos y las variables."
    )

# Exportar a Excel
pronostico.to_excel(
    "Pronostico_Sell_In.xlsx",
    index=False
)


# 3.13.RESUMEN DEL FORECAST POR CANAL

print("\n PRONÓSTICO POR CANAL")

resumen_canal = (
    pronostico.groupby(
        ["Periodo", "Canal"],
        as_index=False
    )["Pronostico_Kg"]
    .sum()
)

print(
    resumen_canal.to_string(index=False)
)


# 3.14.RESUMEN DEL FORECAST POR CATEGORÍA

print("\n PRONÓSTICO POR CATEGORÍA")

resumen_categoria = (
    pronostico.groupby(
        ["Periodo", "Categoría", "Producto"],
        as_index=False
    )["Pronostico_Kg"]
    .sum()
)

print(
    resumen_categoria.to_string(index=False)
)


# 15. CONTROL DE RESULTADOS

print("\n RESULTDOS DEL FORECAST")

print(
    "Pronósticos iguales a cero:",
    (pronostico["Pronostico_Kg"] == 0).sum()
)

print(
    "Pronósticos menores a 1 kg:",
    (pronostico["Pronostico_Kg"] < 1).sum()
)

print(
    "Total de kg pronosticados:",
    round(pronostico["Pronostico_Kg"].sum(), 2)
)

print("\nArchivo generado: Pronostico_Sell_In.xlsx")