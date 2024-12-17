import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import folium
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

# Cargar los datos
file_path = r'C:\Users\alber\OneDrive\Documentos\GitHub\DS_ONLINE_THEBRIDGE_ATC\Project_Break_EDA\DataSetConjunto'
data = pd.read_csv(file_path)

# =============================
# Limpieza de datos
# =============================
# Identificar valores nulos
missing_values = data.isnull().sum()
print("Valores nulos por columna:\n", missing_values)

# Rellenar valores nulos en columnas importantes
data['price'] = data['price'].fillna(data['price'].median())
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

# Eliminar columnas innecesarias (si aplica)
data = data.drop(columns=['license'], errors='ignore')

# Crear columnas adicionales útiles:
data['month'] = pd.to_datetime(data['last_review'], errors='coerce').dt.month

# =============================
# Exploración inicial
# =============================
data.head()  # Verificar las primeras filas
data.info()  # Resumen de columnas y tipos de datos
data.describe()  # Descripción estadística general

# =============================
# Análisis exploratorio por hipótesis
# =============================
# Estacionalidad de precios
price_by_month = data.groupby(['state', 'month'])['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=price_by_month, x='month', y='price', hue='state', marker='o')
plt.title('Precio promedio por mes y ciudad')
plt.xlabel('Mes')
plt.ylabel('Precio promedio')
plt.legend(title='Ciudad')
plt.show()

# Comparación de precios entre ciudades
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='state', y='price')
plt.title('Distribución de precios por ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Precio')
plt.show()

# Tipo de alojamiento
room_type_stats = data.groupby(['state', 'room_type'])['price'].mean().reset_index()
sns.catplot(data=room_type_stats, x='room_type', y='price', hue='state', kind='bar', height=6, aspect=2)
plt.title('Precio promedio por tipo de alojamiento y ciudad')
plt.xlabel('Tipo de alojamiento')
plt.ylabel('Precio promedio')
plt.show()

# Popularidad de barrios
popular_neighbourhoods = data.groupby(['state', 'neighbourhood'])['number_of_reviews'].sum().reset_index()
popular_neighbourhoods = popular_neighbourhoods.sort_values(by='number_of_reviews', ascending=False).groupby('state').head(10)
sns.barplot(data=popular_neighbourhoods, y='neighbourhood', x='number_of_reviews', hue='state')
plt.title('Barrios más populares por ciudad (según reseñas)')
plt.xlabel('Número de reseñas')
plt.ylabel('Barrio')
plt.show()

# =============================
# Clustering de barrios según comportamiento estacional
# =============================
cluster_data = data.groupby(['neighbourhood', 'season'])['price'].mean().unstack().fillna(0)
print("Estructura de cluster_data:", cluster_data.shape)
print("Primeras filas de cluster_data:\n", cluster_data.head())

if cluster_data.shape[1] > 1:
    # Escalado de los datos
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42).fit(cluster_scaled)
    cluster_data['Cluster'] = kmeans.labels_

    # Visualización de clusters
    sns.scatterplot(x=cluster_scaled[:, 0], y=cluster_scaled[:, 1], hue=kmeans.labels_, palette='tab10')
    plt.title("Clusters de barrios según estacionalidad")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend(title="Cluster")
    plt.show()
else:
    print("No hay suficientes columnas para realizar el clustering.")

# Regresión múltiple para impacto de ubicación
data['latitude_scaled'] = StandardScaler().fit_transform(data[['latitude']])
data['longitude_scaled'] = StandardScaler().fit_transform(data[['longitude']])
X = data[['latitude_scaled', 'longitude_scaled', 'availability_365']]
y = data['price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Conecta a una base de datos SQLite
conn = sqlite3.connect(r'C:\Users\alber\OneDrive\Documentos\GitHub\DS_ONLINE_THEBRIDGE_ATC\Project_Break_EDA\DataSetConjunto') 

# Carga el DataFrame en una tabla llamada 'data'
data.to_sql('data', conn, if_exists='replace', index=False)
print("Datos cargados en la base de datos SQLite.")

# Función para ejecutar consultas SQL
def run_query(query, connection):
    return pd.read_sql(query, connection)

# =============================
# SQL: Consultas estructuradas
# =============================
query = '''
SELECT state, month, AVG(price) as avg_price
FROM data
GROUP BY state, month
ORDER BY month;
'''
df_resultado = run_query(query, conn)
print(df_resultado)

query = '''
SELECT state, AVG(price) as avg_price, COUNT(id) as total_listings
FROM data
GROUP BY state;
'''
df_resultado = run_query(query, conn)
print(df_resultado)

query = '''
SELECT state, room_type, AVG(price) as avg_price
FROM data
GROUP BY state, room_type;
'''
df_resultado = run_query(query, conn)
print(df_resultado)

query = '''
SELECT state, neighbourhood, SUM(number_of_reviews) as total_reviews
FROM data
GROUP BY state, neighbourhood
ORDER BY total_reviews DESC
LIMIT 10;
'''
df_resultado = run_query(query, conn)
print(df_resultado)

# Función de análisis por ciudad
def analyze_city(city_name):
    print(f"\n\n=== Análisis para {city_name} ===\n")
    
    # Filtrar datos de la ciudad
    city_data = data[data['state'] == city_name]
    
    # Distribución de precios por temporada
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=city_data, x='season', y='price', palette='coolwarm')
    plt.title(f'Distribución de precios por temporada en {city_name}')
    plt.xlabel('Temporada')
    plt.ylabel('Precio')
    plt.show()

    # Prueba estadística: t-test
    high_season_prices = city_data[city_data['season'] == 'Alta']['price']
    low_season_prices = city_data[city_data['season'] == 'Baja']['price']
    t_stat, p_value = ttest_ind(high_season_prices, low_season_prices, nan_policy='omit')
    print(f"T-test ({city_name}): t-stat = {t_stat:.2f}, p-value = {p_value:.5f}")
    
    # Mann-Whitney U
    u_stat, p_value_u = mannwhitneyu(high_season_prices, low_season_prices, alternative='two-sided')
    print(f"Mann-Whitney U ({city_name}): U-stat = {u_stat:.2f}, p-value = {p_value_u:.5f}")
    
    # Comparación por tipo de alojamiento
    room_season_prices = city_data.groupby(['season', 'room_type'])['price'].mean().reset_index()
    sns.catplot(data=room_season_prices, x='room_type', y='price', hue='season', kind='bar', height=6, aspect=2, palette='Set2')
    plt.title(f'Precio promedio por tipo de alojamiento y temporada en {city_name}')
    plt.xlabel('Tipo de alojamiento')
    plt.ylabel('Precio promedio')
    plt.show()

# Análisis para las tres ciudades
for city in ['Barcelona', 'Madrid', 'Sevilla']:
    analyze_city(city)

# Visualización geográfica para las ciudades
for city in ['Barcelona', 'Madrid', 'Sevilla']:
    city_data = data[data['state'] == city]
    city_map = folium.Map(location=[city_data['latitude'].mean(), city_data['longitude'].mean()], zoom_start=12)
    for _, row in city_data.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=2,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.5
        ).add_to(city_map)
    city_map.save(f'{city.lower()}_map.html')
    print(f"Mapa interactivo de {city} guardado como '{city.lower()}_map.html'.")
