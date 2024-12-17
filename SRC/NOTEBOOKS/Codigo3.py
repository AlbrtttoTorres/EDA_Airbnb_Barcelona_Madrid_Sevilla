import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy.stats import ttest_ind

# Cargar datos
data = pd.read_csv(r'C:\Users\alber\OneDrive\Documentos\GitHub\DS_ONLINE_THEBRIDGE_ATC\Project_Break_EDA\Data\DataSetConjunto')

# 1. ANÁLISIS UNIVARIANTE
# ========================
# Descripción general de las variables
data_summary = data.describe(include='all')
print(data_summary)

# Estadísticos de centralidad
print("Moda de 'price':", data['price'].mode())  # Moda
print("Moda de 'room_type':", data['room_type'].mode())  # Moda

# Estadísticos de dispersión
print("Varianza de 'price':", data['price'].var())  # Varianza

# Distribución de precios
sns.histplot(data['price'], bins=50, kde=True)
plt.title("Distribución de precios")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")
plt.show()

# Diagrama de caja para ver dispersión
sns.boxplot(x='state', y='price', data=data)
plt.title("Distribución de precios por ciudad")
plt.xlabel("Ciudad")
plt.ylabel("Precio")
plt.show()

# Swarm plot (para distribución por categoría)
sns.swarmplot(x='state', y='price', data=data)
plt.title("Distribución de precios por ciudad (Swarm Plot)")
plt.xlabel("Ciudad")
plt.ylabel("Precio")
plt.show()

# Violin plot (para distribución por categoría)
sns.violinplot(x='state', y='price', data=data)
plt.title("Distribución de precios por ciudad (Violin Plot)")
plt.xlabel("Ciudad")
plt.ylabel("Precio")
plt.show()

# 2. ANÁLISIS BIVARIANTE
# =======================
# Relación entre precio y disponibilidad
sns.scatterplot(x='availability_365', y='price', data=data, hue='state', alpha=0.6)
plt.title("Relación entre disponibilidad y precio")
plt.xlabel("Días disponibles")
plt.ylabel("Precio")
plt.legend(title="Ciudad")
plt.show()

# Precios promedio por tipo de alojamiento
room_price = data.groupby(['state', 'room_type'])['price'].mean().reset_index()
sns.barplot(data=room_price, x='room_type', y='price', hue='state')
plt.title("Precio promedio por tipo de alojamiento y ciudad")
plt.xlabel("Tipo de alojamiento")
plt.ylabel("Precio promedio")
plt.legend(title="Ciudad")
plt.show()

# Relación entre precio y número de reseñas
sns.scatterplot(x='number_of_reviews', y='price', data=data, hue='state', alpha=0.6)
plt.title("Relación entre precio y número de reseñas")
plt.xlabel("Número de reseñas")
plt.ylabel("Precio")
plt.legend(title="Ciudad")
plt.show()

# 3. ANÁLISIS MULTIVARIANTE
# ==========================
# Heatmap de correlación
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap de Correlaciones')
plt.show()

# Pairplot (gráfico de relaciones entre todas las variables numéricas)
sns.pairplot(data[['price', 'availability_365', 'number_of_reviews', 'latitude_scaled', 'longitude_scaled']])
plt.show()

# Precios por mes y ciudad
price_by_month = data.groupby(['state', 'month'])['price'].mean().reset_index()
sns.lineplot(data=price_by_month, x='month', y='price', hue='state', marker='o')
plt.title("Precio promedio mensual por ciudad")
plt.xlabel("Mes")
plt.ylabel("Precio promedio")
plt.legend(title="Ciudad")
plt.show()

# Distribución de reseñas por barrio y ciudad
reviews_by_neighbourhood = data.groupby(['state', 'neighbourhood'])['number_of_reviews'].sum().reset_index()
reviews_top_neighbourhoods = reviews_by_neighbourhood.sort_values(by='number_of_reviews', ascending=False).groupby('state').head(10)
sns.barplot(data=reviews_top_neighbourhoods, y='neighbourhood', x='number_of_reviews', hue='state')
plt.title("Barrios con más reseñas por ciudad")
plt.xlabel("Número de reseñas")
plt.ylabel("Barrio")
plt.legend(title="Ciudad")
plt.show()

# Segmentación por temporadas
data['season'] = data['month'].apply(lambda x: 'Alta' if x in [6, 7, 8, 12] else 'Baja')
season_prices = data.groupby(['state', 'season'])['price'].mean().reset_index()
sns.barplot(data=season_prices, x='state', y='price', hue='season')
plt.title("Precio promedio por temporada y ciudad")
plt.xlabel("Ciudad")
plt.ylabel("Precio promedio")
plt.legend(title="Temporada")
plt.show()

# Clustering de barrios según comportamiento estacional
cluster_data = data.groupby(['neighbourhood', 'season'])['price'].mean().unstack().fillna(0)
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)
kmeans = KMeans(n_clusters=4, random_state=42).fit(cluster_scaled)
cluster_data['Cluster'] = kmeans.labels_

# Visualización de clusters
sns.scatterplot(x=cluster_scaled[:, 0], y=cluster_scaled[:, 1], hue=kmeans.labels_, palette='tab10')
plt.title("Clusters de barrios según estacionalidad")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend(title="Cluster")
plt.show()

# 4. ANÁLISIS DE IMPACTO DE UBICACIÓN (Regresión Múltiple)
# =======================================================
data['latitude_scaled'] = scaler.fit_transform(data[['latitude']])
data['longitude_scaled'] = scaler.fit_transform(data[['longitude']])
X = data[['latitude_scaled', 'longitude_scaled', 'availability_365']]
y = data['price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# 5. ANÁLISIS POR CIUDAD
# ========================
for city in ['Barcelona', 'Madrid', 'Sevilla']:
    city_data = data[data['state'] == city]
    
    # Comparación de precios entre temporadas
    high_season = city_data[city_data['season'] == 'Alta']['price']
    low_season = city_data[city_data['season'] == 'Baja']['price']
    t_stat, p_val = ttest_ind(high_season, low_season, equal_var=False)
    print(f"{city}: T-Test: t-stat={t_stat}, p-value={p_val}")

    # Evolución mensual
    monthly_prices = city_data.groupby('month')['price'].mean()
    plt.plot(monthly_prices.index, monthly_prices.values, marker='o')
    plt.title(f"Evolución mensual de precios en {city}")
    plt.xlabel("Mes")
    plt.ylabel("Precio promedio")
    plt.grid()
    plt.show()

# 6. VISUALIZACIÓN GEOGRÁFICA
# ===========================
# Crear mapas interactivos por ciudad
import folium
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
