import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    "nombre": ["Juan", "Pedro", "Maria", "Isabel", "Diego", "Luis", "Lucia", "Francisca", "Alejandro", "Fernando"],
    "edad": [19, 51, 33, 30, 23, 26, 45, 43, 38, 60],
    "montoConsumo": [971, 271, 614, 614, 585, 898, 310, 884, 979, 189]
}

df = pd.DataFrame(data)

X = df[["edad", "montoConsumo"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método del codo para determinar el mejor número de clústers
inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.show()

# Aplicar KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(df)

# Visualización
plt.figure(figsize=(8, 5))
for cluster in range(3):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['edad'], cluster_data['montoConsumo'], label=f'Cluster {cluster}')

plt.xlabel('Edad')
plt.ylabel('Monto de Consumo')
plt.title('Clusters KMeans')
plt.legend()
plt.show()
