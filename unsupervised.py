
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt


def elbow_rule(dataSet):
    """Effettua la regola del gomito per trovare il numero ottimale k di cluster"""

    inertia = []
    maxK=10

    for i in range(1, maxK):
        kmeans = KMeans(n_clusters=i,n_init=5,init='random')
        kmeans.fit(dataSet)
        inertia.append(kmeans.inertia_)

    # tramite KneeLocator troviamo il k ottimale
    kl = KneeLocator(range(1, maxK), inertia, curve="convex", direction="decreasing")
    
    # grafico della curva del gomito
    plt.plot(range(1, maxK), inertia, 'bx-')
    plt.scatter(kl.elbow, inertia[kl.elbow - 1], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Metodo del gomito per trovare il k ottimale')
    plt.legend()
    plt.savefig(f'plots/elbow_rule.png', dpi=300, bbox_inches='tight')
    plt.show()

    return kl.elbow


def clustering_kmeans(dataSet):
    """Effetua il clustering tramite l'algoritmo KMeans"""

    k = elbow_rule(dataSet)
    km = KMeans(n_clusters=k,n_init=10,init='random')
    km = km.fit(dataSet)

    validation(km,dataSet)

    etichette = km.labels_
    centroidi = km.cluster_centers_

    return etichette, centroidi



def validation(k_means, dataset):
    """Stampa le metriche di valutazione del clustering effettuato"""

    wcss = k_means.inertia_
    print("\nWCSS:", wcss)

    silhouette_avg = silhouette_score(dataset, k_means.labels_)
    print("Silhouette Score:", silhouette_avg)






