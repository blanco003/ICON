import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pgmpy.models import BayesianNetwork
import networkx as nx



def plot_learning_curves(model, X, y, differentialColumn, model_name, metodo):
    """Grafico delle curve di apprendimento, training e testing, dei modelli di apprendimento supervisionato"""

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy')

    # errori di training e testing
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # deviazione standard e la varianza degli errori 
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    print(
        f"\033[95m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

 
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name} : {metodo}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()

    plt.savefig(f'plots\curva_{model_name}_{metodo}.png', dpi=300, bbox_inches='tight')

    plt.show()





def plot_model_metrics(model_name, metric_values, metodo):
    """Grafico a barre che mostra le metriche medie del modello di apprendimento supervisionato,
         con accuracy, precision, recall e f1-score."""
    
    metric_mapping = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-score'
    }
    
    # Etichette e valori
    metric_labels = [metric_mapping[metric] for metric in metric_values.keys()]
    metric_vals = list(metric_values.values())

  
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metric_labels, metric_vals, color=['blue', 'green', 'red', 'purple'])


    plt.ylabel("Valore Medio")
    plt.title(f"Metriche Medie del {model_name}")
    plt.ylim(0, 1.1)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)

   
    for bar, value in zip(bars, metric_vals):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, min(height + 0.05, 1.05), f"{value:.3f}", 
                 ha='center', fontsize=12, color='black')

    plt.savefig(f'plots\metriche_{model_name}_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show()



def visualizza_cluster(dataSet, etichette, metodo):
    """ Visualizza i cluster, ottenuti dall'apprendimento non supervisionato, riducendo le dimensioni del dataset a 2D o 3D."""

    if metodo == "pca":
        reducer = PCA(n_components=2) 
    elif metodo == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)  
    else:
        raise ValueError("Metodo non valido. Usa 'pca' o 'tsne'.")

    dati_ridotti = reducer.fit_transform(dataSet)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(dati_ridotti[:, 0], dati_ridotti[:, 1], c=etichette, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Visualizzazione dei Cluster con {metodo.upper()}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")

    plt.savefig(f'plots/cluster_{metodo}.png', dpi=300, bbox_inches='tight')

    plt.show()


def visualize_bayesian_network(bayesianNetwork: BayesianNetwork):
    """Grafico della Rete Bayesiana"""

    G = nx.MultiDiGraph(bayesianNetwork.edges())

    pos = nx.spring_layout(G, iterations=100, k=2, threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=250, node_color="#ff574c")

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=8,
        arrowstyle="->",
        edge_color="blue",
        connectionstyle="arc3,rad=0.2",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.savefig(f'plots\\bayesian_network.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()



def plot_curves_nn(history,best_params,metodo):
    """Curve dell'accuracy e loss dell'apprendimento della rete neurale al variare delle epoche"""

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, best_params['epochs'] + 1)

    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()

   
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()

    plt.savefig(f'plots/curva_nn_{metodo}.png', dpi=300, bbox_inches='tight')

    plt.show()




def visualize_aspect_ratio(df, target, metodo):
    """Grafico a torta per la distribuzione della feature target"""

    # Conta le occorrenze per ciascun valore unico di target
    counts = df[target].value_counts()

    # Dai conteggi estraiamo le etichette
    labels = counts.index.tolist()

    # lunga lista di colori per evitare ripetizionici
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold', 'mediumorchid', 'lightsteelblue', 'lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue']

    # creiamo una figura e gli assi per disegnare il grafico, con dimensione 8x8 pollici
    fig, ax = plt.subplots(figsize=(10, 10))

    # configuriamo il grafico a torta
    wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    # aggiungiamo la leggenda
    ax.legend(labels, loc='lower right', fontsize='small')

    plt.title(f"Distribuzione {target}")  # aggiunge il titolo al grafico

    # salviamo il png
    plt.savefig(f'plots/distribuzione_{target}_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show() # mostra il grafico a schermo
