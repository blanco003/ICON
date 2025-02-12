import pickle
import time

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BayesianEstimator, BicScore
from pgmpy.inference import VariableElimination
from pgmpy.metrics import correlation_score, log_likelihood_score
from pgmpy.models import BayesianNetwork
from sklearn.metrics import balanced_accuracy_score

from sklearn.preprocessing import LabelEncoder

from plot import visualize_bayesian_network


def create_bayesian_network(df):
    """Apprendimento della struttura della Rete Bayesiana con ricerca della struttura ottimale"""

    hc_k2 = HillClimbSearch(df)
    k2_model = hc_k2.estimate(scoring_method='k2score', max_iter=100)

    
    model = BayesianNetwork(k2_model.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)

    """"
    # se volessimo salvare la rete bayesiana su file in modo da poterla caricare direttamente senza apprendimento

    with open('models\\bn_model.pkl', 'wb') as output:
        pickle.dump(model, output)
    """

    # visualizzazione grafo rete bayesiana
    visualize_bayesian_network(model)

    # visualizzazione di ogni cpd delle variabili contenute nella rete
    visualize_info(model)

    return model



def query_report(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    """Effettua inferenza sulla rete bayesiana sulla base delle evidenze"""

    print(f"Query : {desc}")

    print("Risposta : ")
    print(infer.query(variables=variables,
                      evidence=evidence,
                      elimination_order=elimination_order,
                      show_progress=show_progress))
    



def generate_random_example(bayesianNetwork: BayesianNetwork, ):
    """Genera un esempio randomico a partire dalla rete bayesiana appresa"""

    return bayesianNetwork.simulate(n_samples=1)


def predict(bayesianNetwork: BayesianNetwork, example, target):
    """Effetua la predizione del target"""

    inference = VariableElimination(bayesianNetwork)
    result = inference.query(variables=[target], evidence=example, elimination_order='MinFill')
    print(result)

    
def visualize_info(bayesianNetwork: BayesianNetwork):
    """Stampa delle cpd di ogni vairabile nella rete"""

    for cpd in bayesianNetwork.get_cpds():
        print(f'CPD of {cpd.variable}:')
        print(cpd, '\n')


"""
# se volessimo leggere la rete bayesiana da file senza doverla apprendere

def loadBayesianNetwork():

    with open('models\\bn_model.pkl', 'rb') as input:
        model = pickle.load(input)
    
    return model
"""