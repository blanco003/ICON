from data_processing import *

from plot import visualizza_cluster, visualize_bayesian_network
from unsupervised import clustering_kmeans
from supervised import train_model_kfold_lg, train_model_kfold_tree_based
from pgmpy.inference import VariableElimination
from bayesian_network import create_bayesian_network, generate_random_example, predict, query_report
from neural_network import train_neural_network


"""
     N.B : le varie celle sottostanti sono state scritte 
           in modo da essere eseguite in maniera indipendente 
"""


############################################## APPRENDIMENTO SUPERVISIONATO  1 #########################################################

"""
df = pre_processing()
target = 'diabetes'

model_tree_based = train_model_kfold_tree_based(df,target,"Originale")

df = normalize_dataset(df,target)

model_lg = train_model_kfold_lg(df,target,"Originale")
"""


############################################## APPRENDIMENTO SUPERVISIONATO  2 - OVERSAMPLING #########################################################

"""
df = pre_processing()
target = 'diabetes'

df = over_sampling(df,target)

model_tree_based = train_model_kfold_tree_based(df,target,"OverSampling")

df = normalize_dataset(df,target)

model_lg = train_model_kfold_lg(df,target,"OverSampling")
"""

############################################## APPRENDIMENTO SUPERVISIONATO  3 - OVERSAMPLING, UNDERSAMPLING #########################################################

"""
df = pre_processing()
target = 'diabetes'

df = balanced_sampling(df,target)

model_tree_based = train_model_kfold_tree_based(df,target,"Mixed")

df = normalize_dataset(df,target)

model_lg = train_model_kfold_lg(df,target,"Mixed")
"""

############################################## APPRENDIMENTO NON SUPERVISIONATO #########################################################ù

"""

df = pre_processing()

target = 'diabetes'

df_unsupervised = prepare_dataset_for_unsupervised(df,target)

etichette, centroidi = clustering_kmeans(df_unsupervised)

visualizza_cluster(df_unsupervised, etichette, "pca")
visualizza_cluster(df_unsupervised, etichette, "tsne")

"""


############################################## RETE BAYESIANA #########################################################

"""
df = pre_processing()

df = discretize_dataset(df)

bayesian_network = create_bayesian_network(df)


# ESEMPIO GENERAZIONE RANDOMICA E PREDIZIONE
target = "diabetes"

example = generate_random_example(bayesian_network)

print("Esempio randomico generato :")
print(example.T)


# Rimozione di una feature dall'esempio e predizione del valore
del(example['HbA1c_level'])  

print("Esempio randomico senza HbA1c_level:")
print(example.T)

predict(bayesian_network, example.to_dict('records')[0], 'HbA1c_level')


# ESEMPIO QUERY 

# Creazione dell'oggetto inferenza
infer = VariableElimination(bayesian_network)

# Eseguire la query con evidenza
query_report(infer,
            variables=['blood_glucose_level'],
            evidence={ 'age': 4.0, 'bmi': 3.0, 'heart_disease': 0.0, 'HbA1c_level': 2.0 },
            desc = "\nData l'osservazione che un paziente ha un'età discretizzata nel valore massimo (4.0), " +
                   "un indice di massa corporea (BMI) discretizzato a livello alto (3.0), " +
                   "non ha malattie cardiache (0.0), " +
                   "e un livello di HbA1c discretizzato a un valore medio-alto (2.0), " +
                   "qual è la distribuzione di probabilità del livello di glucosio nel sangue del paziente?"
)

"""
############################################## RETE NEURALE #########################################################


"""
df = pre_processing()

feature_target = 'diabetes'

df = normalize_dataset(df, feature_target)

target = df[feature_target]
data = df.drop(columns=[feature_target])

nn = train_neural_network(df,feature_target,"Originale")
"""

############### CON OVERSAMPLING + UNDERSAMPLING

"""
df = pre_processing()

feature_target = 'diabetes'

df = balanced_sampling(df,feature_target)

df = normalize_dataset(df, feature_target)

target = df[feature_target]
data = df.drop(columns=[feature_target])

nn = train_neural_network(df,feature_target,"OverSampling")
"""