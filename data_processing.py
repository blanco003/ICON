import pandas as pd

from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, StandardScaler
from imblearn.over_sampling import SMOTE  
from imblearn.under_sampling import RandomUnderSampler


def pre_processing():
    """Carica il dataset dal file csv, ed effettua la fase di pre processing"""

    df = pd.read_csv("datasets/diabetes_prediction_dataset.csv")

    """
    print(f"Dimensioni del dataset : {df.shape}")

    print("\nValori Nulli : ")
    print(df.isnull().sum())

    print("\nRighe duplicate : ")
    print(df.duplicated().sum())

    print("\nInformazioni sulle colonne del dataset")
    print(df.info())

    print("\nValori unici di ogni feature : ")
    for col in df.columns:
        print(f"\n{col}")
        print(df[col].unique())

    # elimino le 3854 righe duplicate
    df = df.drop_duplicates()  
    """

    # mappatura della feature smoking
    mapping = {
        'never': 'non-smoker',
        'No Info': 'non-smoker',
        'current': 'current',
        'ever': 'past_smoker',
        'former': 'past_smoker',
        'not current': 'past_smoker'
    }

    df['smoking_history'] = df['smoking_history'].map(mapping)

    # elimino valori rari (Other occore 0.00195% nell'intero dataset)
    df = df[df['gender'] != 'Other']

    # One Hot Encoding
    cols_one_hot = ['gender', 'smoking_history']
    df = pd.get_dummies(df, columns=cols_one_hot, drop_first=True)

    # salviamo il dataframe ripulito
    df.to_csv('datasets\dataset_preprocessed.csv', index=0)

    return df



def standardize_dataset(df):
    """Effettua la Standardizzazione del dataset,  
       trasforma i dati in una distribuzione con media 0 e deviazione standard 1 """

    num_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']

    scaler = StandardScaler()

    # Standardizzazione delle variabili numeriche
    df[num_features] = scaler.fit_transform(df[num_features])

    return df


def discretize_dataset(df):
    """Discretizzazione delle features continue del dataset,
       divide le i valori delle features continue in intervalli con strategia uniforme"""

    # inizializzazione discretizer
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')

    # ricaviamo le features a valori continui del dataset
    continuos_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # e discretizziamole
    df[continuos_columns] = discretizer.fit_transform(df[continuos_columns])

    return df


def normalize_dataset(df, target):
    """Normalizzazione delle feature continue del datset,
       riscala i valori numerici nell'intervallo [0,1]"""

    # Separare il target
    X = df.drop(columns=[target])  # Escludere la colonna target
    y = df[target]  # Salvare la colonna target

    # Selezionare solo le colonne numeriche per la normalizzazione
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

    # Normalizzare le colonne numeriche
    scaler = MinMaxScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # Creare un nuovo DataFrame con i dati normalizzati
    df_normalized = pd.DataFrame(X, columns=X.columns)
    df_normalized[target] = y  # Reintegrare la colonna target

    return df_normalized


def over_sampling(df, target):
    "Effettua l'oversampling del dataset, in modo da bilanciare la distribuzione della feature target"

    # ricaviamo dal dataset la colonna della feature target
    y = df[target]

    # ricaviamo le features del dataset, escludendo la feature target
    X = df.drop(columns=[target])

    smote = SMOTE(random_state=42) #random_state=42 per avere sempre lo stesso risultato

    # Applicazione di SMOTE al dataset, per generare nuove istanze sintetiche al fine di bilanciare le classi rispetto al target
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Trasformiamo i dati ottenuti dall'oversampling in un nuovo DataFrame pandas
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled

    print('\n\033[93m' + "OVERSAMPLING EFFETTUATO CON SUCCESSO" + '\033[0m')
    print('\033[93m'+ f'Sono stati aggiunti {df_resampled.shape[0]-df.shape[0]} esempi nel dataset per bilanciare la distribuzione delle diagnosi' + '\033[0m')

    # salviamo il dataset con la distribuzione della feature target bilanciata
    #df_resampled.to_csv('datasets\dataset_oversampled.csv', index=0)

    return df_resampled # il nuovo dataset con le classi bilanciate



def balanced_sampling(df, target):
    """Effettua il bilanciamento del dataest rispetto al target,
       aumentando la classe minoritaria del 30% della classe maggioritaria e
       diminuendo la classe maggioriataria fino al bilanciamento"""

    #print(df[target].value_counts())

    df = over_sampling_2(df,target)
    #print(df[target].value_counts())

    df = under_sampling_2(df,target)
    #print(df[target].value_counts())
    
    return df






def over_sampling_2(df, target):
    """Effettua l'oversampling del dataset rispetto al target, 
       ma aumentando la classe minoritaria solo del 30% rispetto alla classe maggioritaria"""


    y = df[target]
    X = df.drop(columns=[target])

    smote = SMOTE(sampling_strategy=0.3) 

    # Applicazione di SMOTE al dataset, per generare nuove istanze sintetiche al fine di bilanciare le classi rispetto al target
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled

    print('\n\033[93m' + "OVERSAMPLING EFFETTUATO CON SUCCESSO" + '\033[0m')
    print('\033[93m'+ f'Sono stati aggiunti {df_resampled.shape[0]-df.shape[0]} esempi nel dataset per bilanciare la distribuzione delle diagnosi' + '\033[0m')

    # salviamo il dataset con la distribuzione della feature target bilanciata
    #df_resampled.to_csv('datasets\dataset_oversampled.csv', index=0)

    return df_resampled


def under_sampling_2(df, target):
    """Effettua l'undersampling del dataset,
       diminuendo la classe maggioritaria fino al bilanciamento"""
    # Features del dataset
    X = df.drop(columns=[target])

    # Feature target
    y = df[target]

    # Inizializzazione di RandomUnderSampler
    undersampler = RandomUnderSampler()

    # Applicazione dell'undersampling
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    # Creazione del DataFrame risultante
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled

    print('\n\033[93m' + "UNDERSAMPLING EFFETTUATO CON SUCCESSO" + '\033[0m')
    print('\033[93m' + f'Sono stati eliminati {len(df) - len(df_resampled)} esempi dal dataset per bilanciare la distribuzione delle diagnosi' + '\033[0m')

    # Salvataggio del dataset sottocampionato in un file CSV
    #df_resampled.to_csv('datasets/dataset_undersampled.csv', index=False)

    return df_resampled



def prepare_dataset_for_unsupervised(df,target):
    """Prepara il dataset affinche sia pronto per l'apprendimento non supervisionato"""

    # normalizziamo i valori delle feature
    df = normalize_dataset(df,target)

    # rimuoviamo i pazienti non diabetici
    df = df[df[target] != 0]

    # rimuoviamo la feature target in quanto inutile
    df = df.drop(columns=[target])

    return df




