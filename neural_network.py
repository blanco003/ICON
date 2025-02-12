from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from keras import layers
import numpy as np
import warnings
from tensorflow.keras.optimizers import SGD 

from plot import plot_curves_nn


def build_model(units_1, hidden_units_1, hidden_units_2, learning_rate, optimizer):
    """Costruisce la rete neurale con la configurazione passata in input"""

    model = keras.Sequential()
    model.add(layers.Dense(units_1, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(hidden_units_1, activation='relu'))
    model.add(layers.Dense(hidden_units_2, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # recupero del tipo di ottimizzatore
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def train_neural_network(df, target_feature, metodo):
    """Addestramento rete neurale"""

    # divisione dei dati in training e testing, e da training in validation e trainging
    train_data, test_data, train_targets, test_targets = train_test_split(df, target_feature, test_size=0.3, random_state=42)
    train_data, validation_data, train_targets, validation_targets = train_test_split(train_data, train_targets, test_size=0.2, random_state=42)

    # parametri da ricerca con gridsearch
    param_grid = {
        'model__units_1': [8],
        'model__hidden_units_1': [4,8,16,24,32],
        'model__hidden_units_2': [4,8,16,24,32],
        'model__optimizer': ['sgd','adam','rmsprop'],
        'model__learning_rate': [0.01],
        'batch_size': [128],
        'epochs': [50],
    }

    model = KerasClassifier(model=build_model,verbose=1)

    # Grid Search con Cross-Validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1)
    grid_search.fit(train_data, train_targets)

    print("\nConfigurazioni provate e corrispondente score : ")

    # Stampa dei parametri per ogni configurazione testata
    for params, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        print(f"Config: {params}, Score: {score}")

    # Migliore configurazione trovata
    best_params = grid_search.best_params_
    print(f"\nMiglior numero di neuroni nel primo layer: {best_params['model__units_1']}")
    print(f"Miglior numero di unità nel 1 hidden layers: {best_params['model__hidden_units_1']}")
    print(f"Miglior numero di unità nel 2 hidden layers: {best_params['model__hidden_units_2']}")
    print(f"Miglior learning rate: {best_params['model__learning_rate']}")
    print(f"Miglior optimizer: {best_params['model__optimizer']}")

    # Valutazione sul validation set
    val_predictions = grid_search.predict(validation_data)
    val_preds = np.round(val_predictions)
    print('Report di classificazione per il validation set:\n', classification_report(validation_targets, val_preds))

    # Valutazione sul test set
    test_preds = grid_search.predict(test_data)
    test_preds = np.round(test_preds)
    print('Report di classificazione per il test set:\n', classification_report(test_targets, test_preds))

    # modello con i migliori parametri trovati
    best_model = build_model(
        units_1=best_params['model__units_1'],
        hidden_units_1=best_params['model__hidden_units_1'],
        hidden_units_2=best_params['model__hidden_units_2'],
        learning_rate=best_params['model__learning_rate'],
        optimizer=best_params['model__optimizer']
    )

    history = best_model.fit(
        train_data, train_targets,
        validation_data=(validation_data, validation_targets),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=1
    )

    plot_curves_nn(history,best_params,metodo)

    return best_model