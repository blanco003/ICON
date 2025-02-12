from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from plot import plot_learning_curves, plot_model_metrics



def return_best_hyperparameters_tree_based(df, target_feature):
    """Ricerca dei migliori iperparametri dei modelli ad albero, tramite GridSearch e Cross Validation"""

    X = df.drop(target_feature, axis=1).to_numpy()
    y = df[target_feature].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rfc = RandomForestClassifier()
    dtc = DecisionTreeClassifier()
    gdb = GradientBoostingClassifier()

    DecisionTreeHyperparameters = {
        'DecisionTree__criterion': ['gini', 'entropy'],
        'DecisionTree__max_depth': [7, 10, 12],  
        'DecisionTree__min_samples_split': [8, 10, 15],  
        'DecisionTree__min_samples_leaf': [5, 7, 10], 
        'DecisionTree__splitter': ['best']  
    }

    RandomForestHyperparameters = {
        'RandomForest__criterion': ['gini', 'entropy'], 
        'RandomForest__n_estimators': [50, 100, 200], 
        'RandomForest__max_depth': [5, 10, 15],  
        'RandomForest__min_samples_split': [5, 10, 15],  
        'RandomForest__min_samples_leaf': [3, 5, 10]  
    }

    GradientBoostingHyperparameters = {
        'GradientBoosting__n_estimators': [50, 100, 150],  
        'GradientBoosting__learning_rate': [0.01, 0.05, 0.1],  
        'GradientBoosting__max_depth': [5, 7, 10], 
        'GradientBoosting__min_samples_split': [5, 8, 12],  
        'GradientBoosting__min_samples_leaf': [3, 5, 7, 10], 
    }

    gridSearchCV_dtc = GridSearchCV(Pipeline([('DecisionTree', dtc)]), DecisionTreeHyperparameters, cv=5,verbose=1)
    gridSearchCV_rfc = GridSearchCV(Pipeline([('RandomForest', rfc)]), RandomForestHyperparameters, cv=5,verbose=1)
    gridSearchCV_gdb = GridSearchCV(Pipeline([('GradientBoosting', gdb)]), GradientBoostingHyperparameters, cv=5, verbose=1)

    gridSearchCV_dtc.fit(X_train, y_train)
    gridSearchCV_rfc.fit(X_train, y_train)
    gridSearchCV_gdb.fit(X_train, y_train)

    bestParameters = {
        'DecisionTree__criterion': gridSearchCV_dtc.best_params_['DecisionTree__criterion'],
        'DecisionTree__max_depth': gridSearchCV_dtc.best_params_['DecisionTree__max_depth'],
        'DecisionTree__min_samples_split': gridSearchCV_dtc.best_params_['DecisionTree__min_samples_split'],
        'DecisionTree__min_samples_leaf': gridSearchCV_dtc.best_params_['DecisionTree__min_samples_leaf'],

        'RandomForest__n_estimators': gridSearchCV_rfc.best_params_['RandomForest__n_estimators'],
        'RandomForest__max_depth': gridSearchCV_rfc.best_params_['RandomForest__max_depth'],
        'RandomForest__min_samples_split': gridSearchCV_rfc.best_params_['RandomForest__min_samples_split'],
        'RandomForest__min_samples_leaf': gridSearchCV_rfc.best_params_['RandomForest__min_samples_leaf'],
        'RandomForest__criterion': gridSearchCV_rfc.best_params_['RandomForest__criterion'],

        'GradientBoosting__n_estimators': gridSearchCV_gdb.best_params_['GradientBoosting__n_estimators'],
        'GradientBoosting__learning_rate': gridSearchCV_gdb.best_params_['GradientBoosting__learning_rate'],
        'GradientBoosting__max_depth': gridSearchCV_gdb.best_params_['GradientBoosting__max_depth'],
        'GradientBoosting__min_samples_split': gridSearchCV_gdb.best_params_['GradientBoosting__min_samples_split'],
        'GradientBoosting__min_samples_leaf': gridSearchCV_gdb.best_params_['GradientBoosting__min_samples_leaf']
  
    }

    return bestParameters



def train_model_kfold_tree_based(df, target_feature,metodo):
    """Addestramento dei modelli supervisionati basati ad albero"""

    model={
        'DecisionTree':{
            'accuracy_list':[],
            'precision_list':[],
            'recall_list':[],
            'f1':[],
            'dtc' : DecisionTreeClassifier()
        },
        'RandomForest':{
            'accuracy_list':[],
            'precision_list':[],
            'recall_list':[],
            'f1':[],
            'rfc' : RandomForestClassifier()
        },
        'GradientBoostingClassifier': {
            'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1': [],
            'gdb': GradientBoostingClassifier()
        }
    }

    print("\nCalcolo degli iperparametri...")

    bestParameters = return_best_hyperparameters_tree_based(df, target_feature)

    print("Ricerca iperparametri terminata")

    print("\033[94m"+str(bestParameters)+"\033[0m")

    X = df.drop(target_feature, axis=1)
    # features =  X.columns
    X = X.to_numpy()

    y = df[target_feature].to_numpy()

    # inizializzazione dei modelli con iperparametri ottimali trovati

    dtc = DecisionTreeClassifier(criterion=bestParameters['DecisionTree__criterion'],
                                 splitter='best',
                                 max_depth=bestParameters['DecisionTree__max_depth'],
                                 min_samples_split=bestParameters['DecisionTree__min_samples_split'],
                                 min_samples_leaf=bestParameters['DecisionTree__min_samples_leaf'])

    rfc = RandomForestClassifier(n_estimators=bestParameters['RandomForest__n_estimators'],
                                 max_depth=bestParameters['RandomForest__max_depth'],
                                 min_samples_split=bestParameters['RandomForest__min_samples_split'],
                                 min_samples_leaf=bestParameters['RandomForest__min_samples_leaf'],
                                criterion=bestParameters['RandomForest__criterion'])

    gdb = GradientBoostingClassifier(
        n_estimators=bestParameters['GradientBoosting__n_estimators'],
        learning_rate=bestParameters['GradientBoosting__learning_rate'],
        max_depth=bestParameters['GradientBoosting__max_depth'],
        min_samples_split=bestParameters['GradientBoosting__min_samples_split'],
        min_samples_leaf=bestParameters['GradientBoosting__min_samples_leaf']
    )

    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    results_dtc = {}
    results_rfc = {}
    results_gdb = {}

    for metric in scoring_metrics:
        scores_dtc = cross_val_score(dtc, X, y, scoring=metric, cv=cv)
        scores_rfc = cross_val_score(rfc, X, y, scoring=metric, cv=cv)
        scores_gdb = cross_val_score(gdb, X, y, scoring=metric, cv=cv)

        results_dtc[metric] = scores_dtc
        results_rfc[metric] = scores_rfc
        results_gdb[metric] = scores_gdb 

        print("\033[94m")
        print(f"Metric: {metric}")
        print(f"RandomForest: {scores_rfc.mean()}")
        print(f"DecisionTree: {scores_dtc.mean()}")
        print(f"GradientBoostingClassifier: {scores_gdb.mean()}")
        print("\033[0m")

    model['DecisionTree']['accuracy_list'] = (results_dtc['accuracy'])
    model['DecisionTree']['precision_list'] = (results_dtc['precision_macro'])
    model['DecisionTree']['recall_list'] = (results_dtc['recall_macro'])
    model['DecisionTree']['f1'] = (results_dtc['f1_macro'])
    model['DecisionTree']['dtc'] = dtc

    model['RandomForest']['accuracy_list'] = (results_rfc['accuracy'])
    model['RandomForest']['precision_list'] = (results_rfc['precision_macro'])
    model['RandomForest']['recall_list'] = (results_rfc['recall_macro'])
    model['RandomForest']['f1'] = (results_rfc['f1_macro'])
    model['RandomForest']['rfc'] = rfc

    model['GradientBoostingClassifier']['accuracy_list'] = results_gdb['accuracy']
    model['GradientBoostingClassifier']['precision_list'] = results_gdb['precision_macro']
    model['GradientBoostingClassifier']['recall_list'] = results_gdb['recall_macro']
    model['GradientBoostingClassifier']['f1'] = results_gdb['f1_macro']
    model['GradientBoostingClassifier']['gdb'] = gdb

    plot_learning_curves(dtc, X, y, target_feature, 'DecisionTree', metodo)
    plot_learning_curves(rfc, X, y, target_feature, 'RandomForest', metodo)
    plot_learning_curves(gdb, X, y, target_feature, 'GradientBoostingClassifier', metodo)

    mean_metrics_dtc = {
            'accuracy': np.mean(results_dtc['accuracy']),
            'precision':np.mean(results_dtc['precision_macro']),
            'recall': np.mean(results_dtc['recall_macro']),
            'f1': np.mean(results_dtc['f1_macro'])
    }

    mean_metrics_rfc = {
            'accuracy': np.mean(results_rfc['accuracy']),
            'precision':np.mean(results_rfc['precision_macro']),
            'recall': np.mean(results_rfc['recall_macro']),
            'f1': np.mean(results_rfc['f1_macro'])
    }

    mean_metrics_gdb = {
            'accuracy': np.mean(results_gdb['accuracy']),
            'precision':np.mean(results_gdb['precision_macro']),
            'recall': np.mean(results_gdb['recall_macro']),
            'f1': np.mean(results_gdb['f1_macro'])
    }

    plot_model_metrics("Decision Tree", mean_metrics_dtc, metodo)
    plot_model_metrics("Random Forest", mean_metrics_rfc, metodo)
    plot_model_metrics("Gradient Boosting", mean_metrics_gdb, metodo)

    return model



def return_best_hyperparameters_lg(df, target_feature):
    """Ricerca dei migliori iperparametri della regressione logistica, tramite GridSearch e Cross Validation"""

    X = df.drop(target_feature, axis=1).to_numpy()
    y = df[target_feature].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg = LogisticRegression()

    LogisticRegressionHyperparameters = {
        'LogisticRegression__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'LogisticRegression__penalty': ['l2'],
        'LogisticRegression__solver': ['liblinear', 'lbfgs'],
        'LogisticRegression__max_iter': [100000,150000]}

    gridSearchCV_reg = GridSearchCV(Pipeline([('LogisticRegression', reg)]), LogisticRegressionHyperparameters, cv=5,verbose=1)

    gridSearchCV_reg.fit(X_train, y_train)

    bestParameters = {
        'LogisticRegression__C': gridSearchCV_reg.best_params_['LogisticRegression__C'],
        'LogisticRegression__penalty': gridSearchCV_reg.best_params_['LogisticRegression__penalty'],
        'LogisticRegression__solver': gridSearchCV_reg.best_params_['LogisticRegression__solver'],
        'LogisticRegression__max_iter': gridSearchCV_reg.best_params_['LogisticRegression__max_iter'],
    }
    return bestParameters




def train_model_kfold_lg(df, target_feature, metodo):
    """Addestramento modello supervisionato della regressione logistica"""

    model={
        'LogisticRegression':{
            'accuracy_list':[],
            'precision_list':[],
            'recall_list':[],
            'f1':[],
            'reg' : LogisticRegression()
        }
    }

    print("\nCalcolo degli iperparametri...")

    bestParameters = return_best_hyperparameters_lg(df, target_feature)
    
    print("Ricerca iperparametri terminata.")

    print("\033[94m"+str(bestParameters)+"\033[0m")

    X = df.drop(target_feature, axis=1)
    features =  X.columns
    X = X.to_numpy()

    y = df[target_feature].to_numpy()


    reg = LogisticRegression(C=bestParameters['LogisticRegression__C'],
                             penalty=bestParameters['LogisticRegression__penalty'],
                             solver=bestParameters['LogisticRegression__solver'],
                             max_iter=bestParameters['LogisticRegression__max_iter'])

    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    results_reg = {}

    for metric in scoring_metrics:

        scores_reg = cross_val_score(reg, X, y, scoring=metric, cv=cv)

        results_reg[metric] = scores_reg

        print("\033[94m")
        print(f"Metric: {metric}")
        print(f"LogisticRegression: {scores_reg.mean()}")
        print("\033[0m")

    model['LogisticRegression']['accuracy_list'] = (results_reg['accuracy'])
    model['LogisticRegression']['precision_list'] = (results_reg['precision_macro'])
    model['LogisticRegression']['recall_list'] = (results_reg['recall_macro'])
    model['LogisticRegression']['f1'] = (results_reg['f1_macro'])
    model['LogisticRegression']['reg'] = reg

    plot_learning_curves(reg, X, y, target_feature, 'LogisticRegression', metodo)

    mean_metrics_reg = {
            'accuracy': np.mean(results_reg['accuracy']),
            'precision':np.mean(results_reg['precision_macro']),
            'recall': np.mean(results_reg['recall_macro']),
            'f1': np.mean(results_reg['f1_macro'])
    }

    plot_model_metrics("Logistic Regression", mean_metrics_reg, metodo)

    return model


