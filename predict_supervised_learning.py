#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""/*********************************************************************************
 Predict Supervised Learning

 Executa o Aprendizado Supervisionado através de algoritmos de Machine Learning com
 otimização dos hiperparâmetros e tratamento dos dados desbalanceados pelo método
 RENN - Repeated Edited Nearest Neighbour, salvando as métricas obtidas no formato
 DataFrame do pandas e gerando arquivo raster no formato GeoTiff resultante da
 predição do modelo treinado.
                               -------------------
        begin                : 2020-07-01
        copyright            : (C) 2020 by Yoshio Urasaki
        email                : yoshio.urasaki@gmail.com
 ********************************************************************************/"""

import os
import pickle
import numpy as np
import pandas as pd


def convert_raster_2_array(data_features: str, data_target: str, dump_dir: str,
                           features_label: tuple = ('B', 'G', 'R', 'NIR', 'SWIR_1', 'SWIR_2'),
                           target_label: tuple = ('LC',)):
    """
    Converte as Imagem a serem utilizadas no Aprendizado Supervisionado em Datasets para aplicação de Algoritmos
    de Machine Learning.

    :parameter
        data_features (str): path da imagem com dados utilizados como previsores no aprendizado supervisionado
        data_target (str): path da imagem com dados utilizados como rótulo no aprendizado supervisionado
        dump_dir (str): path padrão de saida
        features_label (tuple): label das colunas do dataset de entrada
        target_label (tuple): label das colunas do dataset de saída

    :return
        x (array): dataset de entrada (previsores)
        y (array): dataset de saída (rótulos)
        rows (int): número de linhas do dataset
        columns (int): número de colunas do dataset
        x_dataset (object): objeto do dataset de entrada (previsores)
        y_dataset (object): objeto do dataset de saída (rótulos)
    """
    from pyrsgis import raster

    if os.path.isfile(data_features) and os.path.isfile(data_target):
        try:
            os.chdir(dump_dir)

            x_dataset, x_feature_file = raster.read(data_features, bands='all')

            x = []
            for i in range(len(x_feature_file)):
                band = np.ravel(x_feature_file[i])
                x.append(band)

            x = np.asarray(x)
            x = x.T
            x = pd.DataFrame(data=x, columns=features_label)

            y_dataset, y_feature_file = raster.read(data_target, bands=1)
            y = np.ravel(y_feature_file)
            y = pd.DataFrame(data=y, columns=target_label)

            rows_value = y_feature_file.shape[0]
            columns_value = y_feature_file.shape[1]

            return [x, y, rows_value, columns_value, x_dataset, y_dataset]
        except Exception:
            pass
    else:
        raise ValueError(f'{data_features} or {data_target} files not found.')


def cleaner_data(x, y, class_id: int):
    """
    Remove determinado rótulo no dataset de saída e correspondente registro no dataset de entrada utilizando
    seu valor identificador.

    :parameter
        x (array): dataset de entrada (previsores)
        y (array): dataset de saída (rótulos)
        class_id (int): valor identificador do rótulo

    :return
        x_clean (array): dataset de entrada após classe especificada removida
        y_clean (array): dataset de saída após classe especificada removida
    """
    y_clean = y.replace(class_id, np.nan)
    x_clean = x.drop(x[y_clean['LC'].isnull()].index)
    y_clean.dropna(inplace=True)

    x_clean = np.asarray(x_clean)
    y_clean = np.ravel(np.asarray(y_clean).astype(np.int16))

    return [x_clean, y_clean]


def imbalanced_class(x, y, renn_filename: str):
    """
    Aplica o algoritmo de Undersample, Repeated Edited Nearest Neighbour (RENN), nos datasets e salva em NPZ.

    :parameter
        x (array): dataset de entrada (previsores)
        y (array): dataset de saída (rótulos)
        renn_filename (str): nome do arquivo NPZ para salvar os datasets após aplicação do RENN.

    :return
        x_renn (array): dataset de entrada após aplicação do RENN
        y_renn (array): dataset de saída após aplicação do RENN
    """
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours

    undersample_renn = RepeatedEditedNearestNeighbours()
    x_renn, y_renn = undersample_renn.fit_resample(x, y)
    np.savez(f'{renn_filename}.npz', x_renn, y_renn)

    return [x_renn, y_renn]


def split_dataset(x, y, test_size: float):
    """
    Divisão dos datasets em conjuntos de treinamento e teste.

    :parameter
        x (array): dataset de entrada (previsores)
        y (array): dataset de saída (rótulos)
        test_size (float): proporção do dataset para incluir no conjunto de teste

    :return
        x_train (array): dataset de entrada para treinamento
        x_test (array): dataset de entrada para teste
        y_train (array): dataset de saida para treinamento
        y_test (array): dataset de saida para teste
    """
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=0, shuffle=True, stratify=y)

    return [x_train, x_test, y_train, y_test]


class PreProcessingData:
    """
    Engobla diversas atividades de pré-processamento envolvendo a preparação, limpeza, tratamento de dados
    desbalanceados e divisão do dataset em conjunto de treinamento e teste.
    """
    def __init__(self, data_features, data_target, dump_dir, class_id, renn_file: str = 'RENN',
                 test_size: float = 0.30):
        self._data_features = data_features
        self._data_target = data_target
        self._class_id = class_id
        self._renn_file = renn_file
        self._test_size = test_size
        self.dump_dir = dump_dir

        self.x, self.y, self.rows, self.columns, self.x_dataset, self.y_dataset = convert_raster_2_array(
            self._data_features, self._data_target, self.dump_dir
        )

        if os.path.isfile(os.path.join(self.dump_dir, f'{self._renn_file}.npz')):
            self.x_clean, self.y_clean = cleaner_data(self.x, self.y, self._class_id)

            renn_data = np.load(os.path.join(self.dump_dir, f'{self._renn_file}.npz'))
            self.x_renn = renn_data['arr_0']
            self.y_renn = renn_data['arr_1']
        else:
            self.x_clean, self.y_clean = cleaner_data(self.x, self.y, self._class_id)
            self.x_renn, self.y_renn = imbalanced_class(self.x_clean, self.y_clean, self._renn_file)

        self.x_train, self.x_test, self.y_train, self.y_test = split_dataset(self.x_renn, self.y_renn, self._test_size)

# ---------------------------------------------------------------------------------------------------------------------


def create_domain_space(title: str):
    """
    Define o modelo e espaço de domínio dos hiperparâmetros.

    :parameter
        title (str): título do algoritmo para geração do modelo e espaço de domínio dos hiperparâmetros: (
        'dummy_classifier', 'gaussian_naive_bayes', 'decision_tree', 'random_forest', 'kneighbors',
        'logistic_regression', 'linear_support_vector', 'multi_layer_perceptron', 'adaboost')

    :return
        space_domain (dict): espaço de domínio dos hiperparâmetros
        algorithm_model (object): modelo do algoritmo de machine learning
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from hyperopt import hp

    models_dict = {
        'dummy_classifier': DummyClassifier,
        'gaussian_naive_bayes': GaussianNB,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'kneighbors': KNeighborsClassifier,
        'logistic_regression': LogisticRegression,
        'linear_support_vector': LinearSVC,
        'multi_layer_perceptron': MLPClassifier,
        'adaboost': AdaBoostClassifier
    }

    spaces_dict = {
        'dummy_classifier': {
            # strategy to use to generate predictions
            'strategy': hp.choice('strategy', ['stratified', 'most_frequent', 'prior', 'uniform']),
            # controls the randomness to generate the predictions
            'random_state': 0
        },
        'gaussian_naive_bayes': {
            # portion of the largest variance of all features that is added to variances for calculation stability
            'var_smoothing': hp.choice('var_smoothing', np.logspace(0, -9, num=10)),
        },
        'decision_tree': {
            # function to measure the quality of a split
            'criterion': 'gini',  # hp.choice('criterion', ['gini', 'entropy']),
            # strategy used to choose the split at each node
            'splitter': 'best',  # hp.choice('splitter', ['best', 'random']),
            # minimum number of samples required to split an internal node
            'min_samples_split': hp.choice('min_samples_split', list(range(2, 10))),
            # minimum number of samples required to be at a leaf node
            'min_samples_leaf': hp.choice('min_samples_leaf', list(range(1, 10))),
            # minimum weighted fraction of the sum total of weights
            'min_weight_fraction_leaf': 1e-7,  # hp.choice('min_weight_fraction_leaf', np.logspace(-2, -7, num=6)),
            # number of features to consider when looking for the best split
            'max_features': hp.choice('max_features', ['sqrt', 'log2']),
            # a node will be split if this split induces a decrease of the impurity
            'min_impurity_decrease': 1e-7,  # hp.choice('min_impurity_decrease', np.logspace(-2, -7, num=6)),
            # weights associated with classes
            'class_weight': 'balanced',  # hp.choice('class_weight', [None, 'balanced'])
            # controls the randomness to generate the predictions
            'random_state': 0
        },
        'random_forest': {
            # number of trees in the forest
            'n_estimators': hp.choice('n_estimators', list(range(10, 100, 5))),
            # function to measure the quality of a split
            'criterion': 'gini',  # hp.choice('criterion', ['gini', 'entropy']),
            # maximum depth of the tree;
            'max_depth': hp.choice('max_depth', list(range(0, 30))),
            # minimum number of samples required to split an internal node
            'min_samples_split': hp.choice('min_samples_split', list(range(2, 10))),
            # minimum number of samples required to be at a leaf node
            'min_samples_leaf': hp.choice('min_samples_leaf', list(range(1, 10))),
            # minimum weighted fraction of the sum total of weights
            'min_weight_fraction_leaf': 1e-7,  # hp.choice('min_weight_fraction_leaf', np.logspace(-2, -7, num=6)),
            # number of features to consider when looking for the best split
            'max_features': hp.choice('max_features', ['sqrt', 'log2']),
            # a node will be split if this split induces a decrease of the impurity
            'min_impurity_decrease': 1e-7,  # hp.choice('min_impurity_decrease', np.logspace(-2, -7, num=6)),
            # wether bootstrap samples are used when building trees
            'bootstrap': True,  # hp.choice('bootstrap', [True, False]),
            # weights associated with classes
            'class_weight': 'balanced',  # hp.choice('class_weight', [None, 'balanced'])
            # controls the randomness to generate the predictions
            'random_state': 0
        },
        'kneighbors': {
            # number of neighbors to use by default for kneighbors queries
            'n_neighbors': hp.choice('n_neighbors', list(range(1, 10))),
            # weight function used in prediction
            'weights': 'distance',  # hp.choice('weights', ['uniform', 'distance']),
            # algorithm used to compute the nearest neighbors
            'algorithm': 'auto',  # hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            # leaf size passed to BallTree or KDTree
            'leaf_size': hp.choice('leaf_size', list(range(1, 30))),
            # power parameter for the Minkowski metric
            'p': 6  # hp.choice('p', range(1, 7))
        },
        'logistic_regression': {
            # used to specify the norm used in the penalization
            'penalty': 'l2',  # hp.choice('penalty', ['l1', 'l2']),
            # tolerance for stopping criteria
            'tol': 1e-4,  # hp.choice('tol', np.logspace(-2, -7, num=6)),
            # inverse of regularization strength
            'C': hp.uniform('C', 0.01, 100),
            # weights associated with classes
            'class_weight': 'balanced',  # hp.choice('class_weight', [None, 'balanced'])
            # algorithm to use in the optimization problem
            'solver': 'lbfgs',  # hp.choice('solver', ['newton-cg', 'lbfgs', 'sag']),
            # maximum number of iterations taken for the solvers to converge
            'max_iter': hp.choice('max_iter', list(range(100, 500, 20))),
            # type of adjustment depending on multi-class
            'multi_class': 'ovr',
            # controls the randomness to generate the predictions
            'random_state': 0
        },
        'linear_support_vector': {
            # used to specify the norm used in the penalization
            'penalty': 'l2',  # hp.choice('penalty', ['l1', 'l2']),
            # specifies the loss function
            'loss': 'squared_hinge',  # hp.choice('loss', ['hinge', 'squared_hinge']),
            # tolerance for stopping criteria
            'tol': 1e-4,  # hp.choice('tol', np.logspace(-2, -7, num=6)),
            # inverse of regularization strength
            'C': hp.uniform('C', 1, 100),  # hp.uniform('C', 0.01, 100),
            # type of adjustment depending on multi-class
            'multi_class': 'ovr',
            # weights associated with classes
            'class_weight': 'balanced',  # hp.choice('class_weight', [None, 'balanced'])
            # controls the randomness to generate the predictions
            'random_state': 0
        },
        'multi_layer_perceptron': {
            # number of neurons in hidden layer
            'hidden_layer_sizes': (10, 10, 10,),
            # activation function for the hidden layer
            'activation': 'tanh',  # hp.choice('activation', ['tanh', 'relu', 'logistic']),
            # algorithm to use in the optimization problem
            'solver': 'adam',  # hp.choice('solver', ['sgd', 'adam', 'lbfgs']),
            # L2 penalty (regularization term) parameter;
            'alpha': hp.choice('alpha', np.logspace(-2, -7, num=6)),  # 1e-5,
            # tolerance for stopping criteria
            'tol': 1e-4,  # hp.choice('tol', np.logspace(-2, -7, num=6)),
            # learning rate schedule for weight updates
            'learning_rate': 'adaptive',  # hp.choice('learning_rate', ['constant', 'adaptive']),
            # controls the randomness to generate the predictions
            'random_state': 0
        },
        'adaboost': {
            # base estimator from which the boosted ensemble is built
            'base_estimator': GaussianNB(var_smoothing=1e-2),
            # maximum number of estimators at which boosting is terminated
            'n_estimators': 50,
            # contribution of each classifier
            'learning_rate': 1,
            # boosting algorithm
            'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
            # controls the randomness to generate the predictions
            'random_state': 0
        }
    }

    try:
        space_domain = spaces_dict.get(title)
        algorithm_model = models_dict.get(title)

        return [space_domain, algorithm_model]
    except Exception:
        pass


def search_best_hyperparameter(title: str, x, y, dump_dir: str, stats_filename: str,
                               n_iter: int, n_splits: int, n_core: int):
    """
    Busca a otimização dos hiperparâmetros para o algoritmo selecionado.

    :parameter
        title (str): título do algoritmo para geração do modelo com hiperparâmetros otimizados: (
        'dummy_classifier', 'gaussian_naive_bayes', 'decision_tree', 'random_forest', 'kneighbors',
        'logistic_regression', 'linear_support_vector', 'multi_layer_perceptron', 'adaboost')
        x (array): dataset de entrada (previsores)
        y (array): dataset de saída (rótulos)
        dump_dir (str): path padrão de saida
        stats_filename (str): nome do arquivo para gravar as métricas obtidas
        n_iter (int): número de interações
        n_splits (int):  número de subconjuntos gerados na validação cruzada
        n_core (int): número maximo de core para processamento

    :return
        algorithm (object): algoritmo com os hiperparâmetros otimizados
    """
    import warnings
    from hyperopt import fmin, tpe, Trials, STATUS_OK
    from sklearn.model_selection import cross_val_score

    warnings.filterwarnings("ignore")

    space, model = create_domain_space(title)

    def func_hyperopt(space):
        clf = model(**space)
        score = cross_val_score(
            clf, x, y, scoring='f1_weighted', cv=n_splits, n_jobs=n_core, verbose=100, error_score=0
        ).mean()

        return {'loss': -score, 'status': STATUS_OK}

    tpe_trials = Trials()
    best_estimator = fmin(func_hyperopt, space, algo=tpe.suggest, max_evals=n_iter, trials=tpe_trials,
                          return_argmin=False, rstate=np.random.RandomState(0), timeout=43200)
    algorithm_opt = model(**best_estimator)

    if os.path.isfile(os.path.join(dump_dir, f'{stats_filename}.sav')):
        statistics = pickle.load(open(os.path.join(dump_dir, f'{stats_filename}.sav'), 'rb'))
    else:
        statistics = {}

    if title not in statistics:
        statistics[title] = {}

    statistics[title]['best_estimator'] = best_estimator

    # salva o modelo treinado
    pickle.dump(algorithm_opt, open(os.path.join(dump_dir, f'algorithm_{title}.sav'), 'wb'))

    # salva o desenvolvimento do TPE
    pickle.dump(tpe_trials, open(os.path.join(dump_dir, f'trials_{title}.sav'), 'wb'))

    # salva os valores dos hiperparâmetros obtidos
    pickle.dump(statistics, open(os.path.join(dump_dir, f'{stats_filename}.sav'), 'wb'))

    return algorithm_opt


class CreateModelHyperopt:
    """
    Cria modelo com os hiperparâmetros otimizados e salva os resultados obtidos no processo de otimização.
    """
    def __init__(self, title, x, y, dump_dir, stats_filename: str = 'statistics',
                 n_iter: int = 100, n_splits: int = 5, n_core: int = 5):
        self.title = title
        self._x = x
        self._y = y
        self._dump_dir = dump_dir
        self._stats_filename = stats_filename
        self._n_iter = n_iter
        self._n_splits = n_splits
        self._n_core = n_core

        if os.path.isfile(os.path.join(self._dump_dir, f'algorithm_{self.title}.sav')):
            self.algorithm = pickle.load(open(os.path.join(self._dump_dir, f'algorithm_{self.title}.sav'), 'rb'))
        else:
            self.algorithm = search_best_hyperparameter(
                self.title, self._x, self._y, self._dump_dir, self._stats_filename,
                self._n_iter, self._n_splits, self._n_core)

# ---------------------------------------------------------------------------------------------------------------------


def classifier_statistics(title: str, subtitle: str, algorithm, x, y, dump_dir: str, stats_filename: str):
    """
    Insere as métricas calculadas da predição em um dicionário.

    :parameter
        title (str): título do algoritmo para cálculo das métricas estatísticas: (
        'dummy_classifier', 'gaussian_naive_bayes', 'decision_tree', 'random_forest', 'kneighbors',
        'logistic_regression', 'linear_support_vector', 'multi_layer_perceptron', 'adaboost')
        subtitle (str): subtitulo para referencia do dataset analisado
        algorithm (object): algoritmo com os hiperparâmetros otimizados
        x (array): dataset de entrada (previsores)
        y (array): dataset de saída (rótulos)
        dump_dir (str): path padrão de saida
        stats_filename (str): nome do arquivo para gravar as métricas obtidas

    :return
        statistics (dict): dicionário com as métricas da predição.
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, recall_score, \
        precision_score, f1_score, roc_auc_score, classification_report
    from sklearn.preprocessing import LabelBinarizer

    if os.path.isfile(os.path.join(dump_dir, f'{stats_filename}.sav')):
        statistics = pickle.load(open(os.path.join(dump_dir, f'{stats_filename}.sav'), 'rb'))
    else:
        statistics = {}

    if title not in statistics:
        statistics[title] = {}

    if subtitle not in statistics[title]:
        statistics[title][subtitle] = {}

    algorithm.fit(x, y)
    y_predict = algorithm.predict(x)

    try:
        statistics[title][subtitle]['cohen_kappa'] = cohen_kappa_score(y, y_predict)

        lb = LabelBinarizer().fit(y)
        statistics[title][subtitle]['roc_auc'] = roc_auc_score(lb.transform(y), lb.transform(y_predict),
                                                               average='weighted')

        statistics[title][subtitle]['f1_score'] = f1_score(y, y_predict, average='weighted', zero_division=0)        
        statistics[title][subtitle]['accuracy'] = accuracy_score(y, y_predict)
        statistics[title][subtitle]['precision'] = precision_score(y, y_predict, average='weighted', zero_division=0)        
        statistics[title][subtitle]['recall'] = recall_score(y, y_predict, average='weighted', zero_division=0)
        statistics[title][subtitle]['matthews_corrcoef'] = matthews_corrcoef(y, y_predict)
        statistics[title][subtitle]['confusion_matrix'] = confusion_matrix(y, y_predict)
        statistics[title][subtitle]['report'] = classification_report(y, y_predict, output_dict=True, zero_division=0)

    except Exception:
        pass
    finally:
        pickle.dump(statistics, open(os.path.join(dump_dir, f'{stats_filename}.sav'), 'wb'))

    return statistics


class StatisticsPredict:
    """
    Calcula as métricas da predição utilizando os datasets nas diferentes etapas de processamento e
    salva no formato dataframe do pandas
    """
    def __init__(self, title, x, y, x_clean, y_clean, x_renn, y_renn, x_train, y_train, x_test, y_test,
                 dump_dir, stats_filename: str = 'statistics'):
        self.title = title
        self._x = x
        self._y = y
        self._x_clean = x_clean
        self._y_clean = y_clean
        self._x_renn = x_renn
        self._y_renn = y_renn
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._dump_dir = dump_dir
        self.stats_filename = stats_filename
        self.algorithm = pickle.load(open(os.path.join(self._dump_dir, f'algorithm_{self.title}.sav'), 'rb'))

        classifier_statistics(f'{self.title}', 'base', self.algorithm, self._x, self._y,
                              self._dump_dir, self.stats_filename)
        classifier_statistics(f'{self.title}', 'clean', self.algorithm, self._x_clean, self._y_clean,
                              self._dump_dir, self.stats_filename)
        classifier_statistics(f'{self.title}', 'renn', self.algorithm, self._x_renn, self._y_renn,
                              self._dump_dir, self.stats_filename)
        classifier_statistics(f'{self.title}', 'train', self.algorithm, self._x_train, self._y_train,
                              self._dump_dir, self.stats_filename)
        classifier_statistics(f'{self.title}', 'test', self.algorithm, self._x_test, self._y_test,
                              self._dump_dir, self.stats_filename)

        dados = pickle.load(open(os.path.join(self._dump_dir, f'{stats_filename}.sav'), 'rb'))
        dados = dados[self.title]
        dados = pd.DataFrame(dados)[['base', 'clean', 'renn', 'train', 'test']]
        dados = dados.T
        dados = dados[['cohen_kappa', 'roc_auc', 'f1_score', 'accuracy', 'precision', 'recall', 'matthews_corrcoef']]

        pickle.dump(dados, open(os.path.join(self._dump_dir, f'stats_{self.title}.sav'), 'wb'))

# ---------------------------------------------------------------------------------------------------------------------


def create_raster_predicted(title: str, algorithm, x, y, rows: int, columns: int, y_dataset, dump_dir: str):
    """
    Gera arquivo raster no formato GeoTiff resultante da predição do modelo treinado.

    :parameter
        title (str): título do algoritmo utilizado nas predições: (
        'dummy_classifier', 'gaussian_naive_bayes', 'decision_tree', 'random_forest', 'kneighbors',
        'logistic_regression', 'linear_support_vector', 'multi_layer_perceptron', 'adaboost')
        algorithm (object): algoritmo com os hiperparâmetros otimizados
        x (array): dataset de entrada (previsores)
        y (array): dataset de saída (rótulos)
        rows (int): número de linhas do dataset de saida
        columns (int): número de colunas do dataset de saida
        y_dataset (object): objeto do dataset de saída (rótulos)
        dump_dir (str): path padrão de saida

    :return
        y_predict (array): retorna predição do modelo treinado.
    """
    from pyrsgis import raster

    algorithm.fit(x, y)
    y_predict = algorithm.predict(x)
    y_predict = y_predict.reshape(rows, columns)

    raster.export(y_predict, y_dataset, os.path.join(dump_dir, f'classified_{title}.tif'), dtype='int16', bands=1)

    return y_predict

# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # definição dos caminhos dos arquivos de entrada e saida
    base_path = os.getcwd()
    # features_path = os.path.join(base_path, 'base', 'CLIP_RT_L05_L1TP_219076_20100824_T1_B123457.tif')
    # target_path = os.path.join(base_path, 'base', 'CLIP_COBERTURA_TERRA.tif')
    # dump_path = os.path.join(base_path, 'dump_demo')
    features_path = os.path.join(base_path, 'base', 'RT_L05_L1TP_219076_20100824_T1_B123457.tif')
    target_path = os.path.join(base_path, 'base', 'COBERTURA_TERRA.tif')
    dump_path = os.path.join(base_path, 'dump')

    # configuração das variáveis
    algorithm_list = ['dummy_classifier', 'gaussian_naive_bayes', 'decision_tree', 'random_forest', 'kneighbors',
                      'logistic_regression', 'linear_support_vector', 'multi_layer_perceptron', 'adaboost']

    algorithm_sel = algorithm_list[0]

    # pré-processamento (remoção da classe 7)
    class_clip = PreProcessingData(features_path, target_path, dump_path, 7)

    # criando modelos com hiperparâmetros otimizados
    CreateModelHyperopt(algorithm_sel, class_clip.x_renn, class_clip.y_renn, class_clip.dump_dir)

    # calculando metricas da predição
    StatisticsPredict(algorithm_sel, class_clip.x, class_clip.y, class_clip.x_clean, class_clip.y_clean,
                      class_clip.x_renn, class_clip.y_renn, class_clip.x_train, class_clip.y_train, class_clip.x_test,
                      class_clip.y_test, class_clip.dump_dir)

    # gera arquivo raster da predição
    algorithm_file = pickle.load(open(os.path.join(dump_path, f'algorithm_{algorithm_sel}.sav'), 'rb'))
    create_raster_predicted(algorithm_sel, algorithm_file, class_clip.x, class_clip.y, class_clip.rows,
                            class_clip.columns, class_clip.y_dataset,
                            os.path.join(class_clip.dump_dir, 'classified_raster'))
