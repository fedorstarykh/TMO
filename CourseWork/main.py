import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


class MetricLogger:

    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
             'alg': pd.Series([], dtype='str'),
             'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric'] == metric) & (self.df['alg'] == alg)].index, inplace=True)
        # Добавление нового значения
        temp = [{'metric': metric, 'alg': alg, 'value': value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric'] == metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5,
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a, b in zip(pos, array_metric):
            plt.text(0.5, a - 0.05, str(round(b, 3)), color='white')
        plt.show()


def load_data():
    # Загрузка данных
    data = pd.read_csv('data/pulsar_stars.csv')
    return data


# функции для обучения моделей
def train_model(model_name, model, classMetricLogger, is_print=1):
    model.fit(X_train, Y_train)
    # Предсказание значений
    Y_pred = model.predict(X_test)

    precision = precision_score(Y_test.values, Y_pred)
    recall = recall_score(Y_test.values, Y_pred)
    f1 = f1_score(Y_test.values, Y_pred)

    classMetricLogger.add('precision', model_name, precision)
    classMetricLogger.add('recall', model_name, recall)
    classMetricLogger.add('f1', model_name, f1)

    if is_print == 1:
        st.write(f'--------------------{model_name}--------------------')
        st.write(model)
        st.write(f"precision_score: {precision}")
        st.write(f"recall_score: {recall}")
        st.write(f"f1_score: {f1}")
        st.write(f'--------------------{model_name}--------------------\n')


data = load_data()

parts = np.split(data, [10], axis=1)
data = parts[0]

st.sidebar.header('Логистический регрессор')
cs_1 = st.sidebar.slider('Параметр регуляризации:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('Модель ближайших соседей')
n_estimators_2 = st.sidebar.slider('Количество K:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('SVC')
cs_3 = st.sidebar.slider('Регуляризация:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('Дерево решений')
max_depth_4 = st.sidebar.slider('Максимальная глубина:', min_value=10, max_value=50, value=10, step=1)

st.sidebar.header('Случайный лес')
n_estimators_5 = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('Градиентный бустинг')
n_estimators_6 = st.sidebar.slider('Количество:', min_value=6, max_value=15, value=6, step=1)

# Первые пять строк датасета
st.subheader('Первые 5 значений')
st.write(data.head())

st.subheader('Размер датасета')
st.write(data.shape)

st.subheader('Количество нулевых элементов')
st.write(data.isnull().sum())

st.subheader('Колонки и их типы данных')
st.write(data.dtypes)

st.subheader('Статистические данные')
st.write(data.describe())

# Убедимся, что целевой признак
# для задачи бинарной классификации содержит только 0 и 1
st.subheader('Целевой признак содержит только 0 и 1')
st.write(data['target_class'].unique())

st.subheader('Корреляционная матрица')
fig1, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
st.pyplot(fig1)

# разделение выборки на обучающую и тестовую
# X_train, X_test, Y_train, Y_test, X, Y = preprocess_data(data)
X = data.drop("target_class", axis=1)
Y = data["target_class"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# Числовые колонки для масштабирования
scale_cols = [' Mean of the integrated profile',
              ' Standard deviation of the integrated profile',
              ' Excess kurtosis of the integrated profile',
              ' Skewness of the integrated profile',
              ' Mean of the DM-SNR curve',
              ' Standard deviation of the DM-SNR curve',
              ' Excess kurtosis of the DM-SNR curve',
              ' Skewness of the DM-SNR curve']
sc1 = MinMaxScaler()
sc1_data = sc1.fit_transform(data[scale_cols])
# Добавим масштабированные данные в набор данных
for i in range(len(scale_cols)):
    col = scale_cols[i]
    new_col_name = col + '_scaled'
    data[new_col_name] = sc1_data[:, i]

st.subheader('Проверим, что масштабирование не повлияло на распределение данных')
for col in scale_cols:
    col_scaled = col + '_scaled'

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(data[col], 50)
    ax[1].hist(data[col_scaled], 50)
    ax[0].title.set_text(col)
    ax[1].title.set_text(col_scaled)
    st.pyplot(fig)

st.subheader('Корреляционная матрица после масштабирование')
scale_cols_postfix = [x + '_scaled' for x in scale_cols]
corr_cols_2 = scale_cols_postfix + ['target_class']
fig1, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(data[corr_cols_2].corr(), annot=True, fmt='.2f')
st.pyplot(fig1)

st.subheader('Обучим модели')
# Модели
models = {'LogR': LogisticRegression(C=cs_1),
          'KNN': KNeighborsClassifier(n_neighbors=n_estimators_2),
          'SVC': SVC(C=cs_3, probability=True),
          'Tree': DecisionTreeClassifier(max_depth=max_depth_4, random_state=10),
          'RF': RandomForestClassifier(n_estimators=n_estimators_5, oob_score=True, random_state=10),
          'GB': GradientBoostingClassifier(n_estimators=n_estimators_6, random_state=10)}

# Сохранение метрик
classMetricLogger = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLogger)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader('Сравнение метрик моделей')
# Метрики качества модели
metrics = classMetricLogger.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLogger.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее подобранное значение параметра регуляризации для логистической регрессии:')
params = {'C': np.logspace(1, 3, 20)}
grid_lr = GridSearchCV(estimator=LogisticRegression(),
                       param_grid=params,
                       cv=3,
                       n_jobs=-1)
grid_lr.fit(X_train, Y_train)
st.write(grid_lr.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'LogR': LogisticRegression(C=cs_1),
          'LogRGrid': grid_lr.best_estimator_}

# Сохранение метрик
classMetricLoggerLogR = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerLogR)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerLogR.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerLogR.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение количества ближайших соседей для модели ближайших соседей:')
params = {'n_neighbors': list(range(5, 100, 5))}
grid_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_knn.fit(X_train, Y_train)
st.write(grid_knn.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'KNN': KNeighborsClassifier(n_neighbors=n_estimators_2),
          'KNNGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerKNN = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerKNN)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerKNN.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerKNN.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение параметра регуляризации для SVC модели')
params = {'C': np.logspace(1, 3, 20)}
grid_svc = GridSearchCV(estimator=SVC(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_svc.fit(X_train, Y_train)
st.write(grid_svc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'SVC': SVC(C=cs_3, probability=True),
          'SVCGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerSVC = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerSVC)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerSVC.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerSVC.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение максимальной глубины для дерева решений:')
params = {'max_depth': list(range(5, 500, 10))}
grid_dtc = GridSearchCV(estimator=DecisionTreeClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_dtc.fit(X_train, Y_train)
st.write(grid_dtc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'Tree': DecisionTreeClassifier(max_depth=max_depth_4, random_state=10),
          'TreeGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerTree = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerTree)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerTree.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerTree.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение количества фолдов для случайного леса:')
params = {'n_estimators': list(range(5, 200, 10))}
grid_rfc = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_rfc.fit(X_train, Y_train)
st.write(grid_rfc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'RF': RandomForestClassifier(n_estimators=n_estimators_5, oob_score=True, random_state=10),
          'RFGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerRF = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerRF)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerRF.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerRF.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение количества фолдов для градиентного бустинга:')
params = {'n_estimators': list(range(5, 200, 10))}
grid_gbc = GridSearchCV(estimator=GradientBoostingClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_gbc.fit(X_train, Y_train)
st.write(grid_gbc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'GB': GradientBoostingClassifier(n_estimators=n_estimators_6, random_state=10),
          'GBGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerGB = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerGB)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerGB.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerGB.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Сравнение метрик для всех моделей')
metrics = classMetricLoggerGB.df['metric'].unique()
# Построим графики метрик качества всех моделей
for metric in metrics:
    st.pyplot(classMetricLogger.plot('Метрика: ' + metric, metric, figsize=(7, 6)))
