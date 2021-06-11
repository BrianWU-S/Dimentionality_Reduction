import sklearn
from sklearn import svm, metrics, model_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


def LDA_fit(dimension=49):
    lda_method = LinearDiscriminantAnalysis(n_components=dimension)
    lda_method.fit(X_train, Y_train)
    return lda_method


def LinearSVM_LDA(LDA_reduced_x_train, LDA_reduced_y_train, LDA_reduced_x_test, LDA_reduced_y_test):
    LDA_linear_svc = svm.LinearSVC()
    LDA_linear_svc.fit(LDA_reduced_x_train, LDA_reduced_y_train)
    LDA_pred_linear_svc = LDA_linear_svc.predict(LDA_reduced_x_test)
    return metrics.accuracy_score(LDA_reduced_y_test, LDA_pred_linear_svc)


def LDA_plotting():
    # LDA
    acc_list = []
    for dimension in range(4, 50, 5):
        print("Dimension:", dimension)
        LDA_method = LDA_fit(dimension)
        LDA_reduced_features = LDA_method.transform(X_features)
        LDA_reduced_x_train, LDA_reduced_x_test, LDA_reduced_y_train, LDA_reduced_y_test = model_selection.train_test_split(
            LDA_reduced_features, Y_labels, test_size=0.4, random_state=1234, stratify=Y_labels)
        acc = LinearSVM_LDA(LDA_reduced_x_train, LDA_reduced_y_train, LDA_reduced_x_test, LDA_reduced_y_test)
        acc_list.append(acc)
    print(acc_list)
    plt.figure(figsize=[20, 10])
    plt.plot(np.arange(4, 50, 5), acc_list)
    plt.xlabel("Dimension")
    plt.ylabel("Accuracy")
    plt.title("Linear Discriminant Analysis")
    plt.show()


def RandomForest():
    # random forest
    random_forest_model = RandomForestRegressor(random_state=1234, max_depth=5, n_estimators=500)
    S_train_x, S_test_x, S_train_y, S_test_y = model_selection.train_test_split(X_train, Y_train, test_size=0.1,
                                                                                random_state=1234, stratify=Y_test)
    random_forest_model.fit(S_test_x, S_test_y)
    original_features = X_features.columns
    importances = random_forest_model.feature_importances_
    return original_features, importances


def RandomForest_importance_plotting(importances, original_features):
    # sorted importance
    indices = np.argsort(importances)[-20:]  # we only show the top 20 features
    plt.figure(figsize=[20, 10])
    plt.title('Feature Importance', fontdict={'size': 18})
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [original_features[i] for i in indices], fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('Relative Importance', fontdict={'size': 18})
    plt.show()
    # accumulate sum
    arg = np.argsort(importances)
    accumulate_sum = []
    for i in range(50, 1000, 50):
        accumulate_sum.append(np.sum(importances[arg[-i:]]))
        print("Dimension:", i, "  Accumulate sum:", accumulate_sum[-1])
    plt.figure(figsize=[20, 10])
    plt.plot(range(50, 1000, 50), accumulate_sum)
    plt.xlabel("Number of sorted columns")
    plt.ylabel("Cumulative importance sum")
    plt.title("Column-wise and Cumulative importance sum")


def RandomForest_plotting(importances):
    acc_list = []
    for dimension in range(500, 1000, 100):
        indices = np.argsort(importances)[-dimension:]
        np.sum(importances[indices])
        RF_reduced_features = X_features.iloc[:, indices]
        RF_reduced_x_train, RF_reduced_x_test, RF_reduced_y_train, RF_reduced_y_test = model_selection.train_test_split(
            RF_reduced_features, Y_labels, test_size=0.4, random_state=1234, stratify=Y_labels)
        RF_linear_svc = svm.LinearSVC()
        RF_linear_svc.fit(RF_reduced_x_train, RF_reduced_y_train)
        RF_pred_linear_svc = RF_linear_svc.predict(RF_reduced_x_test)
        acc = metrics.accuracy_score(RF_reduced_y_test, RF_pred_linear_svc)
        acc_list.append(acc)
    print(acc_list)
    plt.plot(np.arange(500, 1000, 100), acc_list)
    plt.xlabel("Dimension")
    plt.ylabel("Accuracy")
    plt.title("Random Forest")
    plt.show()


if __name__ == '__main__':
    # read the data
    X_features = pd.read_csv(r"Dataset\features\ResNet101\AwA2-features.txt", header=None, sep=' ')
    Y_labels = pd.read_csv(r"Dataset\features\ResNet101\AwA2-labels.txt", header=None, sep=' ')
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_features, Y_labels, test_size=0.4,
                                                                        random_state=1234, stratify=Y_labels)
    LDA_plotting()

