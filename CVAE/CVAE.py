from cvae import cvae
import numpy as np
import pandas as pd
from sklearn import svm, model_selection, metrics

if __name__ == '__main__':
    X_features = pd.read_csv(r"D:\Google_Download\DS_Basics\Assignments\Assignment1\Dataset\features\ResNet101\AwA2"
                             r"-features.txt", header=None, sep=' ')
    Y_labels = pd.read_csv(r"D:\Google_Download\DS_Basics\Assignments\Assignment1\Dataset\features\ResNet101\AwA2"
                           r"-labels.txt", header=None, sep=' ')
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_features, Y_labels, test_size=0.4,
                                                                        random_state=1234, stratify=Y_labels)
    X_features=X_features.to_numpy()
    embedder = cvae.CompressionVAE(X_features, dim_latent=10, batch_size=128, train_valid_split=0.9)
    embedder.train()
    z_train = embedder.embed(X_train)
    z_test = embedder.embed(X_test)

    VAE_linear_svc = svm.LinearSVC()
    VAE_linear_svc.fit(z_train, Y_train)
    VAE_pred_linear_svc = VAE_linear_svc.predict(z_test)
    print(metrics.accuracy_score(Y_test, VAE_pred_linear_svc))
    print(np.shape(z_train))
    print(np.shape(z_test))
