import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# def plot_embedding(X, title=None):
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#     plt.figure()
#     ax = plt.subplot(111)
#     for i in range(X.shape[0]):
#         plt.text(X[i, 0], X[i, 1], str(y_test[i]),
#                  color=plt.cm.Set1(y_test[i]))
#     plt.xticks([]), plt.yticks([])
#     if title is not None:
#         plt.title(title)


def plot_tsne(features, labels):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    import pandas as pd
    tsne = TSNE(n_components=1, init='pca', perplexity=99, learning_rate=200, n_iter=10000)
    import seaborn as sns

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化
    plt.show()
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("viridis", class_num),
                    data=df).set(title="T-SNE plot of original features")
    plt.savefig('image.png')



if __name__ == '__main__':

    plot_tsne(results, train_y[:1000])
