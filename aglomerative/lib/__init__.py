import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import rand_score
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')


def clustering(data):
    figure, axis = plt.subplots(2, 2)
    figure.suptitle("Agglomerative Clustering")

    scaler = StandardScaler()
    new_data = scaler.fit_transform(data)
    rand_index = {}
    linkages = ('ward', 'complete', 'average', 'single')
    for linkage, ax in zip(linkages, axis.ravel()):
        agg = AgglomerativeClustering(linkage=linkage)
        model = agg.fit(new_data)
        clusters = model.fit_predict(new_data)
        ax.set_xlabel("t33")
        ax.set_ylabel("t34")
        ax.set_title(linkage.title())
        ax.scatter(data['t33'], data['t34'], c=clusters, s=50)
        rand_index[linkage] = list(clusters)

    d = []
    for method, clustering_1 in rand_index.items():
        d.append([method] + [float(rand_score(clustering_1, clustering_2)) for clustering_2 in rand_index.values()])
    rand_index = spark.createDataFrame(d, schema=['method'] + list(rand_index.keys()))

    plt.show()


def plot_dendrogram(data, **kwargs):
    scaler = StandardScaler()
    new_data = scaler.fit_transform(data)

    agg = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = agg.fit(new_data)

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


if __name__ == '__main__':
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Agglomerative")
        .getOrCreate()
    )

    data = (
        spark.read.csv('../../data/data.csv', header=True, inferSchema=True)
        .withColumnRenamed('Напряжение углекислого газа (PCO2)', 'person_id')
        .select('t33', 't34')
        # .drop('person_id')
    ).toPandas()
    clustering(data)
    # plot_dendrogram(data, truncate_mode="level", p=3)
