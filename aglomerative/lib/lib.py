# Databricks notebook source
import os
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

if int(os.environ.get('LOCAL_DEV', '1')):
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Agglomerative")
        .getOrCreate()
    )
    matplotlib.use('TkAgg')

# COMMAND ----------

class LibAgglomerativeClustering:
    @staticmethod
    def clustering(data, n_clusters: int, linkage: str, with_standartization: bool = False):
        if with_standartization:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        model = agg.fit(data)
        clusters = model.fit_predict(data)
        data.insert(0, "cluster_id", clusters, True)
        return data
    
    @staticmethod
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
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        dendrogram(linkage_matrix, **kwargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
    
    def clustering_all_linkage(self, data, n_clusters: int, with_standartization: bool = False) -> dict:
        linkages = ('ward', 'complete', 'average', 'single')
        linkages_clusters = {}
        for linkage in linkages:
            clusters = self.clustering(data.copy(), n_clusters, linkage, with_standartization)
            linkages_clusters[linkage] = clusters
            
        return linkages_clusters

    @staticmethod
    def display_clusters(data, col_1: str, col_2: str, linkage: str):
        data = data.sort_values([col_1, col_2])
        figure, axis = plt.subplots()
        axis.set_xlabel(col_1)
        axis.set_ylabel(col_2)
        axis.set_title(linkage.title())
        axis.scatter(data[col_1], data[col_2], c=data['cluster'], s=50)
        plt.show()

    @staticmethod
    def rand_index(linkages_clusters: dict):
        d = []
        for method, data in linkages_clusters.items():
            d.append([method] + [float(rand_score(data['cluster_id'].sort_values(), (data_2['cluster_id'].sort_values()))) for data_2 in linkages_clusters.values()])
        rand_index = spark.createDataFrame(d, schema=['method'] + list(linkages_clusters.keys()))
        return rand_index
        
