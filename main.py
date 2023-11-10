from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

import pyspark.sql.functions as f
import pandas as pd
from matplotlib.axes import Axes
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.evaluation import ClusteringEvaluator

from k_mean.base import BaseKMean, KMeantType
from k_mean.lib import KMean as LibKMean
from k_mean.own import KMean as OwnKMean

matplotlib.use('TkAgg')


def find_best_k(data: DataFrame, pk_col_name: str, params: list[str], min_k: int = 2, max_k: int = 10):
    silhouette_score = []
    prediction_col = 'cluster'
    evaluator = ClusteringEvaluator(
        predictionCol=prediction_col,
        featuresCol='scaled_data',
        metricName='silhouette',
        distanceMeasure='squaredEuclidean'
    )
    for k in range(min_k, max_k):
        kmeans = LibKMean(data=data, k=k, pk_col_name=pk_col_name)
        kmeans.clustering(params)
        score = evaluator.evaluate(kmeans.clustered_data)
        silhouette_score.append(score)
        print(f'Silhouette Score for k = {k} is {score}')

    plt.plot(list(range(min_k, max_k)), silhouette_score)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Silhouette Score')
    plt.show()


class KMeanManager:
    def __init__(self, data: DataFrame, k: int, pk_col_name: str):
        self.k_mean_classes: dict[KMeantType, BaseKMean] = {
            KMeantType.OWN: OwnKMean(data=data, k=k, pk_col_name=pk_col_name),
            KMeantType.LIB: LibKMean(data=data, k=k, pk_col_name=pk_col_name)
        }

        self.current_k_mean: Optional[BaseKMean] = None

    def clustering(self, k_mean_method: str, params: list[str]):
        self.current_k_mean = self.k_mean_classes[KMeantType(k_mean_method)]
        self.current_k_mean.clustering(params)

    def _parallel_coordinates(self, columns: list[str], clustered_data_pd: pd.DataFrame, axis: list[Axes]):
        df = clustered_data_pd[::]
        df[self.current_k_mean.cluster_col_name] = 0
        pd.plotting.parallel_coordinates(
            df[columns + [self.current_k_mean.cluster_col_name]],
            class_column=self.current_k_mean.cluster_col_name,
            ax=axis[0]
        )
        axis[0].legend().remove()

        pd.plotting.parallel_coordinates(
            clustered_data_pd[columns + [self.current_k_mean.cluster_col_name]],
            class_column=self.current_k_mean.cluster_col_name,
            ax=axis[1]
        )
        axis[1].legend(title=self.current_k_mean.cluster_col_name)
        plt.show()

    def _dot_coordinates(
            self, param_1: str, param_2: str, clustered_data_pd: pd.DataFrame,
            axis: list[Axes], with_centroids: bool
    ):
        axis[0].scatter(
            clustered_data_pd[param_1],
            clustered_data_pd[param_2],
        )
        axis[0].set_xlabel(param_1)
        axis[0].set_ylabel(param_2)

        sp = axis[1].scatter(
            clustered_data_pd[param_1],
            clustered_data_pd[param_2],
            c=clustered_data_pd[self.current_k_mean.cluster_col_name],
        )
        # axis[1].legend(sp.legend_elements()[0], list(range(self.k)), title='Clusters')
        axis[1].legend(*sp.legend_elements(), title='Clusters')
        axis[1].set_xlabel(param_1)
        axis[1].set_ylabel(param_2)
        if with_centroids:
            centroids = self.current_k_mean.centroids.toPandas()
            axis[1].scatter(centroids[param_1], centroids[param_2], s=70)

        plt.show()

    def show(self, with_centroids: bool = False):
        clustered_data_pd = self.current_k_mean.clustered_data.toPandas()
        figure, axis = plt.subplots(2, 1)
        figure.suptitle('Clustering by KMean')
        axis[0].set_title('Before clustering')
        axis[1].set_title('After clustering')

        if len(self.current_k_mean.params) == 2:
            param_1, param_2 = self.current_k_mean.params
            self._dot_coordinates(param_1, param_2, clustered_data_pd, axis, with_centroids)
        else:
            self._parallel_coordinates(self.current_k_mean.params, clustered_data_pd, axis)

        plt.show()

    def get_clusters_statistic(self) -> DataFrame:
        return (
            self.current_k_mean.clustered_data
            .select(*self.current_k_mean.params, self.current_k_mean.cluster_col_name)
            .unpivot(self.current_k_mean.cluster_col_name, self.current_k_mean.params, "param", "value")
            .groupBy(self.current_k_mean.cluster_col_name)
            .agg(
                f.mean("value").alias("mean"),
                f.stddev("value").alias("se")
            )
            .orderBy(self.current_k_mean.cluster_col_name)
        )


if __name__ == '__main__':
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("K-mean")
        .getOrCreate()
    )

    data = (
        spark.read.csv('data/data.csv', header=True, inferSchema=True)
        .withColumnRenamed('Напряжение углекислого газа (PCO2)', 'person_id')
    )

    k_mean = KMeanManager(
        data=data,
        k=3,
        pk_col_name='person_id'
    )
    k_mean.clustering('own', ['t32', 't33'])
    k_mean.show(with_centroids=True)
    spark.stop()
