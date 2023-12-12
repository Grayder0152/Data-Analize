from abc import ABC, abstractmethod

from enum import Enum

import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


class LinkageMethodName(Enum):
    ward: str = 'ward'


class LinkageMethod(ABC):
    @property
    @abstractmethod
    def name(self):
        pass


class WardLinkage(LinkageMethod):
    name = LinkageMethodName.ward

    def linkage(self, data):
        clusters = [[i.person_id] for i in data.select('person_id').collect()]

        while len(clusters) > 1:
            # Calculate the variance for each cluster
            variances = [np.var(data[cluster], axis=0) for cluster in clusters]

            # Calculate the pairwise squared Euclidean distances between cluster centroids
            distances = np.zeros((len(clusters), len(clusters)))
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    centroid_diff = np.mean(data[clusters[i]], axis=0) - np.mean(data[clusters[j]], axis=0)
                    distances[i, j] = np.sum(centroid_diff ** 2)
                    distances[j, i] = distances[i, j]

            # Find the indices of the two clusters with the smallest distance
            min_i, min_j = np.unravel_index(np.argmin(distances), distances.shape)

            # Merge the two clusters
            clusters[min_i] += clusters[min_j]
            del clusters[min_j]

        return clusters[0]

    @staticmethod
    def ward_linkage(data):
        # variance_udf = spark.udf.register("variance_udf", lambda x: float(np.var(x.toArray())))
        centroids_udf = spark.udf.register("centroids_udf", lambda x, y: float(np.sum((x.toArray() - y.toArray()) ** 2)))
        df = data.withColumn("cluster", col("features"))

        while df.count() > 1:
            # Calculate the variance for each cluster
            # df = df.withColumn("variance", variance_udf(col("cluster")))

            # Calculate the pairwise squared Euclidean distances between cluster centroids

            df = df.withColumnRenamed("cluster", "cluster_1").withColumnRenamed("person_id", "person_id_1").crossJoin(
                df.withColumnRenamed("cluster", "cluster_2").withColumnRenamed("person_id", "person_id_2"))
            df = df.withColumn("centroid_distance", centroids_udf(col("cluster_1"), col("cluster_2")))
            df = df.filter("cluster_1 != cluster_2")

            # Find the indices of the two clusters with the smallest distance
            min_distance_row = df.orderBy("centroid_distance").first()
            min_cluster_1 = min_distance_row.person_id_1
            min_cluster_2 = min_distance_row.person_id_2

            # Merge the two clusters
            df = df.filter("person_id_1 != '{}' AND person_id_2 != '{}'".format(min_cluster_1, min_cluster_2))
            merged_cluster = df.union(min_cluster_1.union(min_cluster_2)).distinct()
            df = df.union(merged_cluster)

        # Show the final cluster
        result = df.select("cluster").distinct().collect()
        return result


if __name__ == '__main__':
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("K-mean")
        .getOrCreate()
    )

    assembler = VectorAssembler(inputCols=['t32', 't33'], outputCol="features")
    data = (
        spark.read.csv('../../data/data.csv', header=True, inferSchema=True)
        .withColumnRenamed('Напряжение углекислого газа (PCO2)', 'person_id')
    )
    data = assembler.transform(data).select('person_id', 'features')
    WardLinkage.ward_linkage(data)
