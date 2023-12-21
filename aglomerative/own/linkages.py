from abc import ABC, abstractmethod

from enum import Enum

import numpy as np
import pyspark.sql.functions as f

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector

from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, DoubleType


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
        pass


if __name__ == '__main__':
    import time

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("K-mean")
        .config("spark.sql.execution.arrow.pyspark.enabled", True)
        .getOrCreate()
    )

    assembler = VectorAssembler(inputCols=['t32', 't33'], outputCol="features")
    data = (
        spark.read.csv('../../data/data.csv', header=True, inferSchema=True)
        .withColumnRenamed('Напряжение углекислого газа (PCO2)', 'cluster_id')
    )
    data = assembler.transform(data).select('cluster_id', 'features')
    data = data.limit(20).groupBy('cluster_id').agg(f.collect_list('features').alias('points'))


    @f.udf(returnType=ArrayType(DoubleType()))
    def calc_centroids(vectors):
        np_vectors = np.array([v.toArray() for v in vectors])
        centroid = np.mean(np_vectors, axis=0)
        return centroid.tolist()


    @f.udf(returnType=DoubleType())
    def sum_of_distances_to_centroid(vectors, centroid):
        centroid = np.array(centroid)
        points = np.array([v.toArray() for v in vectors])
        distances = np.sqrt(((points - centroid) ** 2).sum(axis=1))
        return float(distances.sum())


    i = data.count()
    while i != 3:
        print(f"{i} iteration")
        start = time.time()
        cached_data = data.cache()
        cached_data = cached_data.withColumn('centroid', calc_centroids('points'))
        cached_data = cached_data.withColumn('dist', sum_of_distances_to_centroid('points', 'centroid'))

        df1 = cached_data.select(f.col('cluster_id').alias('cluster_id_1'), f.col('points').alias('points_1'),
                                 f.col('dist'))
        df2 = cached_data.select(f.col('cluster_id').alias('cluster_id_2'), f.col('points').alias('points_2'))

        df = (
            df1
            .join(df2, f.col('cluster_id_1') != f.col('cluster_id_2'), 'cross')
            .withColumn("sorted_ids", f.array_sort(f.array(f.col("cluster_id_1"), f.col("cluster_id_2"))))
            .dropDuplicates(["sorted_ids"])
            .withColumn('points', f.concat('points_1', 'points_2'))
            .withColumn('centroid', calc_centroids('points'))
            .withColumn('new_dist', sum_of_distances_to_centroid('points', 'centroid'))
            .withColumn('diff', f.col('new_dist') - f.col('dist'))
            .orderBy('diff')
        )
        new_centroid = df.select('cluster_id_1', 'cluster_id_2', 'points').first()
        min_cluster_id, max_cluster_id = sorted([new_centroid.cluster_id_1, new_centroid.cluster_id_2])
        new_points = f.array(*[array_to_vector(f.lit(vector.toArray())) for vector in new_centroid.points])
        data = (
            data
            .where(f.col('cluster_id') != max_cluster_id)
            .withColumn('points', f.when(
                f.col('cluster_id') == min_cluster_id, new_points
            ).otherwise(f.col('points')))
        )

        cached_data.unpersist()
        i -= 1
        print(time.time() - start)

