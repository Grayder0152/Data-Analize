from typing import Type

import pyspark.sql.functions as f

from pyspark.pandas import DataFrame
from pyspark.sql import Window
from pyspark.sql.types import FloatType
from scipy.spatial import distance

from k_mean.base import BaseKMean
from k_mean.own.centroid_methods import CentroidMethod, CentroidMethodName, KMeanPp


class KMean(BaseKMean):
    centroid_methods: dict[CentroidMethodName, Type[CentroidMethod]] = {
        KMeanPp.name: KMeanPp
    }

    def __init__(
            self, data: DataFrame, k: int,
            pk_col_name: str, centroid_method: str = 'k-mean++'
    ):
        super().__init__(data, k, pk_col_name)
        self.centroid_method = centroid_method

    def get_new_centroids(self, params: list[str]) -> DataFrame:
        return (
            self.clustered_data
            .groupBy(self.cluster_col_name)
            .agg(
                *[f.mean(param).alias(param) for param in params]
            )
        )

    def check_centroids_eq(self, new_centroids: DataFrame) -> bool:
        old_centroids = self.centroids.sort(self.cluster_col_name).collect()
        new_centroids = new_centroids.sort(self.cluster_col_name).collect()
        for i in range(self.k):
            if old_centroids[i] != new_centroids[i]:
                return False
        return True

    def clustering(self, params: list[str]):
        self.params = params
        distance_udf = f.udf(lambda point_1, point_2: float(distance.euclidean(point_1, point_2)), FloatType())
        window = Window.partitionBy(self.pk_col_name).orderBy('distance')

        centroid_method: CentroidMethod = self.centroid_methods[CentroidMethodName(self.centroid_method)](
            self.data, self.k, self.pk_col_name
        )
        self.centroids = (
            centroid_method.get_centroids(params)
            .select(f.monotonically_increasing_id().alias(self.cluster_col_name), *params)
        )

        while True:
            self.clustered_data = (
                self.data
                .crossJoin(
                    self.centroids
                    .withColumn('centroid_coords', f.array(*params))
                    .drop(*params)
                ).withColumn('coords', f.array(*params))
                .withColumn('distance', distance_udf('coords', 'centroid_coords'))
                .withColumn('row', f.row_number().over(window))
                .where(f.col('row') == 1)
                .select(self.pk_col_name, *params, self.cluster_col_name)
            )

            new_centroids = self.get_new_centroids(params)
            if self.check_centroids_eq(new_centroids):
                break
            self.centroids = new_centroids
