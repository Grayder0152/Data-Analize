import random

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import pyspark.sql.functions as f

from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType
from scipy.spatial import distance


class CentroidMethodName(Enum):
    K_MEAN_PP = 'k-mean++'


class CentroidMethod(ABC):
    name: Optional[CentroidMethodName] = None

    def __init__(self, data: DataFrame, k: int, pk_col_name: str):
        self.data: DataFrame = data
        self.k: int = k
        self.pk_col_name = pk_col_name

    @abstractmethod
    def get_centroids(self, params: list[str]) -> DataFrame:
        pass


class KMeanPp(CentroidMethod):
    name = CentroidMethodName.K_MEAN_PP

    def euclidean_distance(self, params: list[str]) -> DataFrame:
        self.data = self.data.withColumn('coords', f.array(*params))
        df = (
            self.data
            .select(f.col(self.pk_col_name).alias('point_id'), 'coords')
            .withColumn('point_id', f.concat(f.lit('point_'), f.col('point_id')))
        )

        distance_udf = f.udf(lambda point_1, point_2: float(distance.euclidean(point_1, point_2)), FloatType())

        return (
            df.alias('df1')
            .crossJoin(df.alias('df2'))
            .groupBy('df1.point_id')
            .pivot('df2.point_id')
            .agg(
                distance_udf(f.first('df1.coords'), f.first('df2.coords'))
            )
        )

    def get_centroids(self, params: list[str]) -> DataFrame:
        euclidean_distance_df = self.euclidean_distance(params)
        euclidean_distance_df = euclidean_distance_df.toPandas().set_index('point_id').T.to_dict('dict')

        centroids = [random.choice(list(euclidean_distance_df.keys())).split('_')[-1]]
        for _ in range(1, self.k):
            distances = {}
            chosen_points = [f"point_{id_}" for id_ in centroids]

            for point_id, point_distances in euclidean_distance_df.items():
                if point_id not in chosen_points:
                    distances[point_id] = min(point_distances[point_id] for point_id in chosen_points)
            centroid_point_id = max(distances, key=distances.get)
            centroids.append(centroid_point_id.split('_')[-1])
        return self.data.where(f.col(self.pk_col_name).isin(centroids))
