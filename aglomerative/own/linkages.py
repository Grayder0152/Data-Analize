# Databricks notebook source
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd
import pyspark.sql.functions as f

from pyspark.ml.functions import array_to_vector

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, DoubleType

# COMMAND ----------

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

# COMMAND ----------

class LinkageMethodName(Enum):
    ward: str = 'ward'


class LinkageMethod(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

# COMMAND ----------

class WardLinkage(LinkageMethod):
    name = LinkageMethodName.ward

    def linkage(self, data: pd.DataFrame, k: int = 0) -> DataFrame:
        max_iteration = len(data)
        while max_iteration != k:
            df = data.assign(
                dist=lambda x: [((p - np.mean(p, axis=0)) ** 2).sum(axis=0).sum() for p in data.points]
            )
            df = pd.merge(df, df, how='cross', suffixes=('_1', '_2'))
            df = df[df.cluster_id_1 != df.cluster_id_2]
            df['sorted_tuple'] = df.apply(lambda row: tuple(sorted((row['cluster_id_1'], row['cluster_id_2']))), axis=1)
            df = df.drop_duplicates(subset='sorted_tuple').drop(columns='sorted_tuple')
            df['points'] = df.apply(lambda row: [*row['points_1'], *row['points_2']], axis=1)
            df = df.assign(
                new_dist=lambda x: [((p - np.mean(p, axis=0)) ** 2).sum(axis=0).sum() for p in df.points],
            )
            # df['diff'] = df['new_dist'] - df['dist']
            df['diff'] = df['new_dist'] - df['dist_1'] - df['dist_2']
            new_centroid = df.loc[df['diff'].idxmin()][['cluster_id_1', 'cluster_id_2', 'points']]
            min_cluster_id, max_cluster_id = sorted([new_centroid.cluster_id_1, new_centroid.cluster_id_2])
            data.loc[data['cluster_id'] == min_cluster_id, 'points'] = pd.Series([new_centroid.points], index=data[
                data['cluster_id'] == min_cluster_id].index.values)
            data = data[data['cluster_id'] != max_cluster_id]
            max_iteration -= 1
        return data
