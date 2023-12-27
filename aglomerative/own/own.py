# Databricks notebook source
# MAGIC %run ./linkages

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f

from pyspark.ml.feature import VectorAssembler

class HierarchicalClustering:
    pass


class AgglomerativeClustering(HierarchicalClustering):
    def __init__(self):
        self.linkagess = {
            WardLinkage.name: WardLinkage()
        }
        
    def clustering(self, data, k: int, linkage: str = 'ward'):
        if "cluster_id" in data.columns:
            points_name = data.columns[1:]
        else:
            points_name = data.columns
            data = data.withColumn("cluster_id", f.monotonically_increasing_id())
            
        assembler = VectorAssembler(inputCols=points_name, outputCol="features")
        data = assembler.transform(data).select('cluster_id', 'features')
        data = data.groupBy('cluster_id').agg(f.collect_list('features').alias('points'))
        clusters = self.linkagess[LinkageMethodName(linkage)].linkage(data.toPandas(), k)
        clusters = clusters.explode('points')
        clusters[points_name] = clusters['points'].apply(pd.Series)
        clusters = clusters.drop(columns='points').reset_index(drop=True)
        return clusters
