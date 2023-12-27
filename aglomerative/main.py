# Databricks notebook source
# MAGIC %md
# MAGIC # Set Up parameters

# COMMAND ----------

STORAGE_ACCOUNT_NAME = "univerdata"
CONTAINER = "data"
SAS_TOKEN = "sp=rl&st=2023-12-26T14:42:59Z&se=2024-12-26T22:42:59Z&spr=https&sv=2022-11-02&sr=c&sig=9vE5bZTt4doTdVmRlxwb6SaLCQh%2BMpFmGSr%2Fqp8OYUQ%3D"
spark.conf.set(f"fs.azure.sas.fixed.token.{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net", SAS_TOKEN)

# COMMAND ----------

# MAGIC %md
# MAGIC # Extract Data

# COMMAND ----------

import pyspark.sql.functions as f

FILE_NAME = "data_2.csv"
FILE_FORMAT = "csv"
DELIMITER = " "
PATH_TO_FILE = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/{FILE_NAME}"
PARAMS = (f.col('col1').alias('x').cast('float'), f.col('col2').alias('y').cast('float'))

data = (
    spark.read.format(FILE_FORMAT)
    .load(PATH_TO_FILE, header=True, delimiter=DELIMITER)
    # .withColumnRenamed('Напряжение углекислого газа (PCO2)', 'cluster_id')
    .select(*PARAMS)
)
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Lib realization

# COMMAND ----------

# MAGIC %run ./lib/lib

# COMMAND ----------

lib = LibAgglomerativeClustering()

# COMMAND ----------

data1 = data.drop('cluster_id').toPandas()

df = lib.clustering(data1, 3, 'ward')
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dendrogram

# COMMAND ----------

# MAGIC %md
# MAGIC Base on this dendrogramm we can deside that optimal number of clusters is 2 as the most optimal number of clusters in a dendrogram can often be determined by looking for the longest vertical line that isn't crossed by any extended horizontal line. This is known as the "largest gap" method and can suggest where to "cut" the dendrogram to define clusters. But I would like to use k=3 to test will be more complex.
# MAGIC

# COMMAND ----------

lib.plot_dendrogram(data1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ward vs Complete vs Average vs Single

# COMMAND ----------

data1 = data.drop('cluster_id').toPandas()

linkages_clusters = lib.clustering_all_linkage(data1, 2)

for linkage, cluster in linkages_clusters.items():
    print(f"{linkage.title()} linkage method: ")
    display(cluster)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rand Index

# COMMAND ----------

display(
    lib.rand_index(linkages_clusters)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Own realization

# COMMAND ----------

# MAGIC %run ./own/own

# COMMAND ----------

data2 = data
df2 = AgglomerativeClustering().clustering(data2, 3)
display(df2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Own vs Lib

# COMMAND ----------

display(
    lib.rand_index({'lib_ward': df, 'own_ward': df2})
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parallel coordinates

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

def plot_parallel_coordinates(df, class_column):
    if not pd.api.types.is_numeric_dtype(df[class_column]):
        df[class_column] = df[class_column].astype('category')

    parallel_coordinates(df, class_column=class_column, colormap=plt.get_cmap("Set2"))
    plt.title('Parallel Coordinates Plot')
    plt.xlabel('Dimensions')
    plt.ylabel('Values')
    plt.legend(title=class_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

plot_parallel_coordinates(df2, 'cluster_id')
