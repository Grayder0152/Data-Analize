from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession

from k_mean.base import BaseKMean

spark = (
    SparkSession.builder.master("local[*]")
    .appName("K-mean")
    .getOrCreate()
)


class KMean(BaseKMean):
    def clustering(self, params: list[str]) -> None:
        self.params = params
        kmeans = KMeans(featuresCol='scaled_data', predictionCol=self.cluster_col_name, k=self.k)

        data = VectorAssembler(inputCols=params, outputCol='vector_data').transform(self.data)

        scaler = StandardScaler(
            inputCol="vector_data",
            outputCol="scaled_data",
            withStd=True,
            withMean=False
        )
        scaler_model = scaler.fit(data)
        data = scaler_model.transform(data)
        model = kmeans.fit(data)

        self.centroids = spark.createDataFrame(
            [[id_, *c.tolist()] for id_, c in enumerate(model.clusterCenters())],
            schema=[self.cluster_col_name, *params]
        )
        self.clustered_data = model.transform(data)
