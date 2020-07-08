import os
import shutil
import pyspark.sql.functions as fn
from pyspark.sql.types import IntegerType
from pyspark.sql.session import SparkSession
from py4j.java_gateway import JavaClass
from pyspark.ml.common import _py2java
from pyspark.ml.classification import LinearSVC
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

MODEL_SAVE_PATH = 'isolation'

spark = SparkSession.builder \
    .master('local') \
    .appName("Word Count") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sc = spark.sparkContext.getOrCreate()

df = spark.read.csv('shuttle.csv.txt', header=False, inferSchema=True, comment='#')
cols = df.columns
label_col = cols[-1]
cols.remove(label_col)

assembler = VectorAssembler(inputCols=cols, outputCol='features')
data = assembler.transform(df).select(['features', label_col]).withColumnRenamed(label_col, 'label')
data_java = _py2java(sc, data)

iso_class = sc._jvm.com.linkedin.relevance.isolationforest.IsolationForest
isolation = iso_class() \
    .setNumEstimators(100) \
    .setBootstrap(False) \
    .setMaxFeatures(1.0) \
    .setFeaturesCol("features") \
    .setPredictionCol("predictedLabel") \
    .setScoreCol("outlierScore") \
    .setContamination(0.1) \
    .setContaminationError(0.01 * 0.1) \
    .setRandomSeed(1)

isolation_model = isolation.fit(data_java)
data_with_score = isolation_model.transform(data_java)

if os.path.exists(MODEL_SAVE_PATH):
    shutil.rmtree(MODEL_SAVE_PATH)
isolation_model.write().overwrite().save(MODEL_SAVE_PATH)


print(data_with_score.show())


from pyspark2pmml import PMMLBuilder

sc = spark.sparkContext.getOrCreate()
pmmlBuilder = PMMLBuilder(sc, df, isolation_model).putOption(isolation_model, "compact", True)
pmmlBuilder.buildFile("isolation.pmml")