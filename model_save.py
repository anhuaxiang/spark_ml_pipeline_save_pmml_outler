import os
import shutil
import pyspark.sql.functions as fn
from pyspark.sql.types import IntegerType
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator


MODEL_SAVE_PATH = 'pipeline'


spark = SparkSession.builder \
    .master('local') \
    .appName("Word Count") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.read.csv('file:///D:/workspace/sample/multiclass/breast_cancer.csv', header=True)
columns = df.columns
columns.remove('id')
df = df.select(columns)

feature_cols = columns.copy()
label = 'class'
feature_cols.remove(label)


transform_f = 'mitoses'
value_count_df = df.groupBy(transform_f).count()
value_counts = {
    str(v[0]): int(v[1]) for _, v in value_count_df.limit(100).toPandas().dropna()[[transform_f, 'count']].iterrows()
}
col_map = {v[0]: i for i, v in enumerate(sorted(tuple(value_counts.items()), key=lambda x: x[1], reverse=True))}
df = df.withColumn(transform_f, fn.udf(lambda x: col_map.get(x), IntegerType())(df[transform_f]))


for f, d in df.dtypes:
    if d == 'string':
        df = df.withColumn(f, df[f].cast('int'))
    if f == 'class':
        df = df.withColumn(f, df[f].cast('string'))
df = df.dropna()

train, test = df.randomSplit([0.8, 0.2], seed=0)

class_index = StringIndexer(inputCol='class', outputCol='label')
vector = VectorAssembler(inputCols=feature_cols, outputCol='feature')
model = LinearSVC(featuresCol='feature', labelCol='label')
pipeline = Pipeline(stages=[class_index, vector, model])

pipeline = pipeline.fit(train)
if os.path.exists(MODEL_SAVE_PATH):
    shutil.rmtree(MODEL_SAVE_PATH)
pipeline.write().overwrite().save(pipeline)  # pipeline.save('/to/path')


load_pipeline = PipelineModel.load('pipeline')
test_predict = load_pipeline.transform(test)


evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='rawPrediction',
    labelCol='label'
)

print(evaluator.evaluate(test_predict, {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(test_predict, {evaluator.metricName: 'areaUnderPR'}))

origin_test_df = df.select(feature_cols)

predict_df = load_pipeline.transform(origin_test_df)
print(predict_df.show(20))


from pyspark2pmml import PMMLBuilder

sc = spark.sparkContext.getOrCreate()
pmmlBuilder = PMMLBuilder(sc, df, load_pipeline).putOption(load_pipeline.stages[-1], "compact", True)
pmmlBuilder.buildFile("LinearSVC.pmml")