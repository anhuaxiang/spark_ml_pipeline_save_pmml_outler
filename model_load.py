
import pyspark.sql.functions as fn
from pyspark.sql.types import IntegerType
from pyspark.sql.session import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


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


# 示例etl流程, label encoding
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


# 加载模型
load_pipeline = PipelineModel.load('file:///D:/python_test/spark_ml/pipeline')
test_predict = load_pipeline.transform(df)
evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='rawPrediction',
    labelCol='label'
)

print(evaluator.evaluate(test_predict, {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(test_predict, {evaluator.metricName: 'areaUnderPR'}))

origin_test_df = df.select(feature_cols)

predict_df = load_pipeline.transform(origin_test_df)
print(predict_df.show(20))



