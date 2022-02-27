from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Example').getOrCreate()

print(spark)

rdd=spark.sparkContext.parallelize([1,2,3,4,5,6])

print("RDD count"+rdd.count())