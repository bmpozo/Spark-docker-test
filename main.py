from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, TimestampType
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def apply_getGroups(df_input: DataFrame) -> DataFrame:
    w_1 = Window.partitionBy(F.col('userid')).orderBy(F.col('eventtime').desc()).rangeBetween(-1201,Window.currentRow)
    w_2 = Window.orderBy('unique_id')

    data_sessions = df_input.select(F.col('userid'),F.col('musicbrainz-track-id'),F.col('track-name'),F.col('timestamp'),F.col('eventtime')) \
        .withColumn('songs_20_min',F.count('musicbrainz-track-id').over(w_1)) \
        .withColumn('is_new_session',F.when(F.col('songs_20_min')==1,1).otherwise(0)) \
        .withColumn('unique_id',F.monotonically_increasing_id()) \
        .withColumn('group_session_id',F.sum('is_new_session').over(w_2))
    
    return data_sessions

def apply_top50(df_all: DataFrame) -> DataFrame: 
    w_3 = Window.orderBy(F.col('number_songs').desc())

    data_top50_sessions = df_all \
    .groupBy(F.col('userid'),F.col('group_session_id')) \
    .agg(F.count(F.col('track-name')) \
    .alias('number_songs')) \
    .withColumn('order_top_sessions',F.rank().over(w_3)) \
    .filter(F.col('order_top_sessions') <= 50)

    return data_top50_sessions

def  apply_top10(df_all: DataFrame, df_top50: DataFrame) -> DataFrame:
    w_4 = Window.orderBy(F.col('number_reps').desc())

    data_top10_song = df_all.join(df_top50, ['userid','group_session_id'])

    data_top10_song \
    .groupBy(F.col('musicbrainz-track-id'),F.col('track-name')) \
    .agg(F.count(F.col('unique_id')).alias('number_reps')) \
    .withColumn('order_top_songs',F.rank().over(w_4)) \
    .filter(F.col('order_top_songs') <= 10) \
    .select(F.col('track-name'),F.col('number_reps'),F.col('order_top_songs'))

    return data_top10_song


if __name__ == "__main__":
    # build spark session
    spark = SparkSession.builder.appName('Example').getOrCreate()

    schema = StructType() \
        .add("userid",StringType(),True) \
        .add("timestamp",TimestampType(),True) \
        .add("musicbrainz-artist-id",StringType(),True) \
        .add("artist-name",StringType(),True) \
        .add("musicbrainz-track-id",StringType(),True) \
        .add("track-name",StringType(),True)
    
    df = spark.read.options(header='False',schema=schema,delimiter='\t').csv("file:///tmp/spark_scripts/data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv")

    data = df.toDF('userid', 'timestamp', 'musicbrainz-artist-id','artist-name','musicbrainz-track-id','track-name') \
        .withColumn('eventtime',F.col('timestamp').astype('Timestamp').cast("long"))

    df_allgroups = apply_getGroups(data)

    top50_sessions = apply_top50(df_allgroups)
    top50_sessions.coalesce(1).write.format('com.databricks.spark.csv').save('file:///tmp/spark_scripts/output/data_top50_sessions.csv',header = 'true')

    top10_songs = apply_top10(df_allgroups,top50_sessions)
    top10_songs.coalesce(1).write.format('com.databricks.spark.csv').save('file:///tmp/spark_scripts/output/data_top10_songs.csv',header = 'true')