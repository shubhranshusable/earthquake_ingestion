import os
import json
from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, LongType, IntegerType, ArrayType
from pyspark.sql.functions import from_unixtime,split,col
if __name__ == '__main__':
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
    #     r"D:\gcp\metal-center-432709-h7-d1bb5a377b3a.json"
    # )
    def fetch_json_from_gcs(bucket_name, blob_name):
        """
        Fetch JSON data from GCS bucket.
        :param bucket_name: Name of the GCS bucket.
        :param blob_name: Name of the blob in the bucket.
        :return: Parsed JSON data.
        """
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = json.loads(blob.download_as_text())
        return data


    def done():
        # Set up the Spark session
        spark = SparkSession.builder.appName("Earthquake_Data_Fetch")\
            .config("spark.executor.memory", "4g")\
            .config("spark.driver.memory", "4g")\
            .config("spark.sql.debug.maxToStringFields", "100") \
            .getOrCreate()
            # .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
            # .config("spark.jars", "gs://hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar")\
            # .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.5") \
        
        
        # spark.sparkContext.setLogLevel("DEBUG")


        # Define GCS bucket and blob names
        bucket_name = "kraken_2"
        blob_name = "raw/RAW_DATA_20241021.json"

        # Fetch data from GCS
        data = fetch_json_from_gcs(bucket_name, blob_name)

        # Process and convert values
        for feature in data['features']:
            # Convert relevant fields to float
            feature['properties']['mag'] = float(feature['properties']['mag']) if feature['properties'][
                                                                                      'mag'] is not None else None
            feature['properties']['gap'] = float(feature['properties']['gap']) if feature['properties'][
                                                                                      'gap'] is not None else None
            feature['properties']['dmin'] = float(feature['properties']['dmin']) if feature['properties'][
                                                                                        'dmin'] is not None else None
            feature['properties']['rms'] = float(feature['properties']['rms']) if feature['properties'][
                                                                                      'rms'] is not None else None

            # Convert geometry coordinates to float
            feature['geometry']['coordinates'] = [float(coord) for coord in feature['geometry']['coordinates']] if \
            feature['geometry']['coordinates'] is not None else []

        # Define the schema
        schema = StructType([
            StructField("type", StringType(), True),
            StructField("properties", StructType([
                StructField("mag", FloatType(), True),
                StructField("place", StringType(), True),
                StructField("time", LongType(), True),
                StructField("updated", LongType(), True),
                StructField("tz", StringType(), True),
                StructField("url", StringType(), True),
                StructField("detail", StringType(), True),
                StructField("felt", StringType(), True),
                StructField("cdi", StringType(), True),
                StructField("mmi", StringType(), True),
                StructField("alert", StringType(), True),
                StructField("status", StringType(), True),
                StructField("tsunami", IntegerType(), True),
                StructField("sig", IntegerType(), True),
                StructField("net", StringType(), True),
                StructField("code", StringType(), True),
                StructField("ids", StringType(), True),
                StructField("sources", StringType(), True),
                StructField("types", StringType(), True),
                StructField("nst", IntegerType(), True),
                StructField("dmin", FloatType(), True),
                StructField("rms", FloatType(), True),
                StructField("gap", FloatType(), True),
                StructField("magType", StringType(), True),
                StructField("title", StringType(), True)
            ]), True),
            StructField("geometry", StructType([
                StructField("type", StringType(), True),
                StructField("coordinates", ArrayType(FloatType()), True)
            ]), True),
            StructField("id", StringType(), True)
        ])

        # Create DataFrame from the modified data
        df = spark.createDataFrame(data['features'], schema=schema)

        # Flatten the DataFrame
        flattened_df = df.select(
            "properties.mag",
            "properties.place",
            "properties.time",
            "properties.updated",
            "properties.tz",
            "properties.url",
            "properties.detail",
            "properties.felt",
            "properties.cdi",
            "properties.mmi",
            "properties.alert",
            "properties.status",
            "properties.tsunami",
            "properties.sig",
            "properties.net",
            "properties.code",
            "properties.ids",
            "properties.sources",
            "properties.types",
            "properties.nst",
            "properties.dmin",
            "properties.rms",
            "properties.gap",
            "properties.magType",
            "properties.title",
            "geometry.coordinates"
        )

        # Accessing the coordinates properly using selectExpr or a similar method
        flattened_df = flattened_df.withColumn("longitude", flattened_df["coordinates"].getItem(0)) \
            .withColumn("latitude", flattened_df["coordinates"].getItem(1)) \
            .withColumn("depth", flattened_df["coordinates"].getItem(2))

        # Drop the original coordinates and geometry columns if not needed
        #flattened_df = flattened_df.drop("coordinates")

        # Show the flattened DataFrame
        fdf=flattened_df.withColumn("time",from_unixtime(col("time")/1000).cast("timestamp"))\
                    .withColumn("updated",from_unixtime(col("updated")/1000).cast("timestamp"))\
                    .withColumn("place",split(col("place"),"of").getItem(1))\
                    
        df.write.mode('overwrite').json("gs://kraken_2/silver")
        # Stop the Spark session
        spark.stop()

    done()
