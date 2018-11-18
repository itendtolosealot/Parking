import org.apache.spark._


val inputdf  = spark.read.format("CSV").option("header","true").load("/home/ashvin/data.csv")

val processedDf = inputdf.drop("Date Time").groupBy("Elmntkey", "Study_Area", "UnitDesc","Day of Week", "Time").agg(sum("Parking_Spaces"), sum("Total_Vehicle_Count")).filter("sum(Parking_Spaces)>10")
val showData = processedDf.show()
//val summary = processedDf.summary().show()
//val summary = processedDf.count()
processedDf.coalesce(1).write.option("header","true").format("com.databricks.spark.csv").save("/home/ashvin/filter.csv")


