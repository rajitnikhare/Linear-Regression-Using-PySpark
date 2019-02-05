#!/usr/bin/env python
# coding: utf-8

# **Linear Regression project using PySpark**
# 
# Data has been taken from UC Irvine Machine Learning repository. It is a description of Ship carriers and the related parameters which includes: 
#     * Ship Name
#     * Cruise Line
#     * Age of the ship
#     * Tonnage
#     * Passanger Capacity
#     * Length
#     * Number of Cabins
#     * Passenger Density
#     * Crew (Target Variable)
# 
# I would be using PySpark and Machine Learning to predict our target variable, Crew. 
# 

# In[ ]:


#Start a new Spark Session
import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
from pyspark.sql import SparkSession

# App named 'Cruise'
spark = SparkSession.builder.appName('cruise').getOrCreate()


# In[6]:


#Read the csv file in a dataframe
df = spark.read.csv('cruise_ship_info.csv',inferSchema=True,header=True)


# In[7]:


#Check the structure of schema
df.printSchema()


# In[8]:


df.show()


# In[9]:


df.describe().show()


# In[10]:


df.groupBy('Cruise_line').count().show()


# In[23]:


#Convert string categorical values to integer categorical values
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Cruise_line", outputCol = "cruise_cat")
indexed = indexer.fit(df).transform(df)
indexed.head(5)


# In[24]:


from pyspark.ml.linalg import Vectors


# In[25]:


from pyspark.ml.feature import VectorAssembler


# In[26]:


indexed.columns


# In[28]:


# Create assembler object to include only relevant columns 
assembler = VectorAssembler(
inputCols=['Age',
 'Tonnage',
 'passengers',
 'length',
 'cabins',
 'passenger_density',
 'crew',
 'cruise_cat'],
outputCol="Features")


# In[29]:


output = assembler.transform(indexed)


# In[30]:


output.select("features","crew").show()


# In[31]:


final_data = output.select("features","crew")


# In[32]:


#Split the train and test data into 70/30 ratio
train_data,test_data = final_data.randomSplit([0.7,0.3])


# In[33]:


from pyspark.ml.regression import LinearRegression


# In[34]:


#Training the linear model
lr = LinearRegression(labelCol='crew')


# In[35]:



lrModel = lr.fit(train_data)


# In[39]:



print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# In[40]:


#Evaluate the results with the test data
test_results = lrModel.evaluate(test_data)


# In[41]:


print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))


# In[42]:


from pyspark.sql.functions import corr


# In[43]:


#Checking for correlations to explain high R2 values
df.select(corr('crew','passengers')).show()


# In[44]:


df.select(corr('crew','cabins')).show()

