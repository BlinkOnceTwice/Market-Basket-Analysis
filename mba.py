import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel

#Creating spark context and spark sql session
conf = SparkConf()
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()

#Extracting prior dataset and filtering the column values
prior_data=sc.textFile('/home/013638703/project/Dataset/order_products__prior.csv', use_unicode=False)
prior_headder = prior_data.first()
#print(prior_headder)
#Removing the headder and splitting the entires
prior_RDD = prior_data.filter(lambda(row) : (row != prior_headder)).map(lambda(r):(r.split(',')))
#print(prior_RDD.take(10))
prior_RDD = prior_RDD.map(lambda (orderID,productID,cart_order,reordered):(float(orderID),float(productID),float(cart_order),float(reordered)))

#Extracting training dataset and filtering the column values
train_data=sc.textFile('/home/013638703/project/Dataset/order_products__train.csv', use_unicode=False)
train_headder = train_data.first()
#print(train_headder)
#removing the headder and splitting the entries
train_RDD = train_data.filter(lambda(row) : (row != train_headder)).map(lambda(r):(r.split(',')))
#print(traiin_RDD.take(10))
train_RDD = train_RDD.map(lambda (orderID,productID,cart_order,reordered):(float(orderID),float(productID),float(cart_order),float(reordered)))

#Extracting the order dataset and filtering the column values
order_data=sc.textFile('/home/013638703/project/Dataset/orders.csv', use_unicode=False)
order_headder = order_data.first()
print(order_headder)
#removing the headder and splitting the entries
order_RDD = order_data.filter(lambda(row) : (row != order_headder)).map(lambda(r):(r.split(',')))
print(order_RDD.take(10))
order_RDD = order_RDD.map(lambda (orderID,userID,eval_set,order_number,order_dow,order_hod,days_since):(float(orderID),float(userID),0 if eval_set == "prior" else 1 if eval_set=="train" else 2,float(order_number),float(order_dow),float(order_hod),0 if days_since=="" else float(days_since)))

#Extracting the product dataset and filtering the column values
product_data=spark.read.csv("/home/013638703/project/Dataset/products.csv", header=True, mode="DROPMALFORMED").rdd
product_headder = product_data.first()
#print(product_headder)
#Removing the hedder
product_RDD = product_data.filter(lambda (row) : (row != product_headder))
#print(product_RDD.take(10))
product_RDD = product_RDD.map(lambda (row):(float(row[0]),float(row[2]),float(row[3])))


#Number of entries to consider for training and texting. consider 15000 entries for our project
length = 15000
#parallelize the RDD for filtering
orders_limit=sc.parallelize(order_RDD.take(length))
#extract only prior order from order RDD 
orders_prior = orders_limit.filter(lambda(row) : (row[2] == 0))
#exract only train order from order RDD
orders_train = orders_limit.filter(lambda(row) : (row[2] == 1))
#extrct only test order from order rdd
orders_test = orders_limit.filter(lambda(row) : (row[2] == 2))

#print(orders_prior.count())
#print(orders_prior.collect())
#print(orders_train.count())
#print(orders_train.collect())
#print(orders_test.count())
#print(orders_test.collect())

#collect the keys from prior for prior orders of the users
prior_keys = orders_prior.keys().collect()

#Getting training keys for training
train_keys = orders_train.keys().collect()

#Selecting training orders and prior orders based on prior and training keys
limited_prior_products = prior_RDD.filter(lambda(row) : (row[0] in prior_keys))
limited_train_products = train_RDD.filter(lambda(row) : (row[0] in train_keys))

#Converting RDD to pandas dataframe for further processing
ordersDF = orders_limit.toDF().toPandas()
ordersDF.columns=['orderID','userID','eval_set','order_number','order_dow','order_hod','days_since']
prior_productsDF=limited_prior_products.toDF().toPandas()
prior_productsDF.columns=['orderID','productID','cart_order','reordered']
train_productsDF=limited_train_products.toDF().toPandas()
train_productsDF.columns=['orderID','productID','cart_order','reordered']
products_DF=product_RDD.toDF().toPandas()
products_DF.columns=['productID','aisle_id','dep_id']

#Generating new product features from reordered orders
products = pd.DataFrame()
products['numorders'] = prior_productsDF.groupby(prior_productsDF.productID).size().astype(np.int32)
products['numreorders'] = prior_productsDF['reordered'].groupby(prior_productsDF.productID).sum().astype(np.float32)
products['reorder_ratio'] = (products.numreorders / products.numorders).astype(np.float32)
products_DF = products_DF.join(products, on='productID',how="inner")
products_DF.set_index('productID', drop=False, inplace=True)

#print(products_DF)
#print(products_DF.productID.size)


#Combining new features to existing dataframe
ordersDF.set_index('orderID', inplace=True, drop=False)
prior_productsDF = prior_productsDF.join(ordersDF, on='orderID', rsuffix='*', how="inner")
prior_productsDF.drop('orderID*', inplace=True, axis=1)

#print(prior_productsDF)
#generating new feature based on days since ordered
userOrderData = pd.DataFrame()
userOrderData['avg_orderingGap'] = ordersDF.groupby('userID')['days_since'].mean().astype(np.float32)
userOrderData['total_orders'] = ordersDF.groupby('userID').size().astype(np.int16)

#print(userOrderData)


#generating new feature using previous ordered products
userPriorData = pd.DataFrame()
userPriorData['items_purchased_total'] = prior_productsDF.groupby('userID').size().astype(np.int16)
userPriorData['all_items'] = prior_productsDF.groupby('userID')['productID'].apply(set)
userPriorData['distinct_items'] = (userPriorData.all_items.map(len)).astype(np.int16)

#print(userPriorData)

#Complete dataset for training
usersFinalData = userPriorData.join(userOrderData)
usersFinalData['average_items'] = (usersFinalData.items_purchased_total / usersFinalData.total_orders).astype(np.float32)

#print(usersFinalData)


#Generating more data features from dataset
userRelatedProductsData = prior_productsDF.copy()
userRelatedProductsData['userProduct'] = userRelatedProductsData.productID + userRelatedProductsData.userID * 100000
userRelatedProductsData = userRelatedProductsData.sort_values('order_number')
userRelatedProductsData = userRelatedProductsData.groupby('userProduct', sort=False).agg({'orderID': ['size', 'last'], 'cart_order': 'sum'})
userRelatedProductsData.columns = ['numorders', 'final_orderId', 'sum_cartOrder']
userRelatedProductsData.numorders = userRelatedProductsData.numorders.astype(np.int16)
userRelatedProductsData.sum_cartOrder = userRelatedProductsData.sum_cartOrder.astype(np.int16)
userRelatedProductsData.final_orderId = userRelatedProductsData.final_orderId.astype(np.int32)


#print(userRelatedProductsData)


train_productsDF.set_index(['orderID', 'productID'], inplace=True, drop=False)


#Transforming training and testing dataset from the extracted features from the data
def datasetFeatures(orderset):
    orders = []
    products = []
    ord_prod_labels = []
    for tuple1 in orderset.itertuples():
        orderID = tuple1.orderID
        userID = tuple1.userID
        userItems = usersFinalData.all_items[userID]
        products += userItems
        orders += [orderID] * len(userItems)
        ord_prod_labels += [(orderID, productID) in train_productsDF.index for productID in userItems]
        
    finalDF = pd.DataFrame({'orderID':orders, 'productID':products}, dtype=np.int32) 
    finalDF['label'] = ord_prod_labels
    finalDF['userID'] = finalDF.orderID.map(ordersDF.userID)
    finalDF['user_SumOrders'] = finalDF.userID.map(usersFinalData.total_orders)
    finalDF['user_items_purchased_total'] = finalDF.userID.map(usersFinalData.items_purchased_total)
    finalDF['distinct_items'] = finalDF.userID.map(usersFinalData.distinct_items)
    finalDF['user_avg_orderingGap'] = finalDF.userID.map(usersFinalData.avg_orderingGap)
    finalDF['user_average_items'] =  finalDF.userID.map(usersFinalData.average_items)
    finalDF['order_dow'] = finalDF.orderID.map(ordersDF.order_dow)
    finalDF['order_hod'] = finalDF.orderID.map(ordersDF.order_hod)
    finalDF['days_since'] = finalDF.orderID.map(ordersDF.days_since)
    finalDF['days_sinceProportion'] = finalDF.days_since / finalDF.user_avg_orderingGap
    finalDF['numorders'] = finalDF.productID.map(products_DF.numorders).astype(np.float32)
    finalDF['numreorders'] = finalDF.productID.map(products_DF.numreorders)
    finalDF['prod_reorderProportion'] = finalDF.productID.map(products_DF.reorder_ratio)
    finalDF['userProductGrp'] = finalDF.userID * 100000 + finalDF.productID
    finalDF['userProduct_orders'] = finalDF.userProductGrp.map(userRelatedProductsData.numorders)
    finalDF['userProduct_ordersRatio'] = (finalDF.userProduct_orders / finalDF.user_SumOrders).astype(np.float32)
    finalDF['userProduct_final_orderId'] = finalDF.userProductGrp.map(userRelatedProductsData.final_orderId)
    finalDF['userProduct_average_cartOrder'] = (finalDF.userProductGrp.map(userRelatedProductsData.sum_cartOrder) / finalDF.userProduct_orders).astype(np.float32)
    finalDF['userProduct_reorderProportion'] = (finalDF.userProduct_orders / finalDF.user_SumOrders).astype(np.float32)
    finalDF['userProduct_orders_sinceFinal'] = finalDF.user_SumOrders -finalDF.userProduct_final_orderId.map(ordersDF.order_number)
    return (finalDF)
#transforming the features
ordersTest = ordersDF[ordersDF.eval_set == 2]
ordersTrain = ordersDF[ordersDF.eval_set == 1]

trainSet = datasetFeatures(ordersTrain)
testSet = datasetFeatures(ordersTest)


#print(trainSet)

#converting lables to int for easy processing
trainSet.label = trainSet.label.astype(int)
testSet.label = testSet.label.astype(int)

#Lables for training the agent using extracted dataset
features =['label','user_SumOrders', 'user_items_purchased_total', 'distinct_items',
       'user_avg_orderingGap', 'user_average_items','order_dow',
       'order_hod', 'days_since', 'days_sinceProportion','numorders', 'numreorders',
       'prod_reorderProportion', 'userProduct_orders', 'userProduct_ordersRatio',
       'userProduct_average_cartOrder', 'userProduct_reorderProportion', 'userProduct_orders_sinceFinal']

#print(trainSet[features])


#print(testSet[features])

#Creating spark dataframe
trainRDD = spark.createDataFrame(trainSet[features]).rdd
testRDD = spark.createDataFrame(testSet[features]).rdd


#Transforming the datasets to LibSVM format required for the regression model
trainFinal = trainRDD.map(lambda line: LabeledPoint(line[0],[line[1:]]))
testFinal = testRDD.map(lambda line: LabeledPoint(line[0],[line[1:]]))


#print(trainFinal.count())
#print(trainFinal.take(10))

#print(testFinal.count())
#print(testFinal.take(10))

#Splitting the dataset to testing and training (considered 80% of data for training)
(training1, training2) = trainFinal.randomSplit([0.8, 0.2])

training1.collect()


model_1 = RandomForest.trainRegressor(training1, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)

predictionsRFTrain = model_1.predict(training1.map(lambda x: x.features))

#print(predictionsRFTrain.collect())

#print(training1.collect())


#A threshold value of 0.2 was used to choose if the data to be included in the training set
labelsAndPredictionsRF = training1.map(lambda row: row.label).zip(predictionsRFTrain).map(lambda(row):(row[0],(0.0 if row[1] < 0.2 else 1.0)))

metricsRF = MulticlassMetrics(labelsAndPredictionsRF)

#Calculate precision and recall and f1score
precisionRF = metricsRF.precision()
recallRF = metricsRF.recall()
f1ScoreRF = metricsRF.fMeasure()
print("RF summary on Train")
print("Precision = %s" % precisionRF)
print("Recall = %s" % recallRF)
print("F1 Score = %s" % f1ScoreRF)

#Training the RandomForest Regression model
model1 = RandomForest.trainRegressor(trainFinal, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)

predictionsRF = model1.predict(testFinal.map(lambda r: r.features))


#print(predictionsRF.collect())


print(predictionsRF.count())
#Predict the model

testSet['predictionRF']=predictionsRF.map(lambda r: (r, )).toDF().toPandas()


#Creating product ids based on testing
limit = 0.19
dat = dict()
for tuple1 in testSet.itertuples():
    if tuple1.predictionRF > limit:
        try:
            dat[tuple1.orderID] += ' ' + str(tuple1.productID)
        except:
            dat[tuple1.orderID] = str(tuple1.productID)

for ord_id in ordersTest.orderID:
    if ord_id not in dat:
        dat[ord_id] = 'None'

resultRF = pd.DataFrame.from_dict(dat, orient='index')

resultRF.reset_index(inplace=True)
resultRF.columns = ['orderID', 'products']


#print(resultRF)
