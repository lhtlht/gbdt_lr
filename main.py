import pandas as pd
import numpy as np
import pickle
import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from gbdtree import GbdtModelTrees

spark = SparkSession \
    .builder \
    .master("local") \
    .appName("GbdtLr") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
#spark.sparkContext.setLogLevel("ERROR")

data = spark.read.options(delimiter='\t', header=True, nrow=100).csv(r"round1_iflyad_anticheat_traindata.txt").select('label', 'os', 'osv', 'ppi')
data.show()
(trainingData, testData) = data.randomSplit([0.7,0.3], seed=777)

pipeline = Pipeline(stages=[StringIndexer(inputCol="os", outputCol="osIndex", handleInvalid="keep"),
                            StringIndexer(inputCol="osv", outputCol="osvIndex", handleInvalid="keep"),
                            OneHotEncoderEstimator(inputCols=["osIndex", "osvIndex"],outputCols=["osVec", "osvVec"], handleInvalid='keep')])
pipeline_model = pipeline.fit(trainingData)

train = pipeline_model.transform(trainingData)
test = pipeline_model.transform(testData)

def turn_labelpoint(x):
    label = int(x.label)
    features_list = np.array(list(x['osVec']) + list(x['osvVec']))
    features_vec = Vectors.sparse(len(features_list), list(np.where(features_list>0)[0]), list(features_list[features_list>0]))
    return LabeledPoint(label, features_vec)
train_labelPoint = train.rdd.map(turn_labelpoint)
test_labelPoint = test.rdd.map(turn_labelpoint)

gbdt_model = GradientBoostedTrees.trainClassifier(data=train_labelPoint,categoricalFeaturesInfo={}, numIterations=30, maxDepth=3)
tree_num = gbdt_model.numTrees()
print('tree_num',tree_num)
gbdt_model_trees = GbdtModelTrees(gbdt_model.toDebugString())
# for t in gbdt_model_trees.trees:
#     t.pre_order(t.root)
#     print('--------------------')
#sys.exit(-1)
"""
categoricalFeaturesInfo：向量中为分类属性的索引表。任务没有出现在该列表中的特征将会以连续值处理。{n:k}表示第 n 个特征，是 0-k 的分类属性。
"""

def getTreeLeafMap():


    feature_map = dict()
    feature_id = 0
    for treeIndex in range(tree_num):
        treeTopNode = gbdt_model_trees.getTree(treeIndex)
        treeNodeQueue = []
        treeNodeQueue.append(treeTopNode)
        while treeNodeQueue:
            resNode = treeNodeQueue.pop()
            if (resNode.sign == 3):  #  当sign为3时是叶子节点
                feature_map[f'{treeIndex}_{resNode.id}'] = feature_id
                feature_id = feature_id + 1
            if (resNode.left):
                treeNodeQueue.append(resNode.left)
            if (resNode.right):
                treeNodeQueue.append(resNode.right)

    return feature_map,feature_id

feature_map, feature_id = getTreeLeafMap()
feature_len = feature_id + 1
#print(feature_map)

def add_gbdt_leaf(x):
    features = x.features
    label = x.label
    gbdt_leaf_features = np.zeros(feature_len)
    for treeIndex in range(tree_num):
        treeNode = gbdt_model_trees.getTree(treeIndex)

        while (treeNode.sign!=3):
            if features[treeNode.featureId] <= treeNode.threshold:
                treeNode = treeNode.left
            else:
                treeNode = treeNode.right
        key = f'{treeIndex}_{treeNode.id}'
        #print(key)
        #print(feature_map[key])
        gbdt_leaf_features[feature_map[key]] = 1
    return LabeledPoint(label, gbdt_leaf_features)

train_gbdt_lr = train_labelPoint.map(add_gbdt_leaf)
train_gbdt_lr.count()
