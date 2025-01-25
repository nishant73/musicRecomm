
# Music Recommender System using ALS Algorithm with Apache Spark and Python

## Project Overview
This project aims to build a music recommender system using the collaborative filtering technique and the ALS (Alternating Least Squares) algorithm. The system recommends new musical artists to a user based on their listening history. This type of recommender system is widely used in music streaming services, such as Pandora and Spotify, and could also be applied to TV shows or movie recommendations (e.g., Netflix).

## Estimated Execution Time
- Whole Script: 2 minutes
- Estimated Time to Complete: 8 hours

## Dataset Overview
The project uses publicly available song data from Audioscrobbler. The dataset has been reduced to focus on the top 50 most prolific users (highest artist play counts), ensuring that the code runs in a reasonable time on a single machine.

The data files include:
- `user_artist_data.txt`: Contains the plays of artists by users (around 24.2 million user-artist plays).
- `artist_alias.txt`: Contains a mapping of misspelled or variant artist IDs to their canonical IDs.
- `artist_data.txt`: Maps canonical artist IDs to the names of the artists.

## Steps to Run the Project

### 1. Import Libraries
The following libraries are needed:
```python
import findspark
from pyspark.mllib.recommendation import *
from pyspark import SparkContext, SparkConf
import random
from operator import *
from collections import defaultdict
```

### 2. Initialize Spark Context
Start by initializing the Spark Context:
```python
spark = SparkContext.getOrCreate()
spark.stop()
spark = SparkContext('local','Recommender')
```

### 3. Load Data
Load the dataset into RDDs:
```python
artistData = spark.textFile('./data_raw/artist_data_small.txt').map(lambda s:(int(s.split("\t")[0]),s.split("\t")[1]))
artistAlias = spark.textFile('./data_raw/artist_alias_small.txt')
userArtistData = spark.textFile('./data_raw/user_artist_data_small.txt')
```

### 4. Data Exploration
Explore the dataset to find total play counts and identify the top users:
```python
userArtistData = userArtistData.map(lambda s:(int(s.split(" ")[0]),int(s.split(" ")[1]),int(s.split(" ")[2])))
artistAliasDictionary = {}
dataValue = artistAlias.map(lambda s:(int(s.split("\t")[0]),int(s.split("\t")[1])))
```

### 5. Splitting Data for Testing
Use `randomSplit` to divide the data into training, validation, and test datasets:
```python
trainData, validationData, testData = userArtistData.randomSplit((0.4,0.4,0.2),seed=13)
trainData.cache()
validationData.cache()
testData.cache()
```

### 6. Model Evaluation
Define the `modelEval` function to evaluate the model's performance:
```python
def modelEval(model, dataset):
    # Code to evaluate the model
```

### 7. Model Construction
Train the model using ALS with different rank values to choose the best performing model:
```python
rankList = [2,10,20]
for rank in rankList:
    model = ALS.trainImplicit(trainData, rank , seed=345)
    modelEval(model, validationData)
```

### 8. Top Artist Recommendations
Use the best model to recommend the top 5 artists for a user:
```python
TopFive = bestModel.recommendProducts(1059637, 5)
for item in range(0,5):
    print("Artist "+str(item)+": "+artistData.filter(lambda x:x[0] == TopFive[item][1]).collect()[0][1])
```

## Notes
- The project uses Spark's ALS algorithm for collaborative filtering and implicit feedback.
- The dataset was reduced for faster computation on a single machine.
- Adjust the rank parameter to fine-tune the model and evaluate performance.

## Conclusion
This project implements a simple music recommender system using Spark's ALS algorithm, allowing us to recommend artists based on a user's listening history. The model was evaluated using a validation set, and top artists were recommended for a user based on the trained model.

## Files and Folders
- `README.md`: Project documentation.
- `data_raw/`: Folder containing the raw data files (`artist_data_small.txt`, `artist_alias_small.txt`, `user_artist_data_small.txt`).

