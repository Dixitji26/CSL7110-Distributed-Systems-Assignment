from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, length
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
import numpy as np

# Create Spark Session
spark = SparkSession.builder \
    .appName("CSL7110 Spark Analysis") \
    .getOrCreate()

# ----------------------------
# 1. Load Dataset
# ----------------------------
books_df = spark.read.text("/home/antpc/D184MB/*.txt") \
    .withColumn("file_name", input_file_name()) \
    .withColumnRenamed("value", "text")

# ----------------------------
# 2. Extract Metadata
# ----------------------------
books_df = books_df.withColumn(
    "author",
    regexp_extract("text", r"(?i)Author:\s*([^\r\n]*)", 1)
)

books_df = books_df.withColumn(
    "release_date",
    regexp_extract("text", r"(?i)Release Date:\s*([^\r\n]*)", 1)
)

books_df = books_df.withColumn(
    "language",
    regexp_extract("text", r"(?i)Language:\s*([^\r\n]*)", 1)
)

books_df = books_df.withColumn(
    "year",
    regexp_extract("release_date", r"\d{4}", 0)
)

# ----------------------------
# 3. Books Per Year
# ----------------------------
books_df.groupBy("year").count().orderBy("year").show(10)

# ----------------------------
# 4. Language Distribution
# ----------------------------
books_df.groupBy("language") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(5)

# ----------------------------
# 5. TF-IDF Pipeline
# ----------------------------
tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="words",
    pattern="\\W+"
)

words_df = tokenizer.transform(books_df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_df = remover.transform(words_df)

vectorizer = CountVectorizer(
    inputCol="filtered",
    outputCol="rawFeatures",
    vocabSize=20000,
    minDF=5
)

cv_model = vectorizer.fit(filtered_df)
featurized_df = cv_model.transform(filtered_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

tfidf_df.select("file_name", "features").show(5)

# ----------------------------
# 6. Cosine Similarity
# ----------------------------
tfidf_small = tfidf_df.select("file_name", "features")
data = tfidf_small.collect()

def cosine_similarity(v1, v2):
    dot = float(v1.dot(v2))
    norm1 = np.linalg.norm(v1.toArray())
    norm2 = np.linalg.norm(v2.toArray())
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

similarities = []

for i in range(len(data)):
    for j in range(i+1, len(data)):
        sim = cosine_similarity(data[i]["features"], data[j]["features"])
        similarities.append((
            data[i]["file_name"],
            data[j]["file_name"],
            sim
        ))

top5 = sorted(similarities, key=lambda x: x[2], reverse=True)[:5]

print("Top 5 Similar Books:")
for pair in top5:
    print(pair)

# ----------------------------
# 7. Influence Network (5 Year Window)
# ----------------------------
authors_df = books_df.select("author", "year") \
    .filter("author != '' AND year IS NOT NULL")

author_data = authors_df.collect()
edges = []

for i in range(len(author_data)):
    for j in range(i+1, len(author_data)):
        year_diff = abs(int(author_data[i]["year"]) - int(author_data[j]["year"]))
        if year_diff <= 5:
            edges.append((
                author_data[i]["author"],
                author_data[j]["author"],
                year_diff
            ))

print("Sample Influence Edges:")
for e in edges[:10]:
    print(e)

spark.stop()
