# CSL7110-ML-FOR-BIG-DATA-Assignment
Hadoop MapReduce (Java) and Apache Spark (PySpark) implementation including WordCount, split-size experiment, TF-IDF similarity analysis, and influence network construction on Project Gutenberg dataset.

---

## 📖 Overview

This repository contains the complete implementation of the CSL7110 Distributed Systems assignment.  
The project demonstrates distributed data processing using Hadoop MapReduce (Java) and Apache Spark (PySpark).

The assignment includes:

- Hadoop setup and configuration
- Custom WordCount MapReduce implementation
- Split-size performance experiment
- Spark-based metadata extraction
- TF-IDF document vectorization
- Cosine similarity computation
- Author influence network construction

---

## 🛠 Technologies Used

- Hadoop 3.3.6  
- Java 8 (OpenJDK 1.8)  
- Apache Spark 3.5.1  
- Python 3 (PySpark)  
- HDFS  
- Git & GitHub  

---

## 📂 Repository Structure
CSL7110-ML-FOR-BIGDATA-Assignment/
│
├── MapReduce/
│ └── WordCount.java
│
├── Spark/
│ └── spark_analysis.py
│
├── Report/
│ └── <rollnumber>_CSL7110_Assignment.pdf
│
└── README.md

---

# 🔹 Part 1 – Hadoop MapReduce

## Hadoop Configuration

- Installed Hadoop 3.3.6
- Configured core-site.xml, hdfs-site.xml, mapred-site.xml, yarn-site.xml
- Enabled passwordless SSH
- Verified running services using:


---

## HDFS Test

Test file used:

Uploaded and verified on HDFS.

---

## WordCount Implementation

Implemented custom `WordCount.java`:

### Mapper
- Removes punctuation
- Tokenizes text
- Emits (word, 1)

### Reducer
- Aggregates counts
- Emits (word, total_count)

Execution commands:
vac -classpath hadoop classpath WordCount.java
jar cf WordCount.jar WordCount*.class
hadoop jar WordCount.jar WordCount input output
---

## Split Size Experiment

Modified:

Tested with:
- 32 MB
- 64 MB
- 128 MB

### Observation

Execution time remained similar because:

- Dataset size (~8MB) was smaller than split threshold
- Only one map task was generated
- No additional parallelism occurred

---

# 🔹 Part 2 – Apache Spark Analysis

## Dataset

Project Gutenberg collection (~184MB).

Loaded using:

```python
spark.read.text("/home/antpc/D184MB/*.txt")

Metadata Extraction

Extracted:

Author

Release Date

Language

Year

Using Spark SQL regex functions.

# TF-IDF Pipeline

Steps:

1)RegexTokenizer

2)StopWordsRemover

3)CountVectorizer

*IDF

-->TF-IDF highlights words that are frequent in a document but rare across the corpus.

*Cosine Similarity

-->Cosine similarity formula:

cos(θ)=(A⋅B)/(∣∣A∣∣×∣∣B∣∣)
cos(θ)=(A⋅B)/(∣∣A∣∣×∣∣B∣∣)

Used to measure similarity between document vectors.

Q)Why appropriate:

1)Works well with sparse high-dimensional TF-IDF vectors

2)Independent of document length

3)Measures angular similarity

4)Author Influence Network

#) Constructed simplified influence network:

Nodes → Authors

Edge exists if books released within X-year window

Edge weight → Year difference

This demonstrates graph-style analytics in Spark.
