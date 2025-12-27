import pyarrow.parquet as pq
import pandas as pd
import pysbd
import orjson


# Read Arrow file
test_pq = pq.read_table('test-00000-of-00001.parquet')


# Convert to pandas for viewing
test_df = test_pq.to_pandas()

train_pq = pq.read_table("train-00000-of-00001.parquet")
val_pq = pq.read_table("valid-00000-of-00001.parquet")

train_df = train_pq.to_pandas()
val_df = val_pq.to_pandas()data_df = pd.concat([train_df, val_df, test_df], axis=0)

seg = pysbd.Segmenter(language="en", clean=True)

texts = data_df['content'].values

from tqdm import tqdm
sentences = []

for text in tqdm(texts):
    temp = seg.segment(text)
    sentences.extend(temp)

dict_sent =  {str(i): sentence for i, sentence in enumerate(sentences)}
clean_file = open("clean_big.txt", "wb")
clean_file.write(orjson.dumps(dict_sent))

