import pandas as pd 
df = pd.read_csv("classification_inception_imagenet.csv")
print(df['InferenceLatency'].mean())

df = pd.read_csv("classification_resnet_imagenet.csv")
print(df['InferenceLatency'].mean())