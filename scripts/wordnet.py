import pandas as pd

gt = pd.read_csv("inputs/mscoco concepts/mscoco-groundtruth.csv")
props = pd.read_csv("inputs/mscoco concepts/properties.csv")

gt_objects = set(gt["name"].unique())
prop_objects = set(props["object"].unique())

missing = gt_objects - prop_objects

print("Missing objects:")
print(sorted(missing))