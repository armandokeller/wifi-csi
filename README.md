# wifi-csi
Wifi CSI signal dataset for object detection

This repository contains one csv file and two python pickle files (.pkl), one for the raw (the same data as csv) and the other for normalized data (amplitudes and phase).
Each pickle file contains a pandas DataFrame with the following columns:
 - 52 columns of amplitude (with the prefix "amp_")
 - 52 columns of phase (with the prefix "phase_")
 - type: If the object is metallic or organic
 - day: the day of data collection (1 or 2)
 - object_id: The identification of the object (1 to 4)
 - position: The object position on the grid as shown in the Figure ("empty" for empty grid measeurements)
 - configuration: The sensor setup (COMPRIDO for long and CURTO for short)

![image](https://github.com/armandokeller/wifi-csi/assets/1762410/e2b3302b-55fe-460a-816b-99ff77832759)

## Loading the pandas DataFrame from pickle
```python
from picke import load

dataset = load(open("./dataset_normalized.pkl","rb"))
```

The classifiers_example.py file contains a sort of examples of classification using the dataset.
