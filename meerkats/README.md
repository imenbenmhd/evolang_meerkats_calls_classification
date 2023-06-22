# Meerkats Call-Type Classification

The structure of this directory is organized as the following:

```
├── __init__.py     # Init file
├── README.md       # This file
├── src             # Source code
    ├── utils       # scripts
    └── features_extraction
    ├── models      # Model architectures              
    └── data        # dataloaders file and any other mapping 
  

## Notes:
- Regular PyTorch code can be left behind for PyTorch lightening code.
- The files under `src/models` can be used with regular PyTorch or through PyTorch Lightening thanks to the `lit.py` wrapper file. 

Example:

Regular PyTorch:
```python

from src.models.Palazcnn import PalazCNN

model = PalazCNN(n_input=n_input, n_output=num_classes)
```

PyTorch Lightning :zap: :
```python
from src.models.lit import Lit
from src.models.Palazcnn import PalazCNN

model = Lit(PalazCNN(n_input=n_input, n_output=num_classes),learning_rate,num_classes)
```

## To Extract features:
If feature_name is compare catch or egemaps.

```
python src/features_extraction/extract_feats_segments.py -n feature_name -d path_toinfofile.csv -i class_to_index.json -p features/ -c number_of_classes 
```

if feature_name is embeddings:

```
python src/features_extraction/extract_feats_segments.py -n embeddings -m path_model_checkpoint -d path_csvfile.csv -i path_class_to_index.json -p features/ -c number_classes -lr learning_rate_model
```

When we want to extract embeddings we need to add the path of the model checkpoint and the learning rate of the model.
