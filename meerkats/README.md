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
```
python src/features_extraction/extract_feats_segments.py -n feature_name -d path_toinfofile.csv -i class_to_index.json -p features/
```

feature_name is egemaps or compare or catch or embeddings. If embeddings then add -m path_to_model_checkpoint after -n 

