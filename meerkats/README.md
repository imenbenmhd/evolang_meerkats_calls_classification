# Meerkats Call-Type Classification

The structure of this directory is organized as the following:

```
├── environment.yml # File to re-create conda environment 
├── __init__.py     # Init file
├── lightning_logs  # PyTorch Lightning logs
├── lit_submit.sh   # Submit PyTorch Lightning train script to `sgpu` grid
├── logs            # Gridtk/Jman logs
├── README.md       # This file
├── src             # Source code
│   ├── data        # Dataloader files and any other mapping
│   ├── models      # Model architectures              
│   └── utils       # Training scripts for (mostly for regular PyTorch)
├── submit.sh       # Submit regular PyTorch train script to `sgpu` grid
├── submitted.sql3  # Gridtk/Jman submission SQL file
├── train_lit.py    # Main script for PyTorch Lightening code
├── train.py        # Main script for regular Lightening code
└── wandb           # Weights & Biases logs
```

## Usage
```
conda env create -f environment.yml -n pytorch_calltype # Create env
conda activate pytorch_calltype                         # Activate env
python train_lit.py                                     # Run training script
```

## Notes:
- Regular PyTorch code can be left behind for PyTorch lightening code.
- The files under `src/models` can be used with regular PyTorch or through PyTorch Lightening thanks to the `lit.py` wrapper file. 

Example:

Regular PyTorch:
```python
from src.models.cnn_16khz_seg import CNN_16KHz_Seg

model = CNN_16KHz_Seg(n_input=n_input, n_output=num_classes)
```

PyTorch Lightning :zap: :
```python
from src.models.lit import Lit
from src.models.cnn_16khz_seg import CNN_16KHz_Seg

model = Lit(CNN_16KHz_Seg(n_input=n_input, n_output=num_classes))
```
