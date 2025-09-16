# Trajectory Augmented Conditional Flow Matching

## Activating the Virtual Environment

This project uses Python 3.12. See [`requirements.txt`](requirements.txt) for a list of packages for the venv. To activate, run from this directory the following snippet:

```bash
conda activate ./venv
```

## Directory Structure
Directory names are fairly self-explanatory. The main source code is in `src/trajaugcfm/`. In particular, [`src/trajaugcfm/constants.py`](src/trajaugcfm/constants.py) contains several useful constants such as paths to `data/` and `results/` as well as the observable variable names from the MARM simulator. [`src/trajaugcfm/sampler.py`](src/trajaugcfm/sampler.py) contains the main code in the `TrajAugCFMSampler` class which implements a `torch.utils.data.IterableDataset` for (relatively) painless dataloading for the training scripts. Do note that this class automatically handles batching so if passing `TrajAugCFMSampler` to a `torch.utils.data.DataLoader`, the `batch_size` kwarg MUST be set to `None`. Finally, [`src/trajaugcfm/models/model.py`](src/trajaugcfm/models/models.py) contains a simple MLP with SeLU activation functions.

**NOTE**: There is a `TrajAugCFMSamplerForLoop` in `sampler.py`. Ignore--it is for testing and debugging purposes only.

## Running the Code
The main training script can be found at [`scripts/trainmodel.py`](scripts/trainmodel.py). For now, the easiest way to run the code is to call (with the venv activated):
```bash
python scripts/trainmodel.py [--myargs]
```
Inside that script's `main()` function are the necessary data loading, scaling, model setup, model training, and loss plotting. The outputs are a saved `results/<experiment_name>/losses.npz` file which can be keyed into using `train` or `val`, as well as the trained model in `results/<experiment_name>/model.pt`. In addition, the script will also output a `results/<experiment_name>/args.json` file containing all the command line arguments passed into the training script. 
