# Trajectory Augmented Conditional Flow Matching
## Activating the Virtual Environment
This project uses Python 3.12. See [`requirements.txt`](requirements.txt) for a list of packages for the venv. To activate, run from this directory the following snippet:
```bash
conda activate ./venv
```

## Directory Structure
Directory names are fairly self-explanatory. The main source code is in `src/trajaugcfm/`. In particular, [`src/trajaugcfm/constants.py`](src/trajaugcfm/constants.py) contains several useful constants such as paths to `data/` and `results/` as well as the observable variable names from the MARM simulator. [`src/trajaugcfm/sampler.py`](src/trajaugcfm/sampler.py) contains the main code in the `TrajAugCFMSampler` class which implements a `torch.utils.data.IterableDataset` for (relatively) painless dataloading for the training scripts. Do note that this class automatically handles batching so if passing `TrajAugCFMSampler` to a `torch.utils.data.DataLoader`, the `batch_size` kwarg MUST be set to `None`. Finally, [`src/trajaugcfm/models/model.py`](src/trajaugcfm/models/models.py) contains a simple MLP with SeLU activation functions.

## Running the Code
### Training
The main training script can be found at [`scripts/trainmodel.py`](scripts/trainmodel.py). For now, the easiest way to run the code is to call (with the venv activated):
```bash
python scripts/trainmodel.py [--myargs]
```
Inside that script's `main()` function are the necessary data loading, scaling, model setup, model training, and loss plotting.
The fitted scalers are saved in `results/<experiment_name>/{obs, hid}_scaler.z` for data recreating during generation and evaluation.
The outputs are a saved `results/<experiment_name>/losses.npz` file which can be keyed into using `train` or `val`, as well as the trained model in `results/<experiment_name>/model.pt`.
The learning rates per epoch are also saved into `losses.npz` and can be keyed into using `lrs`. This is probably not terribly interesting unless using a learning rate scheduler.
In addition, the script will also output a `results/<experiment_name>/args.json` file containing all the command line arguments passed into the training script.

### Trajectory Generation
The trajectory generation script is at [`scripts/trajgen.py`](scripts/trajgen.py). Call it using
```bash
python scripts/trajgen.py [--myargs]
```
This script will recreate the scaled data by loading the fitted scalers in `results/<experiment_name>/{obs, hid}_scaler.z`.
The initial conditions are taken from the validation split.
The number of initial conditions to use defaults to the whole training split.
Currently only SDE integration is supported using the `torchsde` package. The default integration method is Euler-Maruyama.
The number of function evaluations (NFE) is saved into `results/<experiment_name>/metrics.json`, accessible with the key `NFE`.
The generated trajectories are saved into `results/<experiment_name>/trajs_scaled.npy`. The saved array has shape `(N, T, d)`.
The default number of time points for inference is the number of time points in the reference trajectories.
As evident by the filename, inference happens in the scaled space. The inverse scaling operation is not applied afterwards.
The input arguments are saved into `results/<experiment_name>/trajgen_args.json`.

### Trajectory Evaluation
The evaluation script is at [`scripts/eval.py`](scripts/eval.py). Call it using
```bash
python scripts/eval.py [--myargs]
```
This script will recreate the scaled data by loading the fitted scalers in `results/<experiment_name>/{obs, hid}_scaler.z`.
The generated data is loaded from `results/<experiment_name>/trajs_scaled.npy`.
The RMSE and cosine similarity (metrics over the feature vector) are computed for each time point.
The MAE (metric for each individual feature) is computed for each time point.
The EMD and entropic EMD (distributional distances) using the squared Euclidean cost is computed for each time point.
The metrics are saved into `results/<experiment_name>/evals.npz` which can be keyed into using `RMSE`, `Cosine Similarity`, `abserr`, `EMD`, `Entropic EMD`.
`RMSE` and `Cosine Similarity` have shape `(N, T)`.
`EMD` and `Entropic EMD` have shape `(N,)`.
`abserr` has shape `(N, T, d)`.
The input arguments are saved into `results/<experiment_name>/eval_args.json`.
