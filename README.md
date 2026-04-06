# Trajectory Augmented Conditional Flow Matching
## Activating the Virtual Environment
This project uses Python 3.12. See [`requirements.txt`](requirements.txt) for a list of packages for the venv. To activate, run from this directory the following snippet:
```bash
conda activate ./venv
```

## Directory Structure
Directory names are fairly self-explanatory. The main source code is in `src/trajaugcfm/` and runner scripts are found in `scripts/`.

## Main Source Code
### Sampler
Main logic is located in [`src/trajaugcfm/sampler.py`](src/trajaugcfm/sampler.py). The main class is `GCFMSamplerBase` (short for Guided Conditional FM) which implements `torch.utils.data.IterableDataset` for (relatively) painless dataloading for the training scripts.

The various flow/score matching options are selected via the Mixin pattern.
Flow Mixins:
- `AFMixin` (Anisotropic Flow Mixin)
- `IFCBMixin` (Isotropic Flow Constant Bridge)
- `IFSBMixin` (Isotropic Flow Schrodinger Bridge)

Score Mixins (currently not used for any testing/prototyping):
- `ASMixin` (Anisotropic Score Mixin)
- `NSMixin` (No Score Mixin)

Time Sampler Mixins:
- `UniformTimeMixin` ($t \sim \mathcal{U}(0, 1)$)
- `BetaTimeMixin` ($t \sim \operatorname{Beta}(a, a)$)

Time Enrich Mixins:
- `TimeRFFMixin` (Enrich $t$ with random Fourier features)
- `TimeNoEnrichMixin` (Use raw $t$)

To use the mixins, the actual sampler is constructed and used via the following pseudocode:
```python
bases = (time_mixin, time_enrich_mixin, flow_mixin, score_mixin, GCFMSamplerBase)
GCFMSampler = type('GCFMSampler', bases, {})
sampler = GCFMSampler(*args, **kwargs)
dataloader = DataLoader(sampler, batch_size=None)  # torch.utils.data.DataLoader
for i, batch in enumerate(dataloader):
    # train epoch
```
This class automatically handles batching so if passing `GCFMSampler` to a `torch.utils.data.DataLoader`, the `batch_size` kwarg MUST be set to `None`.

<**Note**> *The `af-explore` branch is currently set up specifically for the `AFMixin` which implements anisotropic flow matching.* </**Note**>

### Eigenvector Orientation
Found at [`src/trajaugcfm/eig_orient.py`](src/trajaugcfm/eigen_orient.py), which itself is a vectorized re-implementation of [A Consistently Oriented Basis for Eigenanalysis: Improved Directional Statistics](https://link.springer.com/article/10.1007/s41060-024-00570-5) ([GitHub](https://gitlab.com/thucyd-dev/thucyd)) by Jay Damask (2025). The re-impementation forces alignment to first octant.

### Models
[`src/trajaugcfm/models/model.py`](src/trajaugcfm/models/models.py) contains a simple MLP with SeLU activation functions.

### Utils
[`src/trajaugcfm/utils.py`](src/trajaugcfm/utils.py) contains a few utility functions. Not all functions used and some may be removed/changed in later commits, especially due to redundancy from `numpy` or `scipy` implementations.

### Constants
[`src/trajaugcfm/constants.py`](src/trajaugcfm/constants.py) contains several useful constants such as paths to `data/` and `results/` as well as the observable variable names from the MARM simulator.

## Main Scripts
All scripts can be called with the `-h` or `--help` flags to see a brief description of all the possible command-line arguments.

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

### Plotting
The plotting script is at [`scripts/make_plots.py`](scripts/make_plots.py). Call it using
```bash
python scripts/make_plots.py [--myargs]
```
This script will plot the following:
- Train/validation loss curve
- Learning rate
- Inferred trajectories
- Metrics computed over the inferred trajectories

Additionally if the `--diagnostics` flag was set for `trainmodel.py` then this will plot the following:
- Observed variable eigenvalues
- Observed variable eigenvalue inverses
- Observed variable (oriented) eigenvectors
- Observed variable spectral scores
- Hidden variable eigenvalues
- Hidden variable eigenvectors
- Mean correction to hidden variable time-varying mean
- Gain

## Running Batch Experiments Using Slurm
The utility script [`batch_run.sh`](batch_run.sh) contains the logic defining the experiment save directory, hyperpameters, and running the train -> trajgen -> eval -> plot pipeline detailed above. Call the script using
```bash
./batch_run [-h | --help] [options]
```
You can use the `-h` or `--help` flags to see a brief descripion of the possible options defining where to save results. In short, the script collects all the possible given argument combinations and saves it into a `.txt` file, which is then read off of to submit jobs to Slurm. See [`runner.slurm`](runner.slurm) for any Slurm configs not detailed in the called `sbatch` commands.

Additionally, the script also outputs a rudimentary `expkeys.json` file outlining which directory contains which experiment.