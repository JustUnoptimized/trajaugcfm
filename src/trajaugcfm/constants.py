from pathlib import Path
import os
from typing import Final, Sequence, LiteralString

## DataFrame column names
CONSTOBS: Final[Sequence[LiteralString]] = [
    'tBRAF_obs', 'tCRAF_obs', 'tRAS_obs', 'tMEK_obs',
    'tERK_obs', 'tGRB2_obs', 'tSOS1_obs', 'tCBL_obs'
]
DYNOBS: Final[Sequence[LiteralString]] = [
    'pMEK_obs', 'pMEK_IF_obs', 'pERK_obs', 'pERK_IF_obs',
    'tDUSP_obs', 'tmDUSP_obs', 'tEGF_obs', 'tEGFR_obs',
    'pEGFR_obs', 'tmEGFR_obs', 'tSPRY_obs', 'tmSPRY_obs', 'pS1134SOS1_obs'
]
METACOLS: Final[Sequence[LiteralString]] = ['cellidx', 'time', 'raficonc', 'mekiconc']
OBS: Final[Sequence[LiteralString]] = CONSTOBS + DYNOBS
LOADCOLS: Final[Sequence[LiteralString]] = OBS + METACOLS

## Base directory
BASEDIR: Final[LiteralString] = os.path.join(Path(__file__).parents[2])
DATADIR: Final[LiteralString] = os.path.join(BASEDIR, 'data')
RESDIR: Final[LiteralString] = os.path.join(BASEDIR, 'results')

## Saved preprocessed names
DATA_SAVENAME: Final[LiteralString] = 'data.npy'
IDX2RCMC_SAVENAME: Final[LiteralString] = 'idx2rcmc.pkl'
RCMC2IDX_SAVENAME: Final[LiteralString] = 'rcmc2idx.pkl'

