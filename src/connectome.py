"""Load the study network or generate a small, reproducible demonstration network."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

# Zero-based indices used in the study. Anatomical names must be documented
# from the original regions array before publication.
FRONTAL = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 74, 75,
])
OTHER = np.setdiff1d(np.arange(80), FRONTAL)


def check_connectome(weights, frontal, other, regions):
    """Stop if the network or its regional partition is inconsistent."""
    if weights.shape != (80, 80):
        raise ValueError('Expected an 80 x 80 matrix.')
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Weights must be finite and non-negative.')
    if not np.allclose(weights, weights.T):
        raise ValueError('The matrix must be symmetric.')
    if not np.allclose(np.diag(weights), 0):
        raise ValueError('The matrix diagonal must be zero.')
    if not np.array_equal(np.sort(frontal), FRONTAL):
        raise ValueError('The motor-frontal partition differs from the study.')
    if not np.array_equal(np.sort(other), OTHER):
        raise ValueError('The non-frontal partition differs from the study.')
    if np.asarray(regions).size != 80:
        raise ValueError('Expected 80 region labels.')


def load_connectome(configuration='configs/demo.json'):
    """Return W, regional indices, labels and provenance information."""
    path = Path(configuration).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    settings = config['connectome']

    if settings['kind'] == 'synthetic':
        # An arbitrary random symmetric network for demonstrating the code.
        # Its weights are not anatomical connectivity measurements.
        if settings['n_regions'] != 80:
            raise ValueError('This demonstration uses 80 regions.')
        rng = np.random.default_rng(settings['seed'])
        upper = np.triu(rng.uniform(0, 1, (80, 80)), k=1)
        weights = upper + upper.T
        # Row sums are at most 1; this is a demo choice, not study preprocessing.
        weights /= weights.sum(axis=1).max()
        frontal, other = FRONTAL.copy(), OTHER.copy()
        regions = np.array([f'Demo_{i+1:02d}' for i in range(80)])
        provenance = {
            'configuration': 'demo',
            'seed': settings['seed'],
            'description': 'Synthetic symmetric network; arbitrary demo scaling.',
            'reproduces_study': False,
        }
    elif settings['kind'] == 'file':
        # Paths in the configuration are relative to the repository root.
        source = path.parent.parent / settings['path']
        if not source.is_file():
            raise FileNotFoundError(
                f'Original connectome missing: {source}. '
                'Provide the authorized study file or select configs/demo.json.'
            )
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        if checksum != settings['sha256']:
            raise ValueError('The file does not match the original study connectome.')
        with np.load(source, allow_pickle=False) as saved:
            weights = saved['W'].astype(float)
            frontal = saved['frontal_indices'].copy()
            other = saved['no_frontal_indices'].copy()
            regions = saved['regions'].copy()
        # Preserve the original weights: no new scaling or symmetrization.
        provenance = {
            'configuration': 'study',
            'connectome_sha256': checksum,
            'description': 'Verified original study network; unchanged weights.',
            'reproduces_study': 'Network verified; full reproduction also requires original simulation and inference settings.',
        }
    else:
        raise ValueError('Unknown connectome configuration.')

    check_connectome(weights, frontal, other, regions)
    return dict(W=weights, frontal_indices=frontal,
                no_frontal_indices=other, regions=regions,
                provenance=provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/demo.json')
    args = parser.parse_args()
    network = load_connectome(args.config)
    print(json.dumps(network['provenance'], indent=2))
    print('Matrix:', network['W'].shape)
    print('Motor-frontal regions:', len(network['frontal_indices']))
    print('Other regions:', len(network['no_frontal_indices']))
