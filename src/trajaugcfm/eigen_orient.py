import jaxtyping as jt
import numpy as np

type Shape = tuple[int, ...]
type MatBatchReal = jt.Real[np.ndarray, '*batch d d']
type MatBatchInt = jt.Int[np.ndarray, '*batch d']
type VecBatchReal = jt.Real[np.ndarray, '*batch d']
type BatchReal = jt.Real[np.ndarray, '*batch']


def orient_eigenvectors(Q: MatBatchReal) -> tuple[MatBatchReal, MatBatchInt, MatBatchReal]:
    '''Returns oriented eigenvectors Qor and corresponding sign flip vector S.

    This re-implementation:
        - Enables orienting a BATCH of eigenvectors Q and
          their corresponding eigenvalues E
        - Only the arctan2 method is re-implemented
        - Forces orient to first orthant for consistent sign alignment

    It is the user's responsibility to:
        - Check Q and E are valid corresponding eigenvectors/jjkvalues
        - Sort both Q and E in DESCending order

    The user must sort both Q and E in DESCending order prior to calling this function.

    This function returns a tuple (Qor, S, Thetas).
    Qor is the oriented batch of eigenvectors.
    S is the batch of sign flip vectors transforming Q to Qor.
    Thetas is the batch of angles matrix to reorient I to Qor

    "A Consistently Oriented Basis for Eigenanalysis: Improved Directional Statistics"
    International Journal of Data Science and Analytics 20.3: 1899-1913
    Damask, Jay (2025)

    This implementation is based off the canonical implementation found at:
    github.com/thucyd-dev/thucyd

    Args:
        Q: batch of eigenvectors where Q[batch_idx, :, i] is the ith eigenvector

    Returns:
        Qor: oriented batch of eigenvectors where Qor[batch_idx, :, i]
             is the ith oriented eigenvector
        S:   batch of sign flips to used in orienting Q
        Thetas: batch of angles used to orient Q
    '''
    batched = Q.ndim > 2
    if not batched:
        ## add dummy batch
        Q = Q[None]

    ## set up working variables
    batch = Q.shape[:-2]
    d = Q.shape[-2]

    Qwork = Q.copy()  ## working copy
    Thetas = np.zeros_like(Qwork)  ## angles
    S = np.ones((*batch, d), dtype=int)
    ptrs = np.arange(d)

    ## orient to first orthant
    orthant_flip_mask = Qwork[..., 0, 0] < 0.0
    S[orthant_flip_mask, 0] = -1
    Qwork[orthant_flip_mask, :, 0] *= S[orthant_flip_mask, None, 0]

    ## iterate over reducible subspaces
    for ptr in ptrs[:-1]:
        Qwork, Thetas_col = reduce_dimension_by_one(d, ptr, batch, Qwork)
        ## gather in upper triangle
        Thetas[..., ptr, :] = Thetas_col

    ## flip sign if necessary for last dimension
    irreducible_mask = Qwork[..., -1, -1] < 0.0
    S[irreducible_mask, -1] = -1

    ## compute oriented Qor
    Qor = Q * S[..., None, :]  # multiply sign to each column

    if not batched:
        ## remove dummy batch
        Qor = Qor[0]
        S = S[0]
        Thetas = Thetas[0]

    ## return everything
    return Qor, S, Thetas


def reduce_dimension_by_one(
    d: int,
    ptr: int,
    batch: Shape,
    Qwork: MatBatchReal
) -> tuple[MatBatchReal, VecBatchReal]:
    '''Reduce subspace to align by 1 dimension.

    Reduce the subspace by finding the rotation matrix R such that the rotation
    R.T @ Q updates the block structure in the following way:

    | I_k        |    | I_k+1          |
    |      W_d-k |    |        W_d-k-1 |

          Q                R.T @ Q

    Where I_k is the (k * k) identity matrix and W_k is the partially oriented
    eigenvector block.

    When Qwork is no longer reducible, we should end up with the matrix

    | I_d-1        |
    |        +/- 1 |

    Args:
        d: dimension
        ptr: pointer to current dimension
        batch: shape of batch
        Qwork: in-progress working copy of oriented eigenvectors

    Returns:
        Qwork: updated Qwork with number of subspaces to orient reduced by 1
    '''
    ## irreducible
    if ptr == d - 1:
        Thetas_col = np.zeros((*batch, d))
        return Qwork, Thetas_col

    ## reducible subspace
    Thetas_col = solve_rotation_angles_in_subdimension_via_arctan2(
        d, ptr, batch, Qwork[..., ptr]
    )

    # construct rotation matrix
    R = construct_subspace_rotation_matrix(d, ptr, batch, Thetas_col)

    Qwork = R.swapaxes(-1, -2) @ Qwork

    return Qwork, Thetas_col


def solve_rotation_angles_in_subdimension_via_arctan2(
    d: int,
    ptr: int,
    batch: tuple[int, ...],
    Qcol: VecBatchReal
) -> VecBatchReal:
    '''Solves for angles used to reduce Qcol dim by 1 using rotations.

    This function uses arctan2 to solve for the angles.

    Needs to handle certain edge cases to avoid dividing by 0 in arctan2.

    Args:
        d: dimension
        ptr: pointer to current dimension
        batch: shape of batch
        Qcol: batch of eigenvectors corresponding to current dimension

    Returns:
        Thetas_col: batch of vectors of angles which orient Qcol
    '''
    ## irreducible
    if ptr == d - 1:
        return np.zeros((*batch, d))

    ## flatten for easier indexing
    Qcol_flat = Qcol.reshape((-1, d))
    b = Qcol_flat.shape[0]

    sub_ptrs = np.arange(ptr, d)
    Thetas_col = np.zeros((b, d))

    ## first angle, maybe major arc
    sub_ptr_tail = sub_ptrs[0]
    sub_ptr_head = sub_ptrs[1]
    Thetas_col[:, sub_ptr_head] = np.arctan2(
        Qcol_flat[:, sub_ptr_head], Qcol_flat[:, sub_ptr_tail]
    )

    ## split into fast and slow using masks
    eps_value = np.finfo(np.float64).eps
    ## check ptr+1: because ptr is handled in first angle
    ## and anything prior should be 0 from previous iterations
    fast_mask = np.all(np.abs(Qcol_flat[:, ptr+1:]) > eps_value, axis=-1)
    slow_mask = ~fast_mask
    fast_idx = np.nonzero(fast_mask)[0]
    slow_idx = np.nonzero(slow_mask)[0]

    ## fast path
    if fast_idx.shape[0] > 0:
        for sub_ptr_head in sub_ptrs[2:]:
            sub_ptr_tail = sub_ptr_head - 1
            Thetas_col[fast_idx, sub_ptr_head] = np.arctan2(
                Qcol_flat[fast_idx, sub_ptr_head]
                * np.abs(np.sin(Thetas_col[fast_idx, sub_ptr_tail])),
                np.abs(Qcol_flat[fast_idx, sub_ptr_tail])
            )

    ## slow path
    if slow_idx.shape[0] > 0:
        ## need to track separate tails for each Qcol in the batch
        tails_batch = np.full(slow_idx.shape[0], ptr, dtype=int)
        for sub_ptr_head in sub_ptrs[2:]:
            ## advance tail cursor if possible
            adv_mask = np.abs(Qcol_flat[slow_idx, sub_ptr_head-1]) > eps_value
            tails_batch[adv_mask] = sub_ptr_head - 1

            ## compute some common masks
            heads_nonzero = np.abs(Qcol_flat[slow_idx, sub_ptr_head]) > eps_value
            tails_positive = tails_batch > 0

            ## compute theta value
            ## general case
            general_mask = heads_nonzero & tails_positive
            general_case = slow_idx[general_mask]
            general_tails = tails_batch[general_mask]
            Thetas_col[general_case, sub_ptr_head] = np.arctan2(
                Qcol_flat[general_case, sub_ptr_head]
                * np.abs(np.sin(Thetas_col[general_case, general_tails])),
                np.abs(Qcol_flat[general_case, general_tails])
            )

            ## special case
            special_mask = heads_nonzero & ~tails_positive
            special_case = slow_idx[special_mask]
            special_tails = tails_batch[special_mask]
            Thetas_col[special_case, sub_ptr_head] = np.arctan2(
                Qcol_flat[special_case, sub_ptr_head],
                np.abs(Qcol_flat[special_case, special_tails])
            )

            ## zero case handled by default from Thetas_col initialization

    return Thetas_col.reshape((*batch, d))


def construct_subspace_rotation_matrix(
    d: int,
    ptr: int,
    batch: Shape,
    Thetas_col: VecBatchReal
) -> MatBatchReal:
    '''Construct R using a sequence of Givens rotations.

    R is the total rotation from repeatedly applying Givens rotations.
    The angles for the Givens rotations are supplied by Thetas_col.

    R = G_ptr @ G_ptr+1 ... @ G_d-1 @ G_d

    for G_i the ith Givens rotation.

    Args:
        d: dimension
        ptr: pointer to current dimension
        batch: shape of batch
        Thetas_col: batch of vectors of angles

    Returns:
        R: batched total rotation matrix
    '''
    ## initialize reference rotation matrix R with broadcastable batch dims
    R = np.expand_dims(np.eye(d), tuple(np.arange(len(batch))))

    ## iterate backwards, build batch of Givens matrices and apply
    sub_ptrs = np.arange(ptr+1, d)
    for sub_ptr in sub_ptrs[::-1]:
        ## call make_givens_rotation_in_subspace()
        R = make_givens_rotation_matrix_in_subspace(
            d, ptr, sub_ptr, batch, Thetas_col[..., sub_ptr]
        ) @ R

    return R


def make_givens_rotation_matrix_in_subspace(
    d: int,
    ptr: int,
    sub_ptr: int,
    batch: Shape,
    thetas: BatchReal
) -> MatBatchReal:
    '''Compute a batch of Givens matrices from a batch of angles.

    A Givens matrix G looks like

               | .             |
               |   .           |
        ptr -> |     c   -s    |
               |       .       |
    sub_ptr -> |     s   c     |
               |           .   |
               |             . |
                     ^   ^
                     |   |
                    ptr  |
                      sub_ptr

    Args:
        d: dimension
        ptr: pointer to current dimension
        sub_ptr: pointer to alignment dimension
        batch: shape of batch
        thetas: batch of angles aligning current dimension to alignment dimension

    Returns:
        G: batch of givens rotations
    '''
    c = np.cos(thetas)
    s = np.sin(thetas)

    ## copy() because broadcast_to() returns a read-only view
    G = np.broadcast_to(np.eye(d), (*batch, d, d)).copy()

    ## fill in Givens matrix
    G[..., ptr, ptr] = c
    G[..., ptr, sub_ptr] = -s
    G[..., sub_ptr, ptr] = s
    G[..., sub_ptr, sub_ptr] = c

    return G


def main() -> None:
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from trajaugcfm.constants import DATADIR

    ## Load data
    datafile = os.path.join(DATADIR, 'easy', 'data.npy')
    data = np.load(datafile)
    N, T, d = data.shape

    ## Extract trajectories and compute eigenanalysis
    obsidx = [0, 1, 2]
    # obsidx = [0, 1]
    do = len(obsidx)
    Xref = data[:, :, obsidx]
    mu = Xref.mean(axis=0, keepdims=True)
    Xref_centered = Xref - mu
    covs = np.einsum('nti,ntj->tij', Xref_centered, Xref_centered)
    covs /= N - 1
    reg = 1e-8
    covs += np.eye(do)[None] * reg  ## regularize just in case
    L, Q = np.linalg.eigh(covs)
    L = np.sqrt(np.flip(L, axis=-1))
    Q = np.flip(Q, axis=-1)
    Qor, S, Thetas = orient_eigenvectors(Q)

    ## Set up time points to plot
    k = 9
    tspan = np.linspace(0, T-1, k).astype(int)

    def plot_eigvals(subfig, L, tspan, T, yscale='linear'):
        ax = subfig.gca()
        ax.set_title(f'Eigenvalues over time ({yscale} scale)')
        ax.set_yscale(yscale)
        ax.grid(axis='y', c='k', alpha=0.3)
        if yscale == 'linear':
            ## if yscale is log, y=0 is impossible since lim_y=0+ log(y) = -inf
            ax.axhline(y=0, c='k')
        for t in tspan:
            ax.axvline(x=t, linestyle='--', c='r', alpha=0.5)
        for i in range(do):
            ax.plot(np.arange(T), L[:, i], label=fr'$\lambda_{i+1}$')
        ax.legend(loc='best')

    def plot_eigvecs_2d(subfig, L, Q, tspan, T):
        subfig.suptitle('Eigenvectors (Axes scaled to box [(-1, -1), (1, 1)])')
        axs = subfig.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
        axs[0, 0].set_xlim((-1, 1))
        axs[0, 0].set_ylim((-1, 1))
        angles = np.degrees(np.arctan2(Q[:, 1, 0], Q[:, 0, 0]))
        for i, t in enumerate(tspan):
            ax = axs[*divmod(i, 3)]
            realt = t / T
            ax.set_title(f'$t = {realt:.2f}$ ({t} / {T-1})')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            ax.axhline(0, alpha=0.3, c='k')
            ax.axvline(0, alpha=0.3, c='k')

            ellipse = Ellipse(
                xy=(0, 0),
                width=2*L[t, 0],
                height=2*L[t, 1],
                angle=angles[t],  # type: ignore
                color='peachpuff',
                alpha=0.5
            )
            ax.add_patch(ellipse)

            Qt_scaled = Q[t] * L[t]
            if i == 0:
                label1 = r'$v_1$'
                label2 = r'$v_2$'
            else:
                label1 = None
                label2 = None
            ax.quiver(
                0, 0,
                Qt_scaled[0, 0], Qt_scaled[1, 0],
                angles='xy', scale_units='xy', scale=1,
                color='tab:blue', label=label1
            )
            ax.quiver(
                0, 0,
                Qt_scaled[0, 1], Qt_scaled[1, 1],
                angles='xy', scale_units='xy', scale=1,
                color='tab:green', label=label2
            )
            if i == 0:
                ax.legend()

    def plot_eigvecs_3d(subfig, L, Q, tspan, T):
        subfig.suptitle('Eigenvectors (Axes scaled to box [(-1, -1, -1), (1, 1, 1)])')
        axs = subfig.subplots(nrows=3, ncols=3, subplot_kw={'projection': '3d'})
        for i, t in enumerate(tspan):
            ax = axs[*divmod(i, 3)]
            realt = t / T
            ax.set_title(f'$t = {realt:.2f}$ ({t} / {T-1})')
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.set_zlim((-1, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))

            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            x = L[t, 0] * np.outer(np.cos(u), np.sin(v))
            y = L[t, 1] * np.outer(np.sin(u), np.sin(v))
            z = L[t, 2] * np.outer(np.ones_like(u), np.cos(v))
            for k in range(len(u)):
                for ell in range(len(v)):
                    [x[k, ell], y[k, ell], z[k, ell]] = Q[t] @ [x[k, ell], y[k, ell], z[k, ell]] + [0, 0, 0]
            ax.plot_surface(x, y, z, color='peachpuff', alpha=0.5)

            Qt_scaled = Q[t] * L[t]
            colors = [f'tab:{c}' for c in ['blue', 'green', 'red']]
            labels = [r'$v_1$', r'$v_2$', r'$v_3$'] if i == 0 else [None, None, None]
            for j in range(3):
                ax.quiver(
                    0, 0, 0,
                    Qt_scaled[0, j], Qt_scaled[1, j], Qt_scaled[2, j],
                    color=colors[j],
                    label=labels[j],
                    arrow_length_ratio=0.1
                )

            if i == 0:
                ax.legend()

    ## Divide figure into two sections
    fig = plt.figure(layout='constrained', figsize=(10, 14))
    subfigs = fig.subfigures(nrows=2, hspace=0.05, height_ratios=[1, 3])
    yscale = 'linear'
    # yscale = 'log'

    ## Top figure plots eigenvalues over time
    plot_eigvals(subfigs[0], L, tspan, T, yscale=yscale)

    ## Bottom figure plots eigenvectors at specified timepoints
    ## This figure is subdivided into a 3 x 3 grid
    if do == 2:
        plot_eigvecs_2d(subfigs[1], L, Q, tspan, T)
    else:
        plot_eigvecs_3d(subfigs[1], L, Q, tspan, T)
    fig.savefig(f'eigs_{do}d.png')
    fig.savefig(f'eigs_{do}d.pdf')
    plt.close(fig)

    ## Plot identical figure but with the oriented eigenvectors
    fig = plt.figure(layout='constrained', figsize=(10, 14))
    subfigs = fig.subfigures(nrows=2, hspace=0.05, height_ratios=[1, 3])

    ## Plot eigenvalues over time
    plot_eigvals(subfigs[0], L, tspan, T, yscale=yscale)

    ## Plot Eigenvectors at specified timepoints
    if do == 2:
        plot_eigvecs_2d(subfigs[1], L, Qor, tspan, T)
    else:
        plot_eigvecs_3d(subfigs[1], L, Qor, tspan, T)
    fig.savefig(f'eigs_oriented_{do}d.png')
    fig.savefig(f'eigs_oriented_{do}d.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
