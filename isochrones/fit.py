"""MCMC fitting utilities using emcee3.

This module provides classes and functions for performing Markov Chain Monte Carlo
(MCMC) fitting of stellar models to observational data using the emcee3 sampler.

Classes:
    Emcee3Model: emcee3 model class with prior and likelihood computation.
    Emcee3PriorModel: emcee3 model class for prior-only sampling.

Functions:
    write_samples: Save MCMC samples to HDF5 file.
    fit_emcee3: Run MCMC fitting using emcee3.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional, Any, Dict, Tuple

import pandas as pd
import numpy as np

import emcee3
from emcee3.backends import Backend, HDFBackend

if TYPE_CHECKING:
    from .starmodel import StarModel


class Emcee3Model(emcee3.Model):
    """emcee3 model class for star model fitting.

    This class wraps a StarModel to provide the log prior and log likelihood
    computation methods required by emcee3.
    """

    def __init__(self, mod: "StarModel", *args: Any, **kwargs: Any) -> None:
        """Initialize the emcee3 model.

        Args:
            mod: The StarModel to fit.
            *args: Additional positional arguments passed to parent.
            **kwargs: Additional keyword arguments passed to parent.
        """
        self.mod = mod
        super().__init__(*args, **kwargs)

    def compute_log_prior(self, state: emcee3.State) -> emcee3.State:
        """Compute the log prior probability.

        Args:
            state: The current sampler state with coordinates.

        Returns:
            State with log_prior attribute set.
        """
        state.log_prior = self.mod.lnprior(state.coords)
        return state

    def compute_log_likelihood(self, state: emcee3.State) -> emcee3.State:
        """Compute the log likelihood.

        Args:
            state: The current sampler state with coordinates.

        Returns:
            State with log_likelihood attribute set.
        """
        state.log_likelihood = self.mod.lnlike(state.coords)
        return state


class Emcee3PriorModel(emcee3.Model):
    """emcee3 model class for prior-only sampling.

    This class computes only the log prior, setting log likelihood to zero.
    Useful for testing priors or burn-in phases.
    """

    def __init__(self, mod: "StarModel", *args: Any, **kwargs: Any) -> None:
        """Initialize the prior-only emcee3 model.

        Args:
            mod: The StarModel with prior to sample.
            *args: Additional positional arguments passed to parent.
            **kwargs: Additional keyword arguments passed to parent.
        """
        self.mod = mod
        super().__init__(*args, **kwargs)

    def compute_log_prior(self, state: emcee3.State) -> emcee3.State:
        """Compute the log prior probability.

        Args:
            state: The current sampler state with coordinates.

        Returns:
            State with log_prior attribute set.
        """
        state.log_prior = self.mod.lnprior(state.coords)
        return state

    def compute_log_likelihood(self, state: emcee3.State) -> emcee3.State:
        """Compute the log likelihood (always zero for prior-only sampling)."""
        state.log_likelihood = 0
        return state


def write_samples(mod: "StarModel", df: pd.DataFrame, resultsdir: str = "results") -> None:
    """Save MCMC samples to an HDF5 file.

    Args:
        mod: The StarModel that was fitted.
        df: DataFrame containing MCMC samples.
        resultsdir: Directory to save the samples file. Defaults to "results".
    """
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    samplefile = os.path.join(resultsdir, "{}.h5".format(mod.name))
    df.to_hdf(samplefile, key="samples")


def fit_emcee3(
    mod: "StarModel",
    nwalkers: int = 500,
    verbose: bool = False,
    nsamples: int = 5000,
    targetn: float = 4,
    iter_chunksize: int = 200,
    pool: Optional[Any] = None,
    overwrite: bool = False,
    maxiter: int = 10,
    sample_directory: str = "mcmc_chains",
    nburn: float = 2,
    mixedmoves: bool = True,
    resultsdir: str = "mcmc_results",
    prior_only: bool = False,
    **kwargs: Any
) -> pd.DataFrame:
    """Fit a star model using emcee3 MCMC sampler.

    This function runs MCMC sampling to infer stellar parameters from
    observational data. It uses automatic convergence checking based on
    autocorrelation time.

    Args:
        mod: The StarModel to fit.
        nwalkers: Number of walkers. Defaults to 500.
        verbose: If True, print progress information. Defaults to False.
        nsamples: Target number of samples to keep. Defaults to 5000.
        targetn: Target effective sample size (in units of autocorrelation time).
            Defaults to 4.
        iter_chunksize: Number of samples per iteration. Defaults to 200.
        pool: Optional pool for parallel sampling.
        overwrite: If True, overwrite existing chains. Defaults to False.
        maxiter: Maximum number of iterations. Defaults to 10.
        sample_directory: Directory to store chain files. Defaults to "mcmc_chains".
        nburn: Number of autocorrelation times to discard as burn-in.
            Defaults to 2.
        mixedmoves: If True, use mixed moves (KDEMove, DEMove, DESnookerMove).
            Defaults to True.
        resultsdir: Directory to save results. Defaults to "mcmc_results".
        prior_only: If True, only sample from the prior. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        DataFrame with MCMC samples.

    Raises:
        ValueError: If model initialization fails.
    """

    # Initialize
    if prior_only:
        walker = Emcee3PriorModel(mod)
    else:
        walker = Emcee3Model(mod)
    ndim = mod.n_params

    if sample_directory is not None:
        sample_file = os.path.join(sample_directory, "{}.h5".format(mod.name))
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)
        backend = HDFBackend(sample_file)
        try:
            coords_init = backend.current_coords
        except (AttributeError, KeyError):
            coords_init = mod.sample_from_prior(nwalkers, require_valid=True, values=True)
    else:
        backend = Backend()
        coords_init = mod.sample_from_prior(nwalkers, require_valid=True, values=True)

    if mixedmoves:
        moves = [
            (emcee3.moves.KDEMove(), 0.4),
            (emcee3.moves.DEMove(1.0), 0.4),
            (emcee3.moves.DESnookerMove(), 0.2),
        ]
    else:
        moves = emcee3.moves.KDEMove()

    sampler = emcee3.Sampler(moves, backend=backend)
    if overwrite:
        sampler.reset()
        coords_init = mod.sample_from_prior(nwalkers, require_valid=True, values=True)

    if pool is None:
        from emcee3.pools import DefaultPool

        pool = DefaultPool()

    try:
        ensemble = emcee3.Ensemble(walker, coords_init, pool=pool)
    except ValueError:
        import pdb

        pdb.set_trace()

    def calc_stats(s):
        """returns tau_max, neff
        """
        tau = s.get_integrated_autocorr_time(c=1)
        tau_max = tau.max()
        neff = s.backend.niter / tau_max - nburn
        if verbose:
            print("Maximum autocorrelation time: {0}".format(tau_max))
            print("N_eff: {0} ({1})\n".format(neff * nwalkers, neff - nburn))
        return tau_max, neff

    done = False
    if not overwrite:
        try:
            if verbose:
                print("Status from previous run:")
            tau_max, neff = calc_stats(sampler)
            if neff > targetn:
                done = True
        except (emcee3.autocorr.AutocorrError, KeyError):
            pass

    chunksize = iter_chunksize
    for iteration in range(maxiter):
        if done:
            break
        if verbose:
            print("Iteration {0}...".format(iteration + 1))
        sampler.run(ensemble, chunksize, progress=verbose)
        try:
            tau_max, neff = calc_stats(sampler)
        except emcee3.autocorr.AutocorrError:
            tau_max = 0
            continue
        if neff > targetn:
            done = True

    burnin = int(nburn * tau_max)
    ntot = nsamples
    samples = sampler.get_coords(flat=True, discard=burnin)
    total_samples = len(samples)
    if ntot > total_samples:
        ntot = total_samples
    if verbose:
        print("Discarding {0} samples for burn-in".format(burnin))
        print("Randomly choosing {0} samples".format(ntot))
    inds = np.random.choice(total_samples, size=ntot, replace=False)
    samples = samples[inds]

    df = pd.DataFrame(samples, columns=mod.param_names)
    write_samples(mod, df, resultsdir=resultsdir)

    return df
    # return sampler
