from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


@dataclass
class GMMParameters:
    """Dataclass of GMM parameters.

    Attributes
    ----------
        weights: np.ndarray.
            Weights of the difference gaussians.

        means: np.ndarray.
            Means of the difference gaussians.

        covariances: np.ndarray.
            Covariances of the difference gaussians (i.e. squared ones).

        likelihood: np.ndarray.
            Likelihood of the different iterations of learning

        bic: float.
            The BIC (Bayesian Information Criterion) of the model.

    """

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    likelihood: np.ndarray
    bic: float


@dataclass
class BinnedArray:
    n_bins: np.ndarray
    intervals: np.ndarray


class GMM:
    """Class that extend sklearn one to fit 1D binned data."""

    def __init__(
        self,
        n_components: int = 2,
        means: Optional[np.ndarray] = None,
        covariances: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        max_iter: int = 100,
        max_tries: int = 100,
        **kwargs_gmm
    ):
        """Instantiate a GMM object.

        Params
        ------
            n_components: int. Defaults to 2.
                Number of components of the GMM.

            means: np.ndarray. Defaults to None.
                Means of the difference gaussians. If None, need to be fit
                before predict.

            covariances: np.ndarray. Defaults to None.
                Covariances of the difference gaussians (i.e. squared ones).
                If None, need to be fit before predict.

            weights: np.ndarray. Defaults to None.
                Weights of the difference gaussians. If None, need to be fit
                before predict.

            max_iter: int. Defaults to 100.
                Maximum number of iterations of the EM algorithm.

            max_tries: int. Defaults to 100.
                Maximum number of tries if learning fails.

        """
        self.n_components = n_components
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.max_iter = max_iter
        self.max_tries = max_tries
        self._estimator = GaussianMixture(
            n_components=n_components, max_iter=max_iter, **kwargs_gmm
        )

    def _standard_fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "GMM":
        """Fit the GMM using standard fit mechanism of sklearn.

        Params
        ------
            x: np.ndarray
                The training features.

            y: np.ndarray. Defaults to None.
                The training targets if needed.

        """
        self._estimator.fit(x, y)
        self.means_ = self._estimator.means_
        self.covariances_ = self._estimator.covariances_
        self.weights_ = self._estimator.weights_
        return self

    def _binned_fit(self, n_bins, intervals) -> "GMM":
        """Fit the GMM using binned fit mechanism.

        The corresponding formula are available in the thesis of Maxime BAELDE:
        https://hal.archives-ouvertes.fr/tel-02399081

        Params
        ------
            n_bins: np.ndarray
                The training bins.

            intervals: np.ndarray.
                The training intervals.

        """
        likelihood = -np.inf * np.ones((self.max_tries, self.max_iter))
        n = n_bins.astype(int).sum()
        n_bins_r = n_bins[np.newaxis, :]
        bic = np.zeros(self.max_tries)
        models = []
        for i in range(self.max_tries):
            old_means = np.random.choice(
                range(-10, 10), replace=False, size=self.n_components
            )
            old_covariances = np.ones(self.n_components)
            old_weights = np.random.rand(self.n_components)
            old_weights /= old_weights.sum()

            for j in range(self.max_iter):
                old_weights = old_weights[:, np.newaxis]
                old_covariances = old_covariances[:, np.newaxis]

                pdfs = np.array(
                    [
                        norm.pdf(intervals, loc=mu, scale=sigma)
                        for mu, sigma in zip(old_means, np.sqrt(old_covariances))
                    ]
                )
                cdfs = np.array(
                    [
                        norm.cdf(intervals, loc=mu, scale=sigma)
                        for mu, sigma in zip(old_means, np.sqrt(old_covariances))
                    ]
                )

                Fs = (old_weights * cdfs).sum(0)
                Fs_diff = np.diff(Fs)[np.newaxis, :]

                H_weights = np.diff(cdfs)
                H_means = np.diff(pdfs)
                H_covariances = np.diff(intervals * pdfs)

                G_weights = H_weights.copy()
                expectations_weights = old_weights * G_weights / Fs_diff
                new_weights = (n_bins_r * expectations_weights).sum(1) / n

                G_means = (
                    old_means[:, np.newaxis] * H_weights - old_covariances * H_means
                )
                expectations_means = old_weights * G_means / Fs_diff
                new_means = (n_bins_r * expectations_means).sum(1) / (
                    n_bins_r * expectations_weights
                ).sum(1)

                G_covariances = (
                    old_covariances
                    * (
                        H_weights
                        + (2 * new_means - old_means)[:, np.newaxis] * H_means
                        - H_covariances
                    )
                    + ((new_means - old_means) ** 2)[:, np.newaxis] * H_weights
                )
                expectations_covariances = old_weights * G_covariances / Fs_diff
                new_covariances = (n_bins_r * expectations_covariances).sum(1) / (
                    n_bins_r * expectations_weights
                ).sum(1)

                if (
                    np.isnan(new_means).any()
                    or np.isnan(new_covariances).any()
                    or np.isnan(new_weights).any()
                ):
                    print("nan, try again")
                    break

                old_means = new_means.copy()
                old_covariances = new_covariances.copy()
                old_weights = new_weights.copy()

                p_j = np.diff(
                    np.array(
                        [
                            weight * norm.cdf(intervals, loc=mu, scale=sigma)
                            for mu, sigma, weight in zip(
                                old_means, np.sqrt(old_covariances), old_weights
                            )
                        ]
                    ).sum(0)
                )
                likelihood[i, j] = (
                    np.log(range(1, n + 1)).sum()
                    - np.array(
                        [np.log(range(1, k + 1)).sum() for k in n_bins.astype(int)]
                    ).sum()
                    + (n_bins * np.log(p_j / p_j.sum())).sum()
                )
            bic[i] = -2 * likelihood[i, j] * n + (3 * self.n_components) * np.log(n)
            models.append(
                GMMParameters(
                    new_weights, new_means, new_covariances, likelihood[i, :], bic[i]
                )
            )

        self.best_model_ = np.argmax(likelihood[:, -1])
        self.weights_ = models[self.best_model_].weights
        self.means_ = models[self.best_model_].means
        self.covariances_ = models[self.best_model_].covariances
        self.likelihood_ = models[self.best_model_].likelihood
        self.bic_ = models[self.best_model_].bic

        return self

    def fit(
        self,
        x: Union[np.ndarray, BinnedArray],
        y: Optional[np.ndarray] = None,
        from_binned_data=False,
    ) -> "GMM":
        """Fit the GMM using training data.

        Params
        ------
            x: np.ndarray or BinnedArray.
                The training features or BinnedArray.

            y: np.ndarray. Defaults to None.
                The training targets if needed.


        """
        if from_binned_data:
            self._binned_fit(x.n_bins, x.intervals)
            self.fit_from_binned_data_ = True
        else:
            self._standard_fit(x, y)
            self.fit_from_binned_data_ = False

        self.weights = self.weights_.copy()
        self.means = self.means_.copy()
        self.covariances = self.covariances_.copy()

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict outcomes from samples.

        Params
        ------
            x: np.ndarray.
                An array of samples for which compute outcomes.

        Returns:
            An ndarray of outcomes.

        """
        if self.fit_from_binned_data_:
            probas = self.predict_proba(x)
            return np.argmax(probas, 1)
        else:
            return self._estimator.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probability outcomes from samples.

        Params
        ------
            x: np.ndarray.
                An array of samples for which compute probabilities.

        Returns:
            An ndarray of probability outcomes.

        """
        if self.fit_from_binned_data_:
            log_weights = np.log(self.weights_)[np.newaxis, :]
            log_pdfs = np.array(
                [
                    norm.logpdf(x.squeeze(), loc=mu, scale=sigma)
                    for mu, sigma in zip(self.means_, np.sqrt(self.covariances_))
                ]
            ).T
            weighted_log_prob = log_weights + log_pdfs
            log_prob_norm = logsumexp(weighted_log_prob, axis=1)
            with np.errstate(under="ignore"):
                # ignore underflow
                log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
            probas = np.exp(log_resp)
        else:
            probas = self._estimator.predict_proba(x)
        return probas

    def sample(self, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from the GMM.

        Each gaussian has a probability self.weights[i] to be drawn.
        One the gaussian is selected (from a uniform draw), a sample from the
        corresponding gaussian is drawn.

        Params
        ------
            n: int.
                Number of samples to draw from the GMM.

        Returns
        -------
            A 2-tuple composed of a ndarray containing samples from the GMM and a
            ndarray containing the corresponding component label.

        """
        if self.means is None or self.covariances is None or self.weights is None:
            raise ValueError("GMM must have been fit before sample from it.")

        x = np.zeros((n, 1))
        y = np.zeros(n)
        # Compute cumsum of weights to optimize search of which gaussian to draw samples from
        cumsum_weights = np.hstack([0, np.cumsum(self.weights)])
        for i in range(n):
            random_choice = np.random.rand()
            mixture_choice = np.where(~(random_choice - cumsum_weights > 0))[0][0] - 1
            x[i] = np.random.normal(
                loc=self.means[mixture_choice],
                scale=np.sqrt(self.covariances[mixture_choice]),
            )
            y[i] = mixture_choice

        return x, y
