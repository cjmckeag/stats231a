import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
import seaborn as sns

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 10)


class tSNE:
    """
    Performs t-Distributed Stochastic Neighbor Embedding dimensionality reduction. Represents high-dim data in a low-dim
     space so that it can be visualized. Creates a reduced feature space where similar samples are modeled by nearby
      points and dissimilar samples are modeled by distant points with high probability.
    """
    def __init__(self, n_components=2, perplexity=30):
        """
        Constructs necessary parameters for tSNE.
        :param n_components: num components to reduce data to
        :param perplexity: related to num of nearest neighbors used in tSNE algo
        """
        # work with hand-drawn digits
        self.X, self.y = load_digits(return_X_y=True)
        # smallest representable positive number s.t. 1 + epsilon != 1
        self.MACHINE_EPSILON = np.finfo(np.double).eps
        self.n_components = n_components
        self.perplexity = perplexity

    def __call__(self):
        """
        Fits tSNE transformation and plots clusters.
        :return: nothing
        """
        X_embedded = self.fit()
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=self.y, legend='full', palette=palette)

    def fit(self):
        """
        Performs tSNE transformation. Uses joint probability distribution to create reduced feature space.
        :return: tSNE transformed embedding
        """
        # store the number of samples
        n_samples = self.X.shape[0]

        # Compute euclidean distance btwn each data point
        distances = pairwise_distances(self.X, metric='euclidean', squared=True)

        # Compute joint probabilities p_ij from distances.
        P = _joint_probabilities(distances=distances, desired_perplexity=self.perplexity, verbose=False)

        # create reduced feature space using randomly selected Gaussian values
        # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
        X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, self.n_components).astype(np.float32)

        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self.tsne(P, degrees_of_freedom, n_samples, X_emb=X_embedded)

    def tsne(self, P, degrees_of_freedom, n_samples, X_emb):
        """
        Applies gradient descent on parameters and returns shaped embedding.
        :param P: current parameters
        :param degrees_of_freedom: n_components - 1
        :param n_samples: number of samples
        :param X_emb: current embedding
        :return: updated embedding
        """
        # flatten into 1d array
        params = X_emb.ravel()
        # def kl divergence as objective function
        obj_func = self.kl_divergence
        # use gradient descent to minimize the kl divergence
        params = self.gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, self.n_components])
        # change embedding back into a 2d array
        X_emb = params.reshape(n_samples, self.n_components)
        return X_emb

    @staticmethod
    def gradient_descent(obj_func, p0, args, it=0, n_iter=1000, n_iter_without_progress=300, momentum=0.8,
                         learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7):
        """
        Updates the values in the embedding by minimizing the KL divergence. Stops either when the gradient norm is
         below the threshold or when we reach the maximum number of iterations without making any progress.
        :param obj_func: objective function (KL divergence)
        :param p0: current parameters
        :param args: other parameters
        :param it: starting iteration
        :param n_iter: number of iterations
        :param n_iter_without_progress: if no progress, num iterations to stop after
        :param momentum: momentum
        :param learning_rate: learning rate
        :param min_gain: minimum value for clipping gains array
        :param min_grad_norm: threshold value for grad norm
        :return: updated params
        """
        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        best_error = np.finfo(np.float).max
        best_iter = it

        for i in range(it, n_iter):
            error, grad = obj_func(p, *args)
            grad_norm = linalg.norm(grad)
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update
            if (i % 100) == 0:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      % (i + 1, error, grad_norm))
            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break

            if grad_norm <= min_grad_norm:
                break
        return p

    def kl_divergence(self, params, P, degrees_of_freedom, n_samples, n_components):
        """
        Computes error in the form of the KL divergence and the gradient.
        :param params: current params
        :param P: other params
        :param degrees_of_freedom: degrees of freedom
        :param n_samples: number of samples
        :param n_components: number of components
        :return: kl_divergence, grad
        """
        X_embedded = params.reshape(n_samples, n_components)
        # calculate the probability distribution over the points in the low-dim map
        dist = pdist(X_embedded, "sqeuclidean")
        dist /= degrees_of_freedom
        dist += 1.
        dist **= (degrees_of_freedom + 1.0) / -2.0
        Q = np.maximum(dist / (2.0 * np.sum(dist)), self.MACHINE_EPSILON)

        # Kullback-Leibler divergence of P and Q
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, self.MACHINE_EPSILON) / Q))

        # Gradient: dC/dY
        grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
        PQd = squareform((P - Q) * dist)
        for i in range(n_samples):
            grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                             X_embedded[i] - X_embedded)
        grad = grad.ravel()
        c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
        grad *= c

        return kl_divergence, grad
