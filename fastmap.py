"""FastMap embedding of objects into Euclidean space.

Implements the FastMap algorithm for embedding a collection of objects
into K-dimensional Euclidean space using a user-specified distance function.
"""

import numpy as np
import pickle


class FastMap:
    """Embed a collection of labeled objects into Euclidean space via FastMap.

    Attributes:
        N: Number of objects.
        objects: List of N objects.
        labels: Labels corresponding to each object.
        dist_func: Symmetric distance function taking two objects and
            returning a non-negative real number.
        K: Dimensionality of the embedding (set after calling ``fit``).
        P: (N, K) array of embedded coordinates (set after calling ``fit``).
        pivot_pairs: List of pivot information per dimension.
    """

    def __init__(self, objects, labels, dist_func):
        """Initialise FastMap with objects, labels, and a distance function.

        Args:
            objects: A list of N objects.
            labels: A list/array of labels in the same order as *objects*.
            dist_func: A symmetric distance function ``f(x1, x2) -> float``.
        """
        self.N = len(objects)
        self.objects = objects
        self.labels = labels
        self.dist_func = dist_func

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, K, max_iters=10, e=0, model_name=None):
        """Run FastMap and optionally save the resulting model.

        Args:
            K: Target embedding dimensionality.
            max_iters: Maximum iterations for pivot selection.
            e: Threshold for detecting diminishing returns.
            model_name: If given, save the model to this file path.
        """
        self.fastmap(K, max_iters, e)
        if model_name:
            self.save_model(model_filename=model_name)

    def fastmap(self, K, max_iters, e):
        """Embed objects into K-dimensional Euclidean space.

        Args:
            K: Target dimensionality.
            max_iters: Maximum iterations for pivot identification.
            e: Diminishing-returns threshold.
        """
        self.K = K
        self.P = np.zeros((self.N, K))
        self.pivot_pairs = []

        for k in range(K):
            # --- Fast Pivot ---
            Oa = np.random.randint(self.N)
            Ob = Oa
            for t in range(max_iters):
                d_ai = self.single_source_distances(Oa)
                d_ai_new2 = (
                    np.power(d_ai, 2)
                    - np.sum(np.power(self.P[Oa, :k] - self.P[:, :k], 2), axis=1)
                )
                Oc = np.argmax(d_ai_new2)
                if Oc == Ob:
                    break
                Ob = Oa
                Oa = Oc
                d_ib_new2 = d_ai_new2

            d_ab_new2 = d_ai_new2[Ob]

            if d_ab_new2 <= e:
                # Diminishing returns — zero out this dimension.
                self.P[:, k] = 0
                self.pivot_pairs.append(
                    (Oa, Ob, 1.0, self.P[Oa].copy(), self.P[Ob].copy())
                )
                continue

            d_ab_new = np.sqrt(d_ab_new2)
            self.P[:, k] = (d_ai_new2 + d_ab_new2 - d_ib_new2) / (2 * d_ab_new)
            self.pivot_pairs.append(
                (Oa, Ob, d_ab_new, self.P[Oa].copy(), self.P[Ob].copy())
            )

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def single_source_distances(self, Os):
        """Compute distances from object *Os* to every other object.

        Args:
            Os: Index of the source object.

        Returns:
            A numpy array of length N where entry *i* is the distance
            between ``objects[Os]`` and ``objects[i]``.
        """
        d_si = np.zeros(self.N)
        for i in range(self.N):
            if i != Os:
                d_si[i] = self.dist_func(self.objects[Os], self.objects[i])
        return d_si

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_coord(self, object_q):
        """Compute the FastMap embedding of a new query object.

        Args:
            object_q: The query object (same type as stored objects).

        Returns:
            A numpy array of length K with the embedded coordinates.
        """
        p = np.zeros(self.K)
        for k in range(self.K):
            Oa, Ob, d_ab_cur, Pa, Pb = self.pivot_pairs[k]
            d_aq = self.dist_func(self.objects[Oa], object_q)
            d_qb = self.dist_func(object_q, self.objects[Ob])
            d_aq_cur2 = np.power(d_aq, 2) - np.sum(np.power(Pa[:k] - p[:k], 2))
            d_qb_cur2 = np.power(d_qb, 2) - np.sum(np.power(p[:k] - Pb[:k], 2))
            p[k] = (d_aq_cur2 + np.power(d_ab_cur, 2) - d_qb_cur2) / (2 * d_ab_cur)
        return p

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, model_filename="model.m"):
        """Save model parameters (pivots and embeddings) to a pickle file.

        Args:
            model_filename: Destination file path.
        """
        model_parameters = {
            "pivot_pairs": self.pivot_pairs,
            "P": self.P,
        }
        with open(model_filename, "wb") as model_file:
            pickle.dump(model_parameters, model_file)

    def load_model(self, model_filename):
        """Load model parameters from a previously saved pickle file.

        Args:
            model_filename: Path to the saved model file.
        """
        with open(model_filename, "rb") as model_file:
            model_parameters = pickle.load(model_file)
        self.pivot_pairs = model_parameters["pivot_pairs"]
        self.K = len(self.pivot_pairs[0][3])
        self.P = model_parameters["P"]