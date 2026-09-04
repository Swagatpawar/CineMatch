import numpy as np


class Prediction:
    def __init__(self, uid: int, iid: int, est: float):
        self.uid = int(uid)
        self.iid = int(iid)
        self.est = float(est)


class CineMatchSVD:
    def __init__(self, user_ids, movie_ids, user_factors, movie_factors, user_bias, movie_bias, global_mean):
        self.user_ids = list(user_ids)
        self.movie_ids = list(movie_ids)
        self.user_factors = np.asarray(user_factors)
        self.movie_factors = np.asarray(movie_factors)
        self.user_bias = np.asarray(user_bias)
        self.movie_bias = np.asarray(movie_bias)
        self.global_mean = float(global_mean)
        self.n_factors = self.user_factors.shape[1]

    def predict(self, uid, iid):
        uid = int(uid)
        iid = int(iid)
        if uid not in self.user_ids:
            return Prediction(uid, iid, float(np.clip(self.global_mean, 0.5, 5.0)))
        if iid not in self.movie_ids:
            u_idx = self.user_ids.index(uid)
            return Prediction(uid, iid, float(np.clip(self.global_mean + self.user_bias[u_idx], 0.5, 5.0)))

        u_idx = self.user_ids.index(uid)
        i_idx = self.movie_ids.index(iid)
        score = (
            self.global_mean
            + self.user_bias[u_idx]
            + self.movie_bias[i_idx]
            + float(np.dot(self.user_factors[u_idx], self.movie_factors[i_idx]))
        )
        return Prediction(uid, iid, float(np.clip(score, 0.5, 5.0)))

    def predict_many(self, uid, item_ids):
        """Predict a batch of known items using the same formula as predict."""
        uid = int(uid)
        if uid not in self.user_ids:
            score = np.full(len(item_ids), self.global_mean, dtype=float)
        else:
            user_index = self.user_ids.index(uid)
            item_indices = [self.movie_ids.index(int(item_id)) for item_id in item_ids]
            score = (
                self.global_mean
                + self.user_bias[user_index]
                + self.movie_bias[item_indices]
                + self.movie_factors[item_indices].dot(self.user_factors[user_index])
            )
        return [Prediction(uid, int(item_id), float(np.clip(value, 0.5, 5.0))) for item_id, value in zip(item_ids, score)]