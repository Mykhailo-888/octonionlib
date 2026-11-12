"""Experimental NumPy network:
16 real inputs -> octonion tanh -> 2 real outputs.
"""

import numpy as np

if __package__:
    from .octonion import Octonion
else:
    from octonion import Octonion


def matrix(values, width, name):
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != width or len(array) == 0:
        raise ValueError(
            f"{name} must have shape (n, {width}), n > 0"
        )

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers")

    return array


class OctonionNetwork:
    def __init__(self, hidden_octonions=8, seed=42):
        if (
            isinstance(hidden_octonions, bool)
            or not isinstance(hidden_octonions, int)
            or hidden_octonions < 1
        ):
            raise ValueError(
                "hidden_octonions must be a positive integer"
            )

        self.hidden_octonions = hidden_octonions
        self.seed = seed

        self.mean = np.zeros(16)
        self.std = np.ones(16)

        self.history = []
        self.best_epoch = 0

        rng = np.random.default_rng(seed)
        h = hidden_octonions

        self.parameters = {
            "W": rng.normal(
                0, 1 / np.sqrt(16), (h, 2, 8)
            ),
            "b": np.zeros((h, 8)),
            "V": rng.normal(
                0, 1 / np.sqrt(8 * h), (8 * h, 2)
            ),
            "c": np.zeros(2),
        }

        basis = [Octonion.real(1)] + [
            Octonion.basis(i) for i in range(1, 8)
        ]

        # tensor[a, b, c]:
        # component c of the octonion product e_a * e_b.
        self.tensor = np.array([
            [(a * b).a for b in basis]
            for a in basis
        ])

    @property
    def parameter_count(self):
        return sum(
            p.size for p in self.parameters.values()
        )

    def _features(self, X):
        # Normalize using training statistics.
        # Group 16 real inputs into two octonions.
        blocks = (
            (X - self.mean) / self.std
        ).reshape(-1, 2, 8)

        # Component c of e_a * input_octonion_i.
        return np.einsum(
            "nib,abc->niac",
            blocks,
            self.tensor,
            optimize=True,
        )

    def _forward(self, features):
        p = self.parameters

        # Octonion left multiplication:
        # weight * input, then add bias.
        z = np.einsum(
            "hia,niac->nhc",
            p["W"],
            features,
            optimize=True,
        ) + p["b"]

        # Hidden activation: componentwise tanh.
        hidden = np.tanh(z).reshape(
            len(features), -1
        )

        # Two linear real outputs.
        output = hidden @ p["V"] + p["c"]

        return hidden, output

    def _loss_grad(self, features, y):
        hidden, output = self._forward(features)

        error = output - y
        loss = float(np.mean(error ** 2))

        # MSE averaged over n samples and TWO outputs:
        # dL/doutput = error / n.
        delta = error / len(y)

        # tanh derivative: 1 - tanh(z)^2.
        dz = (
            delta @ self.parameters["V"].T
        ) * (1 - hidden ** 2)

        dz = dz.reshape(
            -1, self.hidden_octonions, 8
        )

        gradients = {
            "W": np.einsum(
                "nhc,niac->hia",
                dz,
                features,
                optimize=True,
            ),
            "b": dz.sum(axis=0),
            "V": hidden.T @ delta,
            "c": delta.sum(axis=0),
        }

        return loss, gradients

    def predict(self, X):
        X = matrix(X, 16, "X")
        return self._forward(
            self._features(X)
        )[1]

    def fit(
        self,
        X,
        y,
        validation_data=None,
        epochs=600,
        learning_rate=0.003,
        batch_size=128,
        patience=80,
    ):
        X = matrix(X, 16, "X")
        y = matrix(y, 2, "y")

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number of rows"
            )

        for name, value in (
            ("epochs", epochs),
            ("batch_size", batch_size),
            ("patience", patience),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )

        if (
            not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError(
                "learning_rate must be positive and finite"
            )

        validation = None

        if validation_data is not None:
            vx, vy = validation_data

            vx = matrix(vx, 16, "validation X")
            vy = matrix(vy, 2, "validation y")

            if len(vx) != len(vy):
                raise ValueError(
                    "Validation row counts must match"
                )

            validation = (vx, vy)

        # Fit normalization on TRAINING rows only.
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0

        features = self._features(X)

        if validation is not None:
            validation = (
                self._features(validation[0]),
                validation[1],
            )

        def losses():
            train_loss = float(np.mean(
                (self._forward(features)[1] - y) ** 2
            ))

            val_loss = None

            if validation is not None:
                val_loss = float(np.mean(
                    (
                        self._forward(validation[0])[1]
                        - validation[1]
                    ) ** 2
                ))

            return train_loss, val_loss

        train_loss, val_loss = losses()

        best_loss = (
            val_loss
            if val_loss is not None
            else train_loss
        )

        best = {
            k: v.copy()
            for k, v in self.parameters.items()
        }

        self.best_epoch = 0

        self.history = [{
            "epoch": 0,
            "train_mse": train_loss,
            "val_mse": val_loss,
        }]

        first = {
            k: np.zeros_like(v)
            for k, v in self.parameters.items()
        }

        second = {
            k: np.zeros_like(v)
            for k, v in self.parameters.items()
        }

        rng = np.random.default_rng(self.seed + 1)

        step = 0
        stale = 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))

            for start in range(
                0, len(X), batch_size
            ):
                indices = order[
                    start:start + batch_size
                ]

                _, gradients = self._loss_grad(
                    features[indices],
                    y[indices],
                )

                step += 1

                # Adam updates real components
                # of all trainable parameters.
                for name, g in gradients.items():
                    first[name] = (
                        0.9 * first[name] + 0.1 * g
                    )

                    second[name] = (
                        0.999 * second[name]
                        + 0.001 * g ** 2
                    )

                    m = first[name] / (
                        1 - 0.9 ** step
                    )

                    v = second[name] / (
                        1 - 0.999 ** step
                    )

                    self.parameters[name] -= (
                        learning_rate * m
                        / (np.sqrt(v) + 1e-8)
                    )

            train_loss, val_loss = losses()

            monitor = (
                val_loss
                if val_loss is not None
                else train_loss
            )

            if (
                not np.isfinite(train_loss)
                or not np.isfinite(monitor)
            ):
                raise FloatingPointError(
                    "Training diverged: check data "
                    "and learning rate"
                )

            self.history.append({
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
            })

            if monitor < best_loss - 1e-9:
                best_loss = monitor
                stale = 0
                self.best_epoch = epoch

                best = {
                    k: v.copy()
                    for k, v in self.parameters.items()
                }
            else:
                stale += 1

                if stale >= patience:
                    break

        # Restore the best validation checkpoint.
        self.parameters = best

        return self.history

    def save(self, path):
        np.savez_compressed(
            path,
            hidden=np.array(self.hidden_octonions),
            seed=np.array(self.seed),
            mean=self.mean,
            std=self.std,
            **self.parameters,
        )

    @classmethod
    def load(cls, path):
        with np.load(
            path, allow_pickle=False
        ) as data:
            model = cls(
                int(data["hidden"]),
                seed=int(data["seed"]),
            )

            for name, expected in (
                model.parameters.items()
            ):
                value = data[name]

                if (
                    value.shape != expected.shape
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        f"Invalid saved parameter: {name}"
                    )

                model.parameters[name] = value.copy()

            mean = data["mean"]
            std = data["std"]

            if (
                mean.shape != (16,)
                or std.shape != (16,)
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or (std <= 0).any()
            ):
                raise ValueError(
                    "Invalid saved normalization"
                )

            model.mean = mean.copy()
            model.std = std.copy()

        return model


def main():
    # Synthetic arithmetic task for checking learning:
    # output 1 = sum of the first 8 inputs;
    # output 2 = sum of the last 8 inputs.
    rng = np.random.default_rng(123)

    X = rng.uniform(-1, 1, (2000, 16))

    y = np.column_stack((
        X[:, :8].sum(axis=1),
        X[:, 8:].sum(axis=1),
    ))

    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    train_X, train_y = X[:1400], y[:1400]
    val_X, val_y = X[1400:1700], y[1400:1700]
    test_X, test_y = X[1700:], y[1700:]

    model = OctonionNetwork(
        hidden_octonions=8,
        seed=42,
    )

    before = np.mean(
        (model.predict(test_X) - test_y) ** 2
    )

    history = model.fit(
        train_X,
        train_y,
        validation_data=(val_X, val_y),
    )

    prediction = model.predict(test_X)

    mse = np.mean(
        (prediction - test_y) ** 2,
        axis=0,
    )

    baseline = np.mean(
        (train_y.mean(axis=0) - test_y) ** 2
    )

    print(
        "Architecture: 16 real inputs -> "
        "8 octonions / tanh -> 2 linear outputs"
    )

    print(
        "Trainable real parameters:",
        model.parameter_count,
    )

    print(
        "Training / validation / test rows:",
        len(train_X),
        len(val_X),
        len(test_X),
    )

    print(
        "Epochs run:", len(history) - 1,
        "| Best epoch:", model.best_epoch,
    )

    print(
        "Test MSE before training:",
        float(before),
    )

    print(
        "Test MSE after training:",
        float(mse.mean()),
    )

    print("Test MSE per output:", mse)

    print(
        "Training-mean baseline test MSE:",
        float(baseline),
    )

    for i in range(5):
        print(
            "Target:", test_y[i],
            "| Prediction:", prediction[i],
        )

    model.save("octonion_network_16x2.npz")

    restored = OctonionNetwork.load(
        "octonion_network_16x2.npz"
    )

    np.testing.assert_array_equal(
        prediction,
        restored.predict(test_X),
    )

    print(
        "Weights and normalization saved:",
        "octonion_network_16x2.npz",
    )


if __name__ == "__main__":
    main()
