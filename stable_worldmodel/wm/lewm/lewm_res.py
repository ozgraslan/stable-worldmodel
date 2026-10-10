from .lewm import LeWM


class LeWMRes(LeWM):
    """LeWM predicting residual changes in the embedding space."""

    def predict(self, info):
        """Add the projected prediction to the current embedding."""
        emb = info['emb']
        info["preds"] = emb + super().predict(info)["preds"]
        return info


__all__ = ['LeWMRes']
