import numpy as np
import numpy.typing as npt


def summarize_distribution(
    *, values: npt.NDArray[np.floating] | npt.NDArray[np.integer]
) -> dict[str, float]:
    if int(values.size - np.isfinite(values).sum()) > 0:
        raise ValueError("Summarization failed. Some values are not finite")
    return {
        label: float(np.percentile(a=values, q=percentile))
        for label, percentile in {
            "1_p10": 10.0,
            "2_median": 50.0,
            "3_p90": 90.0,
        }.items()
    }
