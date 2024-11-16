# Time-series-AD-with-TCN
Temporal Convolutional Neural Network in the time series anomaly detection task -
model implementation, experimental framework, experiment results.

---

## What is TCN?

Temporal Convolutional Network (TCN) is a variant of Convolutional Neural Network (CNN)
with some enhancements introduced for processing time series (temporal) data. In comparison
to the classical CNN, there are following differences:

- **causal convolutions** - convolutions that take only past values to compute
the value for the current moment; in the temporal context, it means that there is no
leakage of information from the future
- **dilation in convolutions** - in TCN, we use _dilated_ convolutions (kernel skips some values),
which significantly widen the network's perception field in a point of time
- **skip connections** the structure of a TCN network is not purely linear, there are some
additional connections between layers that are not direct neighbors

In TCN, we usually use padding in each convolutional layer so that the size of the output
is the same as the size of the input.

## Dataset

For the purpose of testing the anomaly detection model, synthetic
_SinusRandomWalk_ (SRW) dataset has been used. It has been downloaded
from [this site](https://helios2.mi.parisdescartes.fr/~themisp/norma/).

Reference:

P. Boniol, M. Linardi, F. Roncallo, T. Palpanas, M. Meftah, E. Remy,
Unsupervised and Scalable Subsequence Anomaly Detection in Large Data Series,
VLDBJ (2021)
