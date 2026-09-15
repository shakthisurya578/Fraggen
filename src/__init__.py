"""FragGen: file-fragment type classification for digital forensics.

Package layout
  src.models          ByteCNN, NT-Xent loss, feature baseline, checkpoint loading
  src.preprocessing   FragmentSet container, corpus loaders, augmentation, entropy features
  src.training        training loop  (python -m src.training.train)
                      evaluation CLI (python -m src.training.eval)
  src.inference       FragGenClassifier: bytes/file -> per-fragment predictions -> verdict
  src.explainability  JSON/CSV reports, verdict explanations, PDF stream mapping
  src.utils           config loading, metrics, confidence calibration

Two results live side by side in this project and must never be conflated:
the cross-corpus study (train FFT-75, test GovDocs1) and the Dataset V4
held-out-file benchmark (unseen documents from the training source pool).
"""
__version__ = "0.1.0"
