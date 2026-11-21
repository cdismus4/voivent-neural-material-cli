"""Exception types for the vnm1 CLI.

Exit-code mapping lives in :mod:`vnm1.cli` (design doc section 3.3):
``InputValidationError`` is reported as exit code 2 and means training
was never started; ``TrainingError`` is exit code 4 (the trial could
not be completed); ``ExportVerificationError`` is exit code 5 (the
exported artifact failed to write, read back, or verify).
"""


class VnmError(Exception):
    """Base class for all vnm1 errors."""


class InputValidationError(VnmError):
    """The input texture set or its specification is invalid.

    Raised by the input layer (design doc section 4) before any training
    work happens: missing required maps, dimension mismatches, a missing
    or invalid ``normal_convention``, unsupported image formats, invalid
    material names, and similar problems.
    """


class TrainingError(VnmError):
    """Training could not be completed (exit code 4).

    Divergence (non-finite loss), device errors, a missing torch
    installation, and similar abnormalities that abort the trial before
    an artifact exists.
    """


class ExportVerificationError(VnmError):
    """Export, read-back or format verification failed (exit code 5).

    Raised after training, when the written artifact cannot be parsed
    back, its hashes do not match the manifest, its reserved bytes
    violate the format rule, or the read-back evaluation cannot run.
    """
