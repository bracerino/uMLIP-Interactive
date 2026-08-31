"""Keep TensorFlow's start-up chatter out of the app's console.

GRACE (tensorpotential) pulls TensorFlow in at import time, and TF announces
itself with ~18 lines before any of our own output appears. None of it is
actionable here: the "already registered" messages are the normal result of
loading TF next to a CUDA PyTorch build, and the repeated type-promotion warning
is tensorpotential calling ``experimental_enable_numpy_behavior()`` once per
submodule.

The noise arrives through two channels that need different treatment:

* Python-level — TF's own logger, and the ``[tensorpotential] Info:`` notice.
  ``configure_tf_logging()`` handles these, but only if it runs **before**
  anything imports tensorflow, so it belongs at the very top of the entry point.
* C++/absl-level — the ``E0000``/``W0000`` lines from XLA's plugin registry.
  No environment variable covers those, so ``quiet_stderr()`` takes file
  descriptor 2 away for the duration of the import instead.

Between them the start-up banner drops from 18 lines to none.
"""

import contextlib
import logging
import os
import sys
import tempfile


def configure_tf_logging():
    """Silence TensorFlow's Python-level start-up messages.

    Must be called before ``tensorflow`` is imported anywhere in the process —
    TF reads these variables once, at import. Every value is set with
    ``setdefault`` so an explicitly exported variable still wins, which is what
    you want when debugging TF itself (``TF_CPP_MIN_LOG_LEVEL=0 streamlit run
    app.py`` puts the messages back).
    """
    # 3 = show nothing below FATAL from TF's C++ layer. Drops the oneDNN notice,
    # the cuFFT plugin line and the cpu_feature_guard AVX message.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # tensorpotential sets this itself on import and prints a line saying so.
    # Setting it here first means the same configuration without the notice.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    # tensorpotential enables numpy behaviour in each of its submodules; every
    # call after the first logs "enabling the new type promotion must happen at
    # the beginning of the program" through this logger.
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


@contextlib.contextmanager
def quiet_stderr():
    """Swallow writes to file descriptor 2 inside the block.

    Aimed at the absl/XLA registration messages, which are written from C++ and
    so ignore anything done to :mod:`logging` or :data:`sys.stderr`.

    Nothing is thrown away silently: the output is buffered, and if the block
    raises, the buffer is replayed to the real stderr before the exception
    propagates — so a genuinely broken import still shows why it broke. If fd 2
    cannot be duplicated at all (a harness that replaced it with something not
    backed by a descriptor), the block simply runs unmuffled.
    """
    try:
        saved_fd = os.dup(2)
    except (AttributeError, OSError):
        yield
        return

    buffer = tempfile.TemporaryFile(mode="w+b")
    try:
        sys.stderr.flush()
        os.dup2(buffer.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
    except BaseException:
        buffer.seek(0)
        captured = buffer.read().decode("utf-8", "replace")
        if captured:
            sys.stderr.write(captured)
            sys.stderr.flush()
        raise
    finally:
        os.close(saved_fd)
        buffer.close()
