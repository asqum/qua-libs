from contextlib import contextmanager

from qm.qua import for_


@contextmanager
def nested_binary_loops(loop_vars, idx=0):
    """Recursively create nested QUA loops over binary variables."""
    if idx == len(loop_vars):
        yield
        return

    with for_(loop_vars[idx], 0, loop_vars[idx] < 2, loop_vars[idx] + 1):
        with nested_binary_loops(loop_vars, idx + 1):
            yield


def state_to_label(state_int, n_qubits):
    return format(state_int, f"0{n_qubits}b")
