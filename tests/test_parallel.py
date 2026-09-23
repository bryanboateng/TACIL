import multiprocessing
import unittest
from unittest.mock import Mock, patch

from scripts.evaluation.core.parallel import _receive


class ParallelPoolTests(unittest.TestCase):
    def test_status_is_not_returned_as_an_observation(self):
        parent, child = multiprocessing.Pipe()
        try:
            child.send(("status", "creating environment"))
            child.send(("status", "resetting environment"))
            child.send((True, [1, 2, 3]))
            self.assertEqual(_receive(parent), [1, 2, 3])
        finally:
            parent.close()
            child.close()

    def test_timeout_includes_worker_and_last_stage_without_extending_deadline(self):
        connection = Mock()
        connection.poll.return_value = True
        connection.recv.return_value = (
            "status",
            "creating environment condition=2, seed=10050",
        )
        worker = Mock(pid=123, exitcode=None)
        with patch(
            "scripts.evaluation.core.parallel.monotonic", side_effect=[0, 0, 121]
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "pid=123 timed out after 120s while creating environment condition=2, seed=10050",
            ):
                _receive(connection, worker=worker)
        connection.poll.assert_called_once()

    def test_worker_error_includes_operation(self):
        parent, child = multiprocessing.Pipe()
        try:
            child.send(("status", "resetting environment condition=0, seed=10000"))
            child.send((False, "simulator failed"))
            with self.assertRaisesRegex(
                RuntimeError, "resetting environment condition=0, seed=10000"
            ):
                _receive(parent)
        finally:
            parent.close()
            child.close()


if __name__ == "__main__":
    unittest.main()
