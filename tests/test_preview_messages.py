import queue
import unittest
from unittest.mock import Mock, patch

from scripts.preview import rollouts


class PreviewMessageTests(unittest.TestCase):
    def play(self, items):
        messages = queue.Queue()
        for item in items:
            messages.put(item)
        with (
            patch.object(rollouts, "pygame") as pygame,
            patch.object(
                rollouts,
                "_quit_has_been_requested",
                side_effect=[False] * len(items) + [True],
            ),
        ):
            rollouts._play_preview(worker=Mock(), messages=messages)
        return pygame

    def test_status_never_becomes_an_image_or_opens_window(self):
        pygame = self.play([("status", "Calculating next episode…")])
        pygame.image.frombytes.assert_not_called()
        pygame.display.set_mode.assert_not_called()
        pygame.display.set_caption.assert_called_once_with(
            "Preview — Calculating next episode…"
        )

    def test_frame_after_status_is_displayed(self):
        pygame = self.play([("status", "Playing"), ("frame", b"pixels")])
        pygame.image.frombytes.assert_called_once_with(b"pixels", (600, 600), "RGB")
        pygame.display.set_mode.assert_called_once()
        pygame.display.flip.assert_called_once()
        pygame.display.set_caption.assert_called_once_with("Preview — Playing")

    def test_freeze_keeps_polling_events_without_consuming_next_frame(self):
        messages = queue.Queue()
        messages.put(("freeze", None))
        messages.put(("frame", b"next episode"))
        with (
            patch.object(rollouts, "pygame") as pygame,
            patch.object(rollouts.time, "monotonic", side_effect=[10, 10, 11]),
            patch.object(
                rollouts, "_quit_has_been_requested", side_effect=[False, False, True]
            ),
        ):
            rollouts._play_preview(worker=Mock(), messages=messages)
        pygame.image.frombytes.assert_not_called()
        self.assertEqual(messages.get_nowait(), ("frame", b"next episode"))

    def test_unknown_message_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown preview message"):
            self.play([("unexpected", "text")])

    def test_worker_error_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            self.play([("error", "simulation failed")])

    def test_cleanup_on_playback_error(self):
        with (
            patch.object(rollouts, "multiprocessing") as multiprocessing,
            patch.object(rollouts, "pygame") as pygame,
            patch.object(rollouts, "_play_preview", side_effect=RuntimeError("failed")),
        ):
            context = multiprocessing.get_context.return_value
            worker = context.Process.return_value
            worker.is_alive.side_effect = [True, False]
            with self.assertRaisesRegex(RuntimeError, "failed"):
                rollouts._run(config=Mock())
            worker.terminate.assert_called_once()
            worker.join.assert_called_once()
            worker.close.assert_called_once()
            context.Queue.return_value.close.assert_called_once()
            context.Queue.return_value.join_thread.assert_called_once()
            pygame.quit.assert_called_once()
