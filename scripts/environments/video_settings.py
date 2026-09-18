from dataclasses import dataclass


@dataclass(frozen=True)
class VideoSettings:
    frames_per_second: int
    playback_speed: float

    @property
    def video_frames_per_second(self) -> int:
        return round(self.frames_per_second * self.playback_speed)
