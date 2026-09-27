from dataclasses import dataclass


@dataclass(frozen=True)
class Validation[SettingsT, EnvironmentT]:
    settings: SettingsT
    environment: EnvironmentT
