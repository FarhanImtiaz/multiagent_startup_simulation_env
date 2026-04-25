import importlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mass_startup_env.models import StartupAction
from mass_startup_env.server.startup_environment import StartupOpenEnv


def main() -> None:
    with open("openenv.yaml", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    for key in ("name", "entrypoint", "environment", "api"):
        if key not in manifest:
            raise AssertionError(f"openenv.yaml missing required key: {key}")

    module_name, object_name = manifest["entrypoint"].split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "openenv":
            raise
        print("OpenEnv package is not installed locally; skipping server app import.")
    else:
        if not hasattr(module, object_name):
            raise AssertionError(f"Entrypoint object not found: {manifest['entrypoint']}")

    env = StartupOpenEnv(max_days=3, seed=7)
    obs = env.reset()
    if obs.day != 1:
        raise AssertionError("reset() did not return day 1 observation")

    obs = env.step(StartupAction(action="do_nothing"))
    if env.state.step_count != 1:
        raise AssertionError("state.step_count did not update after step()")
    if obs.metadata.get("action") != "do_nothing":
        raise AssertionError("step() metadata did not preserve action")

    methods = set(manifest["api"]["methods"])
    if not {"reset", "step", "state"}.issubset(methods):
        raise AssertionError("openenv.yaml api.methods must include reset, step, and state")

    print("OpenEnv package validation passed.")


if __name__ == "__main__":
    main()
