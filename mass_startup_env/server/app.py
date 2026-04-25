from openenv.core.env_server.http_server import create_app

from mass_startup_env.models import StartupAction, StartupObservation
from mass_startup_env.server.startup_environment import StartupOpenEnv


app = create_app(
    StartupOpenEnv,
    StartupAction,
    StartupObservation,
    env_name="mass-startup-env",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

