from gymnasium.envs.registration import register
# Automatically register the environment when the package is imported
# Register an ID with a callable entry point (class path string works too)

register(
    id="Ship-v0",
    entry_point="shipenv.ship_env:ShipEnv",  # package.module:Class
)
