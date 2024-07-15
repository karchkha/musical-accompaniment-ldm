import subprocess
import os
import pytest

def run_main_with_config(config_file):
    """ Helper function to run main.py with a given config file """
    # Set environment variable to disable wandb
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"  # This will prevent wandb from logging anything online
    
    """ Helper function to run main.py with a given config file """
    result = subprocess.run(
        ["python", "/home/karchkhadze/ctm_pl/train_audio.py", "--cfg", config_file],
        capture_output=True,
        text=True
    )
    return result
    
@pytest.mark.parametrize("config_file", [
    "/home/karchkhadze/ctm_pl/tests/train_audiodm_uncond.yaml",
    "/home/karchkhadze/ctm_pl/tests/train_audiodm_conditional.yaml",
    "/home/karchkhadze/ctm_pl/tests/train_audiodm_cond_separation.yaml"
    # Add more config files as needed
])


def test_scenarios(config_file):
    # Run the main script with the given config
    result = run_main_with_config(config_file)
    
    # Check if the script runs successfully
    assert result.returncode == 0, f"Config {config_file} failed: {result.stderr}"

    # Optionally print the output for debugging
    print(result.stdout)

# Additional helper functions or setup/teardown functions can be added here if needed
