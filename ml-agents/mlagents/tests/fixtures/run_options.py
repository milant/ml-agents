import pytest

from typing import Dict

from mlagents.trainers.settings import RunOptions, EnvironmentParameterSettings


@pytest.fixture
def raw_run_options_yaml() -> str:
    test_yaml = """
      behaviors:
          3DBall:
              trainer_type: sac
              hyperparameters:
                  learning_rate: 0.0004
                  learning_rate_schedule: constant
                  batch_size: 64
                  buffer_size: 200000
                  buffer_init_steps: 100
                  tau: 0.006
                  steps_per_update: 10.0
                  save_replay_buffer: true
                  init_entcoef: 0.5
                  reward_signal_steps_per_update: 10.0
              network_settings:
                  normalize: false
                  hidden_units: 256
                  num_layers: 3
                  vis_encode_type: nature_cnn
                  memory:
                      memory_size: 1288
                      sequence_length: 12
              reward_signals:
                  extrinsic:
                      gamma: 0.999
                      strength: 1.0
                  curiosity:
                      gamma: 0.999
                      strength: 1.0
              keep_checkpoints: 5
              max_steps: 500000
              time_horizon: 1000
              summary_freq: 12000
              checkpoint_interval: 1
              threaded: true
      env_settings:
          env_path: test_env_path
          env_args:
              - test_env_args1
              - test_env_args2
          base_port: 12345
          num_envs: 8
          seed: 12345
      engine_settings:
          width: 12345
          height: 12345
          quality_level: 12345
          time_scale: 12345
          target_frame_rate: 12345
          capture_frame_rate: 12345
          no_graphics: true
      checkpoint_settings:
          run_id: test_run_id
          initialize_from: test_directory
          load_model: false
          resume: true
          force: true
          train_model: false
          inference: false
      debug: true
      environment_parameters:
          big_wall_height:
              curriculum:
                - name: Lesson0
                  completion_criteria:
                      measure: progress
                      behavior: BigWallJump
                      signal_smoothing: true
                      min_lesson_length: 100
                      threshold: 0.1
                  value:
                      sampler_type: uniform
                      sampler_parameters:
                          min_value: 0.0
                          max_value: 4.0
                - name: Lesson1
                  completion_criteria:
                      measure: reward
                      behavior: BigWallJump
                      signal_smoothing: true
                      min_lesson_length: 100
                      threshold: 0.2
                  value:
                      sampler_type: gaussian
                      sampler_parameters:
                          mean: 4.0
                          st_dev: 7.0
                - name: Lesson2
                  completion_criteria:
                      measure: progress
                      behavior: BigWallJump
                      signal_smoothing: true
                      min_lesson_length: 20
                      threshold: 0.3
                  value:
                      sampler_type: multirangeuniform
                      sampler_parameters:
                          intervals: [[1.0, 2.0],[4.0, 5.0]]
                - name: Lesson3
                  value: 8.0
          small_wall_height: 42.0
          other_wall_height:
              sampler_type: multirangeuniform
              sampler_parameters:
                  intervals: [[1.0, 2.0],[4.0, 5.0]]
      """
    return test_yaml


@pytest.fixture
def env_param_settings() -> Dict[str, EnvironmentParameterSettings]:
    env_params_dict = {
        "mass": {
            "sampler_type": "uniform",
            "sampler_parameters": {"min_value": 1.0, "max_value": 2.0},
        },
        "scale": {
            "sampler_type": "gaussian",
            "sampler_parameters": {"mean": 1.0, "st_dev": 2.0},
        },
        "length": {
            "sampler_type": "multirangeuniform",
            "sampler_parameters": {"intervals": [[1.0, 2.0], [3.0, 4.0]]},
        },
        "gravity": 1,
        "wall_height": {
            "curriculum": [
                {
                    "name": "Lesson1",
                    "completion_criteria": {
                        "measure": "reward",
                        "behavior": "fake_behavior",
                        "threshold": 10,
                    },
                    "value": 1,
                },
                {"value": 4, "name": "Lesson2"},
            ]
        },
    }
    return EnvironmentParameterSettings.structure(
        env_params_dict, Dict[str, EnvironmentParameterSettings]
    )


@pytest.fixture(scope="module")
def simple_run_options() -> RunOptions:
    default_settings = {"max_steps": 1, "network_settings": {"num_layers": 1000}}
    behaviors = {"test1": {"max_steps": 2, "network_settings": {"hidden_units": 2000}}}
    run_options_dict = {"default_settings": default_settings, "behaviors": behaviors}
    return RunOptions.from_dict(run_options_dict)
