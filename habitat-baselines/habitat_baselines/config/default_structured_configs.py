import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import II

cs = ConfigStore.instance()


@dataclass
class HabitatBaselinesBaseConfig:
    pass


@dataclass
class WBConfig(HabitatBaselinesBaseConfig):
    """Weights and Biases config"""

    # The name of the project on W&B.
    project_name: str = ""
    # Logging entity (like your username or team name)
    entity: str = ""
    # The group ID to assign to the run. Optional to specify.
    group: str = ""
    # The run name to assign to the run. If not specified,
    # W&B will randomly assign a name.
    run_name: str = ""


@dataclass
class EvalConfig(HabitatBaselinesBaseConfig):
    # The split to evaluate on
    split: str = "val"
    should_load_ckpt: bool = True
    # The number of time to run each episode through evaluation.
    # Only works when evaluating on all episodes.
    evals_per_ep: int = 1
    video_option: List[str] = field(
        # available options are "disk" and "tensorboard"
        default_factory=lambda: []
    )


@dataclass
class PreemptionConfig(HabitatBaselinesBaseConfig):
    # Append the slurm job ID to the resume state filename if running
    # a slurm job. This is useful when you want to have things from a different
    # job but the same checkpoint dir not resume.
    append_slurm_job_id: bool = False
    # Number of gradient updates between saving the resume state
    save_resume_state_interval: int = 100
    # Save resume states only when running with slurm
    # This is nice if you don't want debug jobs to resume
    save_state_batch_only: bool = False


@dataclass
class ActionDistributionConfig(HabitatBaselinesBaseConfig):
    use_log_std: bool = True
    use_softplus: bool = False
    log_std_init: float = 0.0
    # If True, the std will be a parameter not conditioned on state
    use_std_param: bool = False
    # If True, the std will be clamped to the specified min and max std values
    clamp_std: bool = True
    min_std: float = 1e-6
    max_std: int = 1
    min_log_std: int = -5
    max_log_std: int = 2
    # For continuous action distributions (including gaussian):
    action_activation: str = "tanh"  # ['tanh', '']
    scheduled_std: bool = False


@dataclass
class ObsTransformConfig(HabitatBaselinesBaseConfig):
    pass


@dataclass
class CenterCropperConfig(ObsTransformConfig):
    type: str = "CenterCropper"
    height: int = 256
    width: int = 256
    channels_last: bool = True
    trans_keys: Tuple[str] = (
        "rgb",
        "depth",
        "semantic",
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms/center_cropper",
    name="center_cropper_base",
    node=CenterCropperConfig,
)


@dataclass
class ResizeShortestEdgeConfig(ObsTransformConfig):
    type: str = "ResizeShortestEdge"
    size: int = 256
    channels_last: bool = True
    trans_keys: Tuple[str] = (
        "rgb",
        "depth",
        "semantic",
    )
    semantic_key: str = "semantic"


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms/resize_shortest_edge",
    name="resize_shortest_edge_base",
    node=ResizeShortestEdgeConfig,
)


@dataclass
class Cube2EqConfig(ObsTransformConfig):
    type: str = "CubeMap2Equirect"
    height: int = 256
    width: int = 512
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms/cube_2_eq",
    name="cube_2_eq_base",
    node=Cube2EqConfig,
)


@dataclass
class Cube2FishConfig(ObsTransformConfig):
    type: str = "CubeMap2Fisheye"
    height: int = 256
    width: int = 256
    fov: int = 180
    params: Tuple[float] = (0.2, 0.2, 0.2)
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms/cube_2_fish",
    name="cube_2_fish_base",
    node=Cube2FishConfig,
)


@dataclass
class AddVirtualKeysConfig(ObsTransformConfig):
    # This is kept as reference to rememver this obs_transformer exists
    type: str = "AddVirtualKeys"


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms/add_virtual_keys",
    name="add_virtual_keys_base",
    node=Cube2FishConfig,
)


@dataclass
class Eq2CubeConfig(ObsTransformConfig):
    type: str = "Equirect2CubeMap"
    height: int = 256
    width: int = 256
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    group="habitat_baselines/rl/policy/obs_transforms/eq_2_cube",
    name="eq_2_cube_base",
    node=Eq2CubeConfig,
)


@dataclass
class PolicyConfig(HabitatBaselinesBaseConfig):
    name: str = "PointNavResNetPolicy"
    action_distribution_type: str = "categorical"  # or 'gaussian'
    # If the list is empty, all keys will be included.
    # For gaussian action distribution:
    action_dist: ActionDistributionConfig = ActionDistributionConfig()
    obs_transforms: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PPOConfig(HabitatBaselinesBaseConfig):
    """Proximal policy optimization config"""

    clip_param: float = 0.2
    ppo_epoch: int = 4
    num_mini_batch: int = 2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 2.5e-4
    eps: float = 1e-5
    max_grad_norm: float = 0.5
    num_steps: int = 5
    use_gae: bool = True
    use_linear_lr_decay: bool = False
    use_linear_clip_decay: bool = False
    gamma: float = 0.99
    tau: float = 0.95
    reward_window_size: int = 50
    use_normalized_advantage: bool = False
    hidden_size: int = 512
    entropy_target_factor: float = 0.0
    use_adaptive_entropy_pen: bool = False
    use_clipped_value_loss: bool = True
    # Use double buffered sampling, typically helps
    # when environment time is similar or larger than
    # policy inference time during rollout generation
    # Not that this does not change the memory requirements
    use_double_buffered_sampler: bool = False


@dataclass
class VERConfig(HabitatBaselinesBaseConfig):
    """Variable experience rollout config"""

    variable_experience: bool = True
    num_inference_workers: int = 2
    overlap_rollouts_and_learn: bool = False


@dataclass
class TmpAuxLossConfig(HabitatBaselinesBaseConfig):
    enabled: List[str] = field(default_factory=lambda: [])


@dataclass
class AuxLossConfig(HabitatBaselinesBaseConfig):
    pass


@dataclass
class CPCALossConfig(AuxLossConfig):
    """Action-conditional contrastive predictive coding loss"""

    k: int = 20
    time_subsample: int = 6
    future_subsample: int = 2
    loss_scale: float = 0.1


@dataclass
class DDPPOConfig(HabitatBaselinesBaseConfig):
    """Decentralized distributed proximal policy optimization config"""

    sync_frac: float = 0.6
    distrib_backend: str = "GLOO"
    rnn_type: str = "GRU"
    num_recurrent_layers: int = 1
    backbone: str = "resnet18"
    pretrained_weights: str = "data/ddppo-models/gibson-2plus-resnet50.pth"
    # Loads pretrained weights
    pretrained: bool = False
    # Loads just the visual encoder backbone weights
    pretrained_encoder: bool = False
    # Whether the visual encoder backbone will be trained
    train_encoder: bool = True
    # Whether to reset the critic linear layer
    reset_critic: bool = True
    # Forces distributed mode for testing
    force_distributed: bool = False


@dataclass
class RLConfig(HabitatBaselinesBaseConfig):
    """Reinforcement learning config"""

    preemption: PreemptionConfig = PreemptionConfig()
    policy: PolicyConfig = PolicyConfig()
    ppo: PPOConfig = PPOConfig()
    ver: VERConfig = VERConfig()

    # Auxiliary Losses
    # auxiliary_losses: Any = MISSING
    # TODO : Replace TmpAuxLossConfig with AuxLossConfig
    auxiliary_losses: TmpAuxLossConfig = TmpAuxLossConfig()
    ddppo: DDPPOConfig = DDPPOConfig()


@dataclass
class ORBSLAMConfig(HabitatBaselinesBaseConfig):
    """ORB-SLAM config"""

    slam_vocab_path: str = "habitat_baselines/slambased/data/ORBvoc.txt"
    slam_settings_path: str = (
        "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
    )
    map_cell_size: float = 0.1
    map_size: int = 40
    # camera_height = (
    #     get_task_config().habitat.simulator.depth_sensor.position[1]
    # )
    camera_height: float = II("habitat.simulator.depth_sensor.position[1]")
    beta: int = 100
    # h_obstacle_min = 0.3 * _C.orbslam2.camera_height
    h_obstacle_min: float = 0.3 * 1.25
    # h_obstacle_max = 1.0 * _C.orbslam2.camera_height
    h_obstacle_max = 1.0 * 1.25
    d_obstacle_min: float = 0.1
    d_obstacle_max: float = 4.0
    preprocess_map: bool = True
    # Note: hydra does not support basic operators in interpolations of numbers
    # see https://github.com/omry/omegaconf/issues/91 for more details
    # min_pts_in_obstacle = (
    #     get_task_config().habitat.simulator.depth_sensor.width / 2.0
    # )
    # Workaround for the operation above:
    # (640 is the default habitat depth sensor width)
    min_pts_in_obstacle: float = 640 / 2.0
    angle_th: float = math.radians(15)  # float(np.deg2rad(15))
    dist_reached_th: float = 0.15
    next_waypoint_th: float = 0.5
    num_actions: int = 3
    dist_to_stop: float = 0.05
    planner_max_steps: int = 500
    # depth_denorm = (
    #     get_task_config().habitat.simulator.depth_sensor.max_depth
    # )
    depth_denorm: float = II("habitat.simulator.depth_sensor.max_depth")


@dataclass
class ProfilingConfig(HabitatBaselinesBaseConfig):
    capture_start_step: int = -1
    num_steps_to_capture: int = -1


@dataclass
class HabitatBaselinesConfig(HabitatBaselinesBaseConfig):
    # task config can be a list of configs like "A.yaml,B.yaml"
    # base_task_config_path: str = (
    #     "habitat-lab/habitat/config/task/pointnav.yaml"
    # )
    cmd_trailing_opts: List[str] = field(default_factory=list)
    trainer_name: str = "ppo"
    torch_gpu_id: int = 0
    video_render_views: List[str] = field(default_factory=list)
    tensorboard_dir: str = "tb"
    writer_type: str = "tb"
    video_dir: str = "video_dir"
    video_fps: int = 10
    test_episode_count: int = -1
    # path to ckpt or path to ckpts dir
    eval_ckpt_path_dir: str = "data/checkpoints"
    num_environments: int = 16
    num_processes: int = -1  # deprecated
    checkpoint_folder: str = "data/checkpoints"
    num_updates: int = 10000
    num_checkpoints: int = 10
    # Number of model updates between checkpoints
    checkpoint_interval: int = -1
    total_num_steps: float = -1.0
    log_interval: int = 10
    log_file: str = "train.log"
    force_blind_policy: bool = False
    verbose: bool = True
    eval_keys_to_include_in_name: List[str] = field(default_factory=list)
    # For our use case, the CPU side things are mainly memory copies
    # and nothing of substantive compute. PyTorch has been making
    # more and more memory copies parallel, but that just ends up
    # slowing those down dramatically and reducing our perf.
    # This forces it to be single threaded.  The default
    # value is left as false as it's different from how
    # PyTorch normally behaves, but all configs we provide
    # set it to true and yours likely should too
    force_torch_single_threaded: bool = False
    # Weights and Biases config
    wb: WBConfig = WBConfig()
    # When resuming training or evaluating, will use the original
    # training config if load_resume_state_config is True
    load_resume_state_config: bool = True
    eval: EvalConfig = EvalConfig()
    rl: RLConfig = RLConfig()

    orbslam2: ORBSLAMConfig = ORBSLAMConfig()
    profiling: ProfilingConfig = ProfilingConfig()


cs.store(
    group="habitat_baselines",
    name="habitat_baselines_config_base",
    node=HabitatBaselinesConfig,
)
cs.store(
    group="habitat_baselines/rl/policy", name="policy_base", node=PolicyConfig
)

cs.store(
    group="habitat_baselines/rl/auxiliary_losses/cpca",
    name="cpca_loss_base",
    node=CPCALossConfig,
)


from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HabitatBaselinesConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://habitat_baselines/config/",
        )
