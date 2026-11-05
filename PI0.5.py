ACTION_DIM = 14                                   # ←01
ACTION_HORIZON = 32
STATE_FIELDS = ("follow1_pos", "follow2_pos")
CAMERA_FIELDS = {
    "camera_front": "observation/image",
    "camera_left": "observation/left_wrist_image",
    "camera_right": "observation/right_wrist_image",
}

def build_joint_observation(obs: dict[str, Any], *, instruction: str) -> dict[str, Any]:
    """Map the PolicySpace bimanual joint envelope to OpenPI X2RealInputs."""
    # ...原文此处先取 images = obs.get("images")、state = obs.get("state") 并做 dict 类型校验，略...
    result: dict[str, Any] = {}
    for camera, model_key in CAMERA_FIELDS.items():
        if camera not in images:
            raise ValueError(f"missing camera {camera!r}")
        result[model_key] = _rgb_image(images[camera], name=f"images[{camera!r}]")   # ←02

    result["observation/state"] = np.concatenate(
        [_finite_vector(state[field], name=f"state[{field!r}]", size=7) for field in STATE_FIELDS]
    )                                             # ←03
    current_instruction = obs.get("instruction")
    result["prompt"] = (
        current_instruction.strip()
        if isinstance(current_instruction, str) and current_instruction.strip()
        else instruction.strip()
    )                                             # ←04
    if not result["prompt"]:
        raise ValueError("joint Pi0.5 inference requires a non-empty instruction")
    return result

class Model(PolicyModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        ...
        self._policy = _create_openpi_policy(self.train_config, self.checkpoint_path, self.device, openpi_root)
                                                # ←05（权重加载：openpi 的 create_trained_policy）
    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = obs                    # ←06（只缓冲，不算）

    def infer_actions(self) -> dict[str, Any]:
        if self._latest_obs is None:
            raise RuntimeError("No observation buffered. Call ingest_observation first.")
        started = time.monotonic()
        result = self._policy.infer(build_joint_observation(self._latest_obs, instruction=self.instruction))
                                                # ←07（"推理"发生在这里）
        actions = validate_joint_actions(result)[: self.action_horizon]   # ←08
        return {
            "actions": actions,
            "openpi_timing": {"infer_ms": round((time.monotonic() - started) * 1000.0, 3)},
        }
        
"""
    deploy.yaml（原生π0.5上）:    
        
    metadata:
  supported_schema_pairs:
    - observation_schema: bimanual_rgb_joint@1
      action_schema: bimanual_joint_position@1
      embodiment_constraints: [ex001_6r@1]
  action_mode: joint_position
  action_type: absolute_joint
  action_dim: 14
  action_horizon: 32
  gripper_convention: normalized_01
  recommended_ingest_observation_each_step: false   # ←09
  """
  
  
"""
    deploy.yaml(在WALL-OS-0.5适配器上)
    
    model:
  cfg:
    robot_type: "desktop"
    norm_key: "ex_normal"                          # ←10
    action_dim: 26                                 # ←11
    action_horizon: 32
    num_inference_timesteps: 10                    # ←12
    control_rate_hz: 20
    # Native WallX actions are sampled at 20 Hz: 32 rows span 1.60 s. The
    # client interpolates them into 48 simulator commands at 30 Hz, preserving
    # the same 1.60 s span. This value describes the native RTC model rate; it
    # must not override the simulator's post-interpolation execution rate.
    image_wire_encoding: "lossless_png"            # ←13
metadata:
  supported_schema_pairs:
    - observation_schema: bimanual_rgb_ee@1
      action_schema: bimanual_ee_absolute@1
  action_dim: 14                                   # ←14
  action_horizon: 33
  # ...（gripper_convention 注释原文警告：未换算的 [0,1] 指令是"合法弧度值"，
  #      会被动作项静默钳位，夹爪只能开到约 18% 行程）
  gripper_convention: normalized_01
    
"""