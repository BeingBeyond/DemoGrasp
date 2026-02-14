import os
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import gym
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import tasks

def build_runner(cfg, env):
    train_param = cfg.train.params
    is_testing = cfg.test  # train_param["test"]
    ckpt_path = cfg.checkpoint

    if not is_testing:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.task_name}_{time_str}"
        if "run_name" in cfg:
            run_name = cfg.run_name
        log_dir = os.path.join(train_param.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4)
    else:
        log_dir = None

    if train_param.name == "ppo_onestep":
        # DemoGrasp: train one-step planner
        assert env.randomize_tracking_reference
        act_dim = 6
        if env.randomize_grasp_pose:
            act_dim += env.num_active_hand_dofs

        from algo import ppo_onestep
        runner = ppo_onestep.PPO(
            vec_env=env,
            actor_critic_class=ppo_onestep.ActorCritic,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False,
            action_dim=act_dim,
        )
    else:
        raise ValueError("Unrecognized algorithm!")

    if is_testing and ckpt_path != "":
        print(f"Loading model from {ckpt_path}")
        runner.test(ckpt_path)
    elif ckpt_path != "":
        print(f"\nWarning: load pre-trained policy. Loading model from {ckpt_path}\n")
        runner.load(ckpt_path)

    return runner

def check_joint(cfg, env):
    per_joint_duration = 10 * 1
    for t in range(100000):
        act = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float, device=env.device)
        act[:, :env.num_arm_dofs] = env.active_robot_dof_default_pos[:env.num_arm_dofs]
        act[:, env.hand_dof_start_idx:] = env.active_robot_dof_default_pos[env.num_arm_dofs:]
        i_joint = int(t / per_joint_duration) % env.num_actions
        print(i_joint)
        t_ = t % per_joint_duration
        if t_ < per_joint_duration//2:
            act[:,i_joint] = env.robot_dof_lower_limits[env.active_robot_dof_indices[i_joint]]
        else:
            act[:,i_joint] = env.robot_dof_upper_limits[env.active_robot_dof_indices[i_joint]]
        act[:, :env.num_arm_dofs] = unscale(
            act[:, :env.num_arm_dofs],
            env.robot_dof_lower_limits[env.arm_dof_indices],
            env.robot_dof_upper_limits[env.arm_dof_indices],
        )
        act[:, env.hand_dof_start_idx:] = unscale(
            act[:, env.hand_dof_start_idx:],
            env.robot_dof_lower_limits[env.active_hand_dof_indices],
            env.robot_dof_upper_limits[env.active_hand_dof_indices],
        )
        env.step(act)

def test_demo_replay(cfg, env):
    for t in range(100000):
        action = env.compute_reference_actions()
        obs, reward, reset, extras = env.step(action)
        if (t+1)%env.max_episode_length==0:
            print("success rate:", env.current_successes.mean())
        # reset when done
        env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            env.reset_idx(env_ids)

def collect_real_dataset(cfg, env):
    NUM_EPISODES = cfg.get("num_episodes", 10)
    PLAY_POLICY = True # Use DemoGrasp policy or direct demo replay?

    if PLAY_POLICY:
        cfg.test = True
        runner = build_runner(cfg, env)
        policy = runner.actor_critic
        policy.eval()

    from data.dataset_utils import LerobotDatasetWriter
    dataset_writer = LerobotDatasetWriter(
        output_path=f"{env.hand_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        camera_ids=env.camera_ids,
        data_type=env.render_data_type,
        image_shape=(*env.render_cfg["resize"], 3),
        fps=round(1/env.dt/env.decimation),
    )

    n_saved_episodes = 0
    while True:
        if n_saved_episodes >= NUM_EPISODES:
            break
        env.reset_idx(torch.arange(env.num_envs))
        obs = env.obs_dict["obs"].clone()
        if PLAY_POLICY:
            with torch.no_grad():
                plan = policy(obs, inference=True)
                env.generate_reaching_plan_idx(torch.arange(env.num_envs), actions=plan)

        episode_data_buffer = []
        for t in range(env.max_episode_length):
            real_obs = env.compute_real_observation_dict()
            action = env.compute_reference_actions()
            _, _, reset, extras = env.step(action)
            real_obs['action'] = action.cpu().numpy()
            episode_data_buffer.append(real_obs)
        
        assert reset.all()
        success = extras["current_successes"] > 0.5
        for env_id in range(env.num_envs):
            if n_saved_episodes >= NUM_EPISODES:
                break
            if success[env_id]:
                for t in range(len(episode_data_buffer)):
                    episode_end = (t == len(episode_data_buffer)-1)
                    dataset_writer.append_step(
                        {k: v[env_id:env_id+1] for k, v in episode_data_buffer[t].items()},
                        episode_end=episode_end
                    )
                n_saved_episodes += 1
                print(f"Saved episode {n_saved_episodes} from env {env_id}")


@hydra.main(version_base="1.3", config_path="./tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env = create_isaacgym_env()
    env.reset_idx(torch.arange(env.num_envs))
    
    if "debug" in cfg: 
        if cfg["debug"] == "check_joint":
            check_joint(cfg, env)

        elif cfg["debug"] == "test_demo_replay":
            test_demo_replay(cfg, env)
        
        elif cfg["debug"] == "collect_real_dataset":
            collect_real_dataset(cfg, env)
        
        else:
            for t in range(100000):
                action = env.no_op_action
                env.step(action)
    
    else:
        runner = build_runner(cfg, env)
        runner.run()

if __name__ == "__main__":
    main()
