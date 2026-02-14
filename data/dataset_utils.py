import shutil
import numpy as np
from tqdm import tqdm
import os, sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, compute_stats, serialize_dict, write_json, STATS_PATH

HF_LEROBOT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))


class LerobotDatasetWriter:
    def __init__(self, 
                 output_path: str, 
                 camera_ids: list = [1],
                 data_type: str = "rgb",
                 action_dim = 13,
                 state_dim = 13, 
                 image_shape = (256,256,3), 
                 depth_shape = (256,256),
                 fps = 10,
                 depth_dmin_m: float = 0.15,
                 depth_dmax_m: float = 1.00,
                ):
        repo_id = output_path
        output_path = os.path.join(HF_LEROBOT_HOME, output_path)
        if os.path.exists(output_path):
            print(f"{output_path} exists, input Y to delete: ")
            if input().strip().lower() == "y":
                shutil.rmtree(output_path)
            else:
                exit(0)

        self.camera_ids = camera_ids
        self.data_type = data_type
        self.image_shape = image_shape
        self.depth_shape = depth_shape
        self.fps = fps
        self.depth_dmin_m = depth_dmin_m
        self.depth_dmax_m = depth_dmax_m

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"qpos_{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"action_{i}" for i in range(action_dim)],
            },
        }

        for cid in camera_ids:
            if "rgb" in data_type:
                features[f"observation.camera_{cid}.rgb"] = {
                    "dtype": "video",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                }
            if "depth" in data_type:
                features[f"observation.camera_{cid}.depth"] = {
                    "dtype": "video",
                    "shape": depth_shape,
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                }
            if "pcl" in data_type:
                raise NotImplementedError

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_path,
            robot_type="MyDexHand",
            fps=fps,
            features=features,
        )
        self.push_to_hub = False
    
    def append_step(self, data: Dict[str, np.ndarray], episode_end: bool = False):
        if 'right_arm_eef_pose' in data:
            state = np.concatenate([data['right_arm_eef_pose'], data['right_hand_qpos']], axis=-1).reshape(-1)
        else:
            state = np.concatenate([data['right_arm_qpos'], data['right_hand_qpos']], axis=-1).reshape(-1)
        action = data['action'].reshape(-1)
        frame_data = {
            'observation.state': state.astype(np.float32),
            'action': action.astype(np.float32),
        }
        for cid in self.camera_ids:
            if "rgb" in self.data_type:
                frame_data[f'observation.camera_{cid}.rgb'] = data[f'camera_{cid}.rgb'].reshape(*self.image_shape).astype(np.uint8)
            if "depth" in self.data_type:
                depth_01 = np.clip(
                    (data[f'camera_{cid}.depth'] - self.depth_dmin_m) / (self.depth_dmax_m - self.depth_dmin_m),
                    0,
                    1
                )
                depth_uint8 = (depth_01 * 255).round().astype(np.uint8)
                frame_data[f'observation.camera_{cid}.depth'] = depth_uint8.reshape(*self.depth_shape)
            if "pcl" in self.data_type:
                raise NotImplementedError

        if 'instruction' in data:
            if isinstance(data['instruction'], str):
                self.text_des = data['instruction']
            elif isinstance(data['instruction'], list):
                self.text_des = data['instruction'][0]
                assert isinstance(self.text_des, str)
            else:
                raise ValueError(f"Unsupported instruction type: {type(data['instruction'])}")
        else:
            self.text_des = "Do the task."

        self.dataset.add_frame(frame_data)

        if episode_end:
            self.dataset.save_episode(task=self.text_des)


class LerobotDatasetReader:
    def __init__(self, repo_id, arm_action_type="right_arm_eef_pose", num_arm_actions=7):
        pth = os.path.join(HF_LEROBOT_HOME, repo_id)
        self.dataset = LeRobotDataset(repo_id, root=pth, local_files_only=True)
        self.episode_ends = [x.item()-1 for x in self.dataset.episode_data_index["to"]]
        self.arm_action_type = arm_action_type
        self.num_arm_actions = num_arm_actions

    def compute_stats(self):
        self.dataset.stop_image_writer()
        self.dataset.meta.stats = compute_stats(self.dataset)
        serialized_stats = serialize_dict(self.dataset.meta.stats)
        write_json(serialized_stats, self.dataset.root / STATS_PATH)
    
    def _get_total_steps(self):
        return len(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, t):
        data = self.dataset[t]
        ret = {}
        for k in data.keys():
            if "rgb" in k:
                ret[k[12:]] = (np.asarray(data[k]).transpose(1,2,0) * 255).astype(np.uint8) 
        ret["instruction"] = data["task"]
        ret[self.arm_action_type] = data["observation.state"][:self.num_arm_actions].cpu().numpy()
        ret["right_hand_qpos"] = data["observation.state"][self.num_arm_actions:].cpu().numpy()
        ret["action"] = data["action"].cpu().numpy()
        return ret

