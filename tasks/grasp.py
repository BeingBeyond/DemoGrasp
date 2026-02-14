'''
The environment to train RL grasping policies
'''
import os, sys, re, pickle
import yaml
import random
import torch
import numpy as np
#from torch.nn import functional as F

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
#import glfw, types
from .reward import REWARD_DICT
from .utils import batch_linear_interpolate_poses, COLORS_DICT, load_object_point_clouds, transform_points
import math
from scipy.spatial.transform import Rotation
from copy import deepcopy

class Grasp(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.init_configs(cfg)

        super().__init__(
            self.cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        print("num obs: {}, num actions: {}".format(self.num_obs, self.num_acts))

        # viewer camera setup
        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.8, -3.2, 3.0)
            cam_target = gymapi.Vec3(1.8, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to eef
        self.j_eef = jacobian[:, self.arm_eef_index, :, :self.num_arm_dofs]


        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_robot_dofs
        ]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        #self.tracking_timestep = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.has_hit_table = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        if self.arm_controller == "qpos" and not self.use_relative_control:
            self.no_op_action = unscale(
                self.robot_dof_default_pos.clone().unsqueeze(0).repeat(self.num_envs, 1)[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
            self.no_op_action = torch.cat([
                self.no_op_action,
                torch.zeros((self.num_envs, self.hand_dof_start_idx - self.num_arm_dofs), dtype=torch.float, device=self.device)
            ], dim=-1)
        elif self.arm_controller == "pose":
            self.no_op_action = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7].clone()
        else:
            self.no_op_action = torch.zeros(
                (self.num_envs, self.hand_dof_start_idx), dtype=torch.float, device=self.device
            )
        
        self.no_op_action = torch.cat([
            self.no_op_action,
            unscale(
                self.robot_dof_default_pos.clone().unsqueeze(0).repeat(self.num_envs, 1)[:, self.active_hand_dof_indices],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
        ], dim=-1)
        self.delta_action_scale = to_torch(self.cfg["env"]["deltaActionScale"], dtype=torch.float, device=self.device)

    def init_configs(self, cfg):
        self.cfg = cfg
        self.reward_type = self.cfg["env"]["rewardType"]
        assert self.reward_type == "binary"
        self.reward_function = REWARD_DICT[self.reward_type]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.debug_vis = self.cfg["env"]["enableDebugVis"]

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.arm_controller = self.cfg["env"]["armController"]
        self.act_max_ang_vel_arm = self.cfg["env"]["actionsMaxAngVelArm"]
        self.act_max_ang_vel_hand = self.cfg["env"]["actionsMaxAngVelHand"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.random_episode_length = self.cfg["env"]["randomEpisodeLength"]
        self.reset_time = -1 #self.cfg["env"].get("resetTime", -1.0)
        assert self.arm_controller in ["qpos", "worlddpose", "eedpose", "pose"]

        self.hand_name = self.cfg["hand_config"]["name"] #self.cfg["hand_name"]
        self.hand_specific_cfg = self.cfg["hand_config"] #self.cfg["hand_specific"][self.cfg["hand_name"]]
        self.palm_offset = self.hand_specific_cfg["palm_offset"]
        
        self.num_obs_dict = self.hand_specific_cfg["num_obs_dict"]
        self.obs_type = self.cfg["env"]["observationType"]
        self.cfg["env"]["numObservations"] = \
            sum([(self.num_obs_dict[i] if i in self.obs_type else 0) for i in self.num_obs_dict]) #self.num_obs_dict[self.obs_type]
        #self.cfg["env"]["numObservations"] = self.hand_specific_cfg["numObs"]
        self.cfg["env"]["numStates"] = 0 
        self.cfg["env"]["numActions"] = self.hand_specific_cfg["numActions"]

        self.render_cfg = self.cfg["env"]["render"]
        self.render_data_type = self.render_cfg["data_type"]
        self.apply_render_randomization = self.render_cfg["randomize"]
        self.render_randomization_params = self.render_cfg["randomization_params"]
        self.camera_cfg = self.cfg["env"]["camera_config"]
        self.use_camera = self.render_cfg["enable"]
        if self.use_camera:
            self.camera_ids = self.render_cfg["camera_ids"]
            self.fixed_camera_ids = [i for i in self.camera_ids if self.camera_cfg[f'camera_{i}']['mount'] == 'fixed']
            self.save_depth_range = self.render_cfg["save_depth_range"]
            self.depth_ranges = [self.camera_cfg[f'camera_{i}']['depth_range'] for i in self.camera_ids]
            self.camera_handles = []
            if "depth" in self.render_data_type or "pcl" in self.render_data_type:
                self.camera_depth_tensor_lists = [[] for i in range(len(self.camera_ids))]
            if "seg" in self.render_data_type:
                self.camera_seg_tensor_lists = [[] for i in range(len(self.camera_ids))]
            self.camera_rgb_tensor_lists = [[] for i in range(len(self.camera_ids))]
            print(f"Camera ids: {self.camera_ids}, fixed camera ids: {self.fixed_camera_ids}")

        # point clouds
        self.enable_pcl = self.cfg["env"]["enablePointCloud"]
        self.points_per_object = self.cfg["env"]["pointsPerObject"]


    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.decimation = self.cfg["sim"]["decimation"]
        self.up_axis_idx = 2 if self.cfg["sim"]["up_axis"] == "z" else 1

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        if self.cfg["env"]["enableRobotTableCollision"]:
            self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # if randamizing, apply once immediately on startup before the first sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _prepare_camera_pad_assets(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        camera_pad_asset = self.gym.load_asset(self.sim, self.asset_root, "camera_pad.urdf", asset_options)

        camera_pad_start_poses = [[] for i in self.fixed_camera_ids]
        for i, cam_id in enumerate(self.fixed_camera_ids):
            pose = gymapi.Transform()
            extrinsics = np.array(self.camera_cfg[f'camera_{cam_id}']['extrinsics'])
            quat = Rotation.from_matrix(extrinsics[:3, :3]).as_quat()
            pose.p = gymapi.Vec3(extrinsics[0][3], extrinsics[1][3], extrinsics[2][3])
            pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            camera_pad_start_poses[i].append(pose)

        return camera_pad_asset, camera_pad_start_poses

    def _load_cameras(self,env_ptr):
        for i, cam_id in enumerate(self.camera_ids):
            name = "camera_{}".format(cam_id)
            intrinsics = self.camera_cfg[name]['intrinsics']
            fx = intrinsics[0][0] #677.63903809
            fy = intrinsics[1][1] #677.48712158 
            cx = intrinsics[0][2] #489.16671753
            cy = intrinsics[1][2] #269.35379028
            fov_x = math.degrees(2 * math.atan(self.camera_cfg[name]['width'] / (2 * fx)))
            cam_props = gymapi.CameraProperties()
            cam_props.width = self.camera_cfg[name]['width']
            cam_props.height = self.camera_cfg[name]['height']
            cam_props.enable_tensors = True
            cam_props.horizontal_fov = fov_x

            ### add fixed cameras
            if cam_id in self.fixed_camera_ids:
                #camera_pose = to_torch(self.camera_cfg[name]['extrinsics'], dtype=torch.double, device=self.device)
                #camera_position = camera_pose[:3, 3]
                #rot_matrix = camera_pose[:3, :3]
                #point_vector = torch.tensor([0.0,0.0,1.0], dtype=torch.double,device=self.device)
                #rot_point_vector = torch.matmul(rot_matrix, point_vector)
                #target_point = camera_position + rot_point_vector
                #self.camera_positions[i].append(camera_position)
                #self.camera_target_positions[i].append(target_point)
                #camera_position = gymapi.Vec3(camera_position[0], camera_position[1], camera_position[2])
                #target_point = gymapi.Vec3(target_point[0], target_point[1], target_point[2])
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                #self.gym.set_camera_location(fix_cam_handle, env_ptr, camera_position, target_point)
                attach_body_handle = self.gym.get_rigid_handle(env_ptr, f"camera_pad_{cam_id}", "camera_pad")
                local_transform = gymapi.Transform()
                local_transform.r = gymapi.Quat.from_euler_zyx(1.5708, -1.5708, 0)
                self.gym.attach_camera_to_body(cam_handle, env_ptr, attach_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            ### add moving wrist camera
            else:
                rigid_body_name = self.camera_cfg[name]['mount']
                rigid_body_handle = self.gym.get_rigid_handle(env_ptr, "robot", rigid_body_name)
                assert rigid_body_handle != -1
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                local_transform = gymapi.Transform()
                local_transform.p = gymapi.Vec3(0, 0, 0)
                #local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(45))
                self.gym.attach_camera_to_body(cam_handle, env_ptr, rigid_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

            raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
            rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
            self.camera_rgb_tensor_lists[i].append(rgb_tensor)
            if "depth" in self.render_data_type or "pcl" in self.render_data_type:
                raw_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_DEPTH)
                depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)
                self.camera_depth_tensor_lists[i].append(depth_tensor)
            if "seg" in self.render_data_type:
                raw_seg_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, cam_handle, gymapi.IMAGE_SEGMENTATION
                )
                seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)
                self.camera_seg_tensor_lists[i].append(seg_tensor)
            self.camera_handles.append(cam_handle)


    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]       
        self.robot_asset_file = self.hand_specific_cfg["robotAssetFile"]
        if self.apply_render_randomization: #self.render_cfg["appearance_realistic"]:
            self.robot_asset_file = self.hand_specific_cfg["robotAssetFileVisualRealistic"]
        robot_asset, robot_dof_props, robot_start_pose = self._prepare_robot_asset(
            self.asset_root, self.robot_asset_file
        )

        self.num_object_shapes = 0
        self.num_object_bodies = 0
        
        # load main objects
        self.multi_object = self.cfg["env"]["asset"]["multiObject"]
        if self.multi_object:
            object_assets = []
            #object_asset_dir = self.cfg["env"]["asset"]["objectAssetDir"]
            object_asset_dir = self.cfg["env"]["asset"]["multiObjectList"].split('/')[0] + '/urdf/'
            #object_asset_fn_list = sorted(os.listdir(os.path.join(self.asset_root, object_asset_dir)))
            with open(os.path.join(self.asset_root, self.cfg["env"]["asset"]["multiObjectList"]), 'r') as f:
                object_asset_fn_list = sorted(yaml.safe_load(f))
            #print(object_asset_dir, object_asset_fn_list)
            self.object_fns = [os.path.join(object_asset_dir, fn) for fn in object_asset_fn_list] # [ObjDatasetName/urdf/xxx.urdf]
            for fn in self.object_fns:
                object_asset, _ = self._prepare_object_asset(self.asset_root, fn)
                object_assets.append(object_asset)
        else:
            object_urdf = self.cfg["env"]["asset"]["objectAssetFile"]
            self.object_names = [object_urdf]
            object_asset, _ = self._prepare_object_asset(self.asset_root, object_urdf)
            self.object_fns = [object_urdf]
        
        # main object pcls
        if self.enable_pcl:
            self.object_pcls = load_object_point_clouds(self.object_fns, self.asset_root)
            self.obj_pcl_buf = torch.zeros((num_envs, self.points_per_object, 3), device=self.device, dtype=torch.float)
        
        # load distractor objects
        self.use_distractor_objects = self.cfg["env"]["asset"]["useDistractorObjects"]
        self.num_distractor_objects = self.cfg["env"]["asset"]["numDistractorObjects"]
        self.random_remove_distractor_objects = self.cfg["env"]["asset"]["randomRemoveDistractorObjects"]
        if self.use_distractor_objects:
            if self.multi_object:
                distractor_object_assets = object_assets
            else:
                distractor_object_urdf = self.cfg["env"]["asset"]["distractorObjectAssetFile"]
                distractor_object_asset, _ = self._prepare_object_asset(self.asset_root, distractor_object_urdf)
        
        table_asset, table_start_poses, mat_asset, mat_start_pose, wall_asset, wall_start_pose,\
             wooden_table_asset, wooden_table_start_pose = self._prepare_table_asset()

        # camera pads
        if self.use_camera:
            camera_pad_asset, camera_pad_start_poses = self._prepare_camera_pad_assets()
            self.camera_pad_indices = [[] for i in self.fixed_camera_ids]
            self.camera_pad_start_states = [[] for i in self.fixed_camera_ids]

        if self.render_cfg["appearance_realistic"]:
            self.wall_indices, self.mat_indices, self.wooden_table_indices = [], [], []

        self.envs = []
        #self.robots = []
        #self.objects = []
        self.eef_idx =  []
        self.robot_indices = []
        self.object_indices = []
        self.distractor_object_indices = []
        self.table_indices = []
        self.robot_start_states = []
        self.pc_features = []
        self.table_start_pos, self.mat_start_pos = [], []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # aggregate size
            max_agg_bodies = self.num_robot_bodies + self.num_object_bodies + 1 # robot + object + table
            max_agg_shapes = self.num_robot_shapes + self.num_object_shapes + 1
            if self.use_distractor_objects:
                max_agg_bodies += self.num_distractor_objects * self.num_object_bodies
                max_agg_shapes += self.num_distractor_objects * self.num_object_shapes
            if self.use_camera:
                # add camera pads
                max_agg_bodies += len(self.fixed_camera_ids)
                max_agg_shapes += len(self.fixed_camera_ids)
            if self.render_cfg["appearance_realistic"]:
                # add mat and wall bodies
                max_agg_bodies += 3
                max_agg_shapes += 3
            if self.aggregate_mode > 0:
               self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot actor
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_start_pose, "robot", i, -1 if self.cfg["env"]["enableSelfCollision"] else 1, 1 # seg id=1
            )
            self.robot_start_states.append(
                [robot_start_pose.p.x,robot_start_pose.p.y,robot_start_pose.p.z,
                robot_start_pose.r.x,robot_start_pose.r.y,robot_start_pose.r.z,
                robot_start_pose.r.w,0,0,0,0,0,0,])
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(
                env_ptr, robot_actor, gymapi.DOMAIN_SIM
            )
            self.robot_indices.append(robot_idx)

            # add object
            if self.multi_object:
                object_asset = object_assets[i % len(object_assets)]
            object_handle = self.gym.create_actor(
                env_ptr, object_asset, gymapi.Transform(), "object", i, -1, 2 # seg id=2
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)

            # add object point cloud to buffer
            if self.enable_pcl:
                self.obj_pcl_buf[i] = to_torch(self.object_pcls[i % len(self.object_pcls)], dtype=torch.float32, device=self.device)

            # set object friction
            object_rb_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for j in range(len(object_rb_props)):
                object_rb_props[j].friction = self.cfg["env"]["objectFriction"]
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_rb_props)

            # add distractor objects
            if self.use_distractor_objects:
                for j in range(self.num_distractor_objects):
                    if self.multi_object:
                        ast = random.choice(distractor_object_assets)
                    else:
                        ast = distractor_object_asset
                    object_handle = self.gym.create_actor(
                        env_ptr, ast, gymapi.Transform(), "distractor", i, -1, 0
                    )
                    object_idx = self.gym.get_actor_index(
                        env_ptr, object_handle, gymapi.DOMAIN_SIM
                    )
                    self.distractor_object_indices.append(object_idx)

            # add table
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_start_poses[i], "table", i, -1, 0
            )
            table_idx = self.gym.get_actor_index(
                env_ptr, table_handle, gymapi.DOMAIN_SIM
            )
            self.table_indices.append(table_idx)
            self.table_start_pos.append([table_start_poses[i].p.x, table_start_poses[i].p.y, table_start_poses[i].p.z])

            if (not self.cfg["env"]["enableRobotTableCollision"]) and i<self.num_envs//2:
                assert not self.cfg["env"]["enableSelfCollision"] # robot collision filter should not be 0
                props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
                #print([p.filter for p in props])
                props[0].filter = 1
                self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, props)

            # add other visual parts; change color and texture
            if self.render_cfg["appearance_realistic"]:
                # add mat
                mat_handle = self.gym.create_actor(
                    env_ptr, mat_asset, mat_start_pose, "mat", i, -1, 0
                )
                mat_idx = self.gym.get_actor_index(
                    env_ptr, mat_handle, gymapi.DOMAIN_SIM
                )
                self.mat_indices.append(mat_idx)
                self.mat_start_pos.append([mat_start_pose.p.x, mat_start_pose.p.y, mat_start_pose.p.z])
                # add wall
                wall_handle = self.gym.create_actor(
                    env_ptr, wall_asset, wall_start_pose, "wall", i, -1, 0
                )
                wall_idx = self.gym.get_actor_index(
                    env_ptr, wall_handle, gymapi.DOMAIN_SIM
                )
                self.wall_indices.append(wall_idx)
                # add wooden table
                wooden_table_handle = self.gym.create_actor(
                    env_ptr, wooden_table_asset, wooden_table_start_pose, "wooden_table", i, -1, 0
                )
                wooden_table_idx = self.gym.get_actor_index(
                    env_ptr, wooden_table_handle, gymapi.DOMAIN_SIM
                )
                self.wooden_table_indices.append(wooden_table_idx)

                if (not self.cfg["env"]["enableRobotTableCollision"]) and i<self.num_envs//2:
                    assert not self.cfg["env"]["enableSelfCollision"] # robot collision filter should not be 0
                    props = self.gym.get_actor_rigid_shape_properties(env_ptr, mat_handle)
                    props[0].filter = 1
                    self.gym.set_actor_rigid_shape_properties(env_ptr, mat_handle, props)
                    props = self.gym.get_actor_rigid_shape_properties(env_ptr, wooden_table_handle)
                    props[0].filter = 1
                    self.gym.set_actor_rigid_shape_properties(env_ptr, wooden_table_handle, props)

            if self.apply_render_randomization:
                self.object_random_texture = self.render_randomization_params["object_random_texture"]
                self.object_color_choices = self.render_randomization_params["object_color_choices"]
                self.wall_color = [200/255, 200/255, 200/255]
                self.gym.set_rigid_body_color(env_ptr, wall_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*self.wall_color))

            # add camera pads
            if self.use_camera:
                for j, cam_id in enumerate(self.fixed_camera_ids):
                    camera_pad_handle = self.gym.create_actor(
                        env_ptr, camera_pad_asset, camera_pad_start_poses[j][0], "camera_pad_{}".format(cam_id), i, -1, 0
                    )
                    camera_pad_idx = self.gym.get_actor_index(
                        env_ptr, camera_pad_handle, gymapi.DOMAIN_SIM
                    )
                    self.camera_pad_indices[j].append(camera_pad_idx)
                    self.camera_pad_start_states[j].append(
                        [camera_pad_start_poses[j][0].p.x, camera_pad_start_poses[j][0].p.y, camera_pad_start_poses[j][0].p.z,
                        camera_pad_start_poses[j][0].r.x, camera_pad_start_poses[j][0].r.y, camera_pad_start_poses[j][0].r.z,
                        camera_pad_start_poses[j][0].r.w, 0, 0, 0, 0, 0, 0]
                    )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            eef_idx = self.gym.find_actor_rigid_body_index(
                env_ptr, robot_actor, self.hand_specific_cfg["eef_link"], gymapi.DOMAIN_SIM
            )
            self.eef_idx.append(eef_idx)

            if self.use_camera:
                self._load_cameras(env_ptr)

        self.pc_features = to_torch(
            self.pc_features, device=self.device
        )
        self.robot_start_states = to_torch(
            self.robot_start_states, device=self.device
        ).view(num_envs, 13)
        self.object_init_states = to_torch(
            self.object_init_states, device=self.device
        ).view(num_envs, 13)
        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.palm_handle = to_torch(
            self.palm_handle, dtype=torch.long, device=self.device
        )
        self.robot_indices = to_torch(
            self.robot_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        if self.use_distractor_objects:
            self.distractor_object_indices = to_torch(
                self.distractor_object_indices, dtype=torch.long, device=self.device
            ).view(self.num_envs, self.num_distractor_objects)
        self.table_indices = to_torch(
            self.table_indices, dtype=torch.long, device=self.device
        )
        self.eef_idx = to_torch(self.eef_idx, dtype=torch.long, device=self.device)
        self.reset_position_range = to_torch(self.cfg["env"]["resetPositionRange"], dtype=torch.float, device=self.device)
        self.reset_random_rot = self.cfg["env"]["resetRandomRot"]
        self.table_height_range = to_torch(self.cfg["env"]["tableHeightRange"], dtype=torch.float, device=self.device)
        self.ee_safe_workspace = to_torch(self.cfg["env"]["eeSafeWorkspace"], dtype=torch.float, device=self.device)
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_hand_dof_pos_full_range = self.cfg["env"]["resetHandDofPosFullRange"]
        self.table_start_pos = to_torch(
            self.table_start_pos, dtype=torch.float, device=self.device
        )

        if self.use_camera:
            self.camera_pad_indices = [to_torch(
                self.camera_pad_indices[i], dtype=torch.long, device=self.device
            ) for i in range(len(self.fixed_camera_ids))]
            self.camera_pad_start_states = to_torch(
                self.camera_pad_start_states, dtype=torch.float, device=self.device
            )
            self.rgb_tensors = [torch.zeros((self.num_envs, self.camera_cfg[f'camera_{i}']['width'], 
                                             self.camera_cfg[f'camera_{i}']['height'], 4), dtype=torch.uint8, device=self.device)
                                             for i in self.camera_ids]
            if "depth" in self.render_data_type or "pcl" in self.render_data_type:
                self.depth_tensors = [torch.zeros((self.num_envs, self.camera_cfg[f'camera_{i}']['width'],  
                                                    self.camera_cfg[f'camera_{i}']['height']), dtype=torch.float, device=self.device)
                                                    for i in self.camera_ids]
            if "seg" in self.render_data_type:
                self.seg_tensors = [torch.zeros((self.num_envs, self.camera_cfg[f'camera_{i}']['width'],  
                                                    self.camera_cfg[f'camera_{i}']['height']), dtype=torch.int, device=self.device)
                                                    for i in self.camera_ids]
            
        if self.render_cfg["appearance_realistic"]:
            self.wall_indices = to_torch(
                self.wall_indices, dtype=torch.long, device=self.device
            )
            self.mat_indices = to_torch(
                self.mat_indices, dtype=torch.long, device=self.device
            )
            self.wooden_table_indices = to_torch(
                self.wooden_table_indices, dtype=torch.long, device=self.device
            )
            self.mat_start_pos = to_torch(
                self.mat_start_pos, dtype=torch.float, device=self.device
            )

        # load tracking reference
        with open(self.cfg["env"]["trackingReferenceFile"], "rb") as f:
            self.tracking_reference = pickle.load(f)
        for key in self.tracking_reference:
            self.tracking_reference[key] = to_torch(
                self.tracking_reference[key], dtype=torch.float, device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T_ref = self.tracking_reference["wrist_initobj_pos"].shape[1]
        self.T_ref_start_lifting = self.cfg["env"]["trackingReferenceLiftTimestep"]
        self.randomize_tracking_reference = self.cfg["env"]["randomizeTrackingReference"]
        self.randomize_tracking_reference_range = to_torch(self.cfg["env"]["randomizeTrackingReferenceRange"], dtype=torch.float, device=self.device)
        self.randomize_grasp_pose = self.cfg["env"]["randomizeGraspPose"]
        self.randomize_grasp_pose_range = self.cfg["env"]["randomizeGraspPoseRange"]
        # initialize reaching plan
        self.reaching_plan_ee = torch.zeros(
            (self.num_envs, self.max_episode_length, 7),
            dtype=torch.float32, device=self.device
        )
        self.reaching_plan_timesteps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # instruction
        self.use_advanced_instruction = self.render_cfg["use_advanced_instruction"]
        if self.use_advanced_instruction:
            self.instruction_template = self.render_cfg["advanced_instruction_template"]
            object_name_list = yaml.safe_load(open(
                os.path.join(self.asset_root, self.render_cfg["object_name_list"]), 'r'
            ))
            self.object_name_list = [object_name_list[i%len(object_name_list)] for i in range(self.num_envs)]
        else:
            self.instruction_template = self.render_cfg["instruction_template"]


    def _prepare_robot_asset(self, asset_root, asset_file):
        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.cfg["env"]["useRobotVhacd"]:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0

        print("Loading robot asset: ", asset_root, asset_file)
        #asset_file = 'robot/inspire_tac/fr3_inspire_tac_L_right_safety.urdf'
        #exit(0)
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # get asset info
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        #self.num_robot_actuators = self.num_robot_dofs
        print("self.num_robot_bodies: ", self.num_robot_bodies)
        print("self.num_robot_shapes: ", self.num_robot_shapes)
        print("self.num_robot_dofs: ", self.num_robot_dofs)
        #print("self.num_robot_actuators: ", self.num_robot_actuators)

        self.palm = self.hand_specific_cfg["palm_link"] #"palm_lower"
        self.fingertips = self.hand_specific_cfg["fingertips_link"]
        self.num_fingers = len(self.fingertips)
        self.num_arm_dofs = self.hand_specific_cfg["num_arm_dofs"]
        self.robot_dof_names = []
        for i in range(self.num_robot_dofs):
            joint_name = self.gym.get_asset_dof_name(robot_asset, i)
            self.robot_dof_names.append(joint_name)
        
        self.palm_handle = self.gym.find_asset_rigid_body_index(robot_asset, self.palm)
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(robot_asset, fingertip)
            for fingertip in self.fingertips
        ]
        if -1 in self.fingertip_handles or self.palm_handle==-1:
            raise Exception("Fingertip names or palm name not found!")
        self.arm_dof_names = self.hand_specific_cfg["arm_dof_names"]
        self.arm_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in self.arm_dof_names
        ]
        self.hand_dof_names = []
        for name in self.robot_dof_names:
            if name not in self.arm_dof_names:
                self.hand_dof_names.append(name)
        self.hand_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in self.hand_dof_names
        ]
        self.robot_dof_indices = self.arm_dof_indices + self.hand_dof_indices
        self.robot_dof_indices = to_torch(
            self.robot_dof_indices, dtype=torch.long, device=self.device
        )
        self.hand_dof_start_idx = 7 #len(self.arm_dof_indices)

        # process tendon joints
        if "passive_joints" in self.hand_specific_cfg:
            self.have_passive_joints = True
            self.passive_hand_dof_indices, self.mimic_parent_dof_indices, self.mimic_multipliers = [],[],[]
            for k, v in self.hand_specific_cfg["passive_joints"].items():
                self.passive_hand_dof_indices.append(self.gym.find_asset_dof_index(robot_asset, k))
                self.mimic_parent_dof_indices.append(self.gym.find_asset_dof_index(robot_asset, v["mimic"]))
                self.mimic_multipliers.append(v["multiplier"])
            #print("Passive joints:", self.hand_specific_cfg["passive_joints"])
            #print(self.passive_hand_dof_indices, self.mimic_parent_dof_indices, self.mimic_multipliers)
            self.active_hand_dof_indices = []
            for i in self.hand_dof_indices:
                if i not in self.passive_hand_dof_indices:
                    self.active_hand_dof_indices.append(i)
            #print(self.active_hand_dof_indices)
            self.mimic_multipliers = to_torch(self.mimic_multipliers, device=self.device)
        else:
            self.have_passive_joints = False
            self.active_hand_dof_indices = self.hand_dof_indices
        self.active_robot_dof_indices = self.arm_dof_indices + self.active_hand_dof_indices
        self.active_robot_dof_indices = to_torch(
            self.active_robot_dof_indices, dtype=torch.long, device=self.device
        )
        self.active_robot_dof_names = [self.robot_dof_names[i] for i in self.active_robot_dof_indices]
        print("Active dof names:", self.active_robot_dof_names)
        self.active_hand_dof_names = [self.robot_dof_names[i] for i in self.active_hand_dof_indices]
        print("Active hand dof names:", self.active_hand_dof_names)
        print("Hand dof names:", self.hand_dof_names)

        # count dofs
        assert self.arm_dof_indices == [i for i in range(self.num_arm_dofs)]
        print("arm dof indices, active hand dof indices, hand dof start idx:", \
              self.arm_dof_indices, self.active_hand_dof_indices, self.hand_dof_start_idx)
        assert self.num_arm_dofs == len(self.arm_dof_indices)
        self.num_hand_dofs = len(self.hand_dof_indices)
        self.num_active_hand_dofs = len(self.active_hand_dof_indices)
        self.num_passive_hand_dofs = len(self.passive_hand_dof_indices) if self.have_passive_joints else 0
        #self.num_active_robot_dofs = self.num_arm_dofs + self.num_active_hand_dofs
        assert self.num_arm_dofs+self.num_hand_dofs==self.num_robot_dofs

        # get eef index
        robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.arm_eef_index = robot_link_dict[self.hand_specific_cfg["eef_link"]]

        # dof properties
        self.default_dof_pos = np.array(self.hand_specific_cfg["default_dof_pos"], dtype=np.float32)
        print("Default DoF positions: ", self.default_dof_pos)
        assert self.num_robot_dofs == len(self.default_dof_pos)
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.robot_dof_default_pos = []
        self.robot_dof_default_vel = []

        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.robot_dof_upper_limits.append(robot_dof_props["upper"][i])
            self.robot_dof_default_pos.append(self.default_dof_pos[i])
            self.robot_dof_default_vel.append(0.0)
        
            # large kp, kd to simulate position control
            if i in self.arm_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 16000 * self.cfg["env"]["pdParamScale"]
                robot_dof_props["damping"][i] = 600 * self.cfg["env"]["pdParamScale"]
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
            elif i in self.hand_dof_indices:
                robot_dof_props["driveMode"][i] = 1
                robot_dof_props["stiffness"][i] = 600 * self.cfg["env"]["pdParamScale"]
                robot_dof_props["damping"][i] = 20 * self.cfg["env"]["pdParamScale"]
                robot_dof_props["friction"][i] = 0.01
                robot_dof_props["armature"][i] = 0.001
            print('DoF {} effort {:.2} stiffness {:.2} damping {:.2} friction {:.2} armature {:.2} limit {:.2}~{:.2}'.format(
                self.robot_dof_names[(self.arm_dof_indices + self.hand_dof_indices).index(i)], 
                robot_dof_props['effort'][i], robot_dof_props['stiffness'][i],
                robot_dof_props['damping'][i], robot_dof_props['friction'][i],
                robot_dof_props['armature'][i], robot_dof_props['lower'][i], 
                robot_dof_props['upper'][i]))


        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_default_pos = to_torch(self.robot_dof_default_pos, device=self.device)
        self.active_robot_dof_default_pos = self.robot_dof_default_pos[:self.num_arm_dofs + self.num_active_hand_dofs]
        self.robot_dof_default_vel = to_torch(self.robot_dof_default_vel, device=self.device)
        print(f"Arm DoF limits: {[(i.item(),j.item()) for (i,j) in zip(self.robot_dof_lower_limits[self.arm_dof_indices], self.robot_dof_upper_limits[self.arm_dof_indices])]}")
        print(f"Hand DoF limits: {[(i.item(),j.item()) for (i,j) in zip(self.robot_dof_lower_limits[self.hand_dof_indices], self.robot_dof_upper_limits[self.hand_dof_indices])]}")
        print(f"Active Hand Dof Lower: {self.robot_dof_lower_limits[self.active_hand_dof_indices]}")
        print(f"Active Hand Dof Upper: {self.robot_dof_upper_limits[self.active_hand_dof_indices]}")

        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0, 0, 0)
        robot_start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        #print(robot_start_pose.p, robot_start_pose.r)
        return robot_asset, robot_dof_props, robot_start_pose


    def _prepare_object_asset(self, asset_root, asset_file):
        # load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.cfg["env"]["useObjectVhacd"]:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # drive_mode: 0: none, 1: position, 2: velocity, 3: force
        asset_options.default_dof_drive_mode = 0

        object_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # get object asset info
        self.num_object_bodies = max(self.num_object_bodies, self.gym.get_asset_rigid_body_count(object_asset))
        self.num_object_shapes = max(self.num_object_shapes, self.gym.get_asset_rigid_shape_count(object_asset))
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)

        object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []
        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props["lower"][i])
            self.object_dof_upper_limits.append(object_dof_props["upper"][i])

        self.object_dof_lower_limits = to_torch(
            self.object_dof_lower_limits, device=self.device
        )
        self.object_dof_upper_limits = to_torch(
            self.object_dof_upper_limits, device=self.device
        )
        self.object_init_states = to_torch([0.,0.,0.1, 0.,0.,0.,1., 0.,0.,0.,0.,0.,0.], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        
        return object_asset, object_dof_props
    
    def _prepare_table_asset(self):
        self.table_thickness = 0.3
        self.table_heights = to_torch(self.cfg["env"]["tableHeightRange"][0], dtype=torch.float, device=self.device).repeat(self.num_envs)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        
        if self.render_cfg['appearance_realistic']:
            mat_thickness = 0.003
            mat_dims = gymapi.Vec3(0.6, 0.8, mat_thickness)
            table_dims = gymapi.Vec3(0.9, 0.8, self.table_thickness)
            table_start_pose = gymapi.Transform()
            table_start_pose.p = gymapi.Vec3(0.51, -0.075, self.table_heights[0] - mat_thickness - self.table_thickness/2)
            table_start_poses = [table_start_pose] * self.num_envs
            table_asset = self.gym.create_box(
                self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
            )

            mat_start_pose = gymapi.Transform()
            mat_start_pose.p = gymapi.Vec3(0.51, -0.075, self.table_heights[0] - mat_thickness/2)
            mat_asset = self.gym.create_box(
                self.sim, mat_dims.x, mat_dims.y, mat_dims.z, asset_options
            )

            wall_dims = gymapi.Vec3(0.1, 2, 1.5)
            wall_start_pose = gymapi.Transform()
            wall_start_pose.p = gymapi.Vec3(-0.5, -0.075, 0.75)
            wall_asset = self.gym.create_box(
                self.sim, wall_dims.x, wall_dims.y, wall_dims.z, asset_options
            )

            wooden_table_dims = gymapi.Vec3(0.9, 1.5, 0.004) #(1.0, 0.4, 0.01)
            wooden_table_start_pose = gymapi.Transform()
            wooden_table_start_pose.p = gymapi.Vec3(0.51, -0.075, 0.002) #(0.51, -0.075-0.6, 0.005)
            wooden_table_asset = self.gym.create_box(
                self.sim, wooden_table_dims.x, wooden_table_dims.y, wooden_table_dims.z, asset_options
            )

            if self.apply_render_randomization:
                # table textures
                texture_fns = sorted(os.listdir(os.path.join(self.asset_root, self.render_randomization_params["texture_folder"], "background")))
                self.background_texture_handles = []
                for fn in texture_fns:
                    if fn.endswith(".jpg") or fn.endswith(".png"):
                        texture_handle = self.gym.create_texture_from_file(
                            self.sim, os.path.join(self.asset_root, self.render_randomization_params["texture_folder"], "background", fn)
                        )
                        self.background_texture_handles.append(texture_handle)
                    print(f"Loaded background texture: {fn}.")

                # object textures
                texture_fns = sorted(os.listdir(os.path.join(self.asset_root, self.render_randomization_params["texture_folder"], "object")))
                self.object_texture_handles = []
                for fn in texture_fns:
                    if fn.endswith(".jpg") or fn.endswith(".png"):
                        texture_handle = self.gym.create_texture_from_file(
                            self.sim, os.path.join(self.asset_root, self.render_randomization_params["texture_folder"], "object", fn)
                        )
                        self.object_texture_handles.append(texture_handle)
                    print(f"Loaded object texture: {fn}.")
                self.white_texture = self.gym.create_texture_from_file(
                    self.sim, os.path.join(self.asset_root, self.render_randomization_params["texture_folder"], "white.png")
                )

            return table_asset, table_start_poses, mat_asset, mat_start_pose, wall_asset, wall_start_pose, \
                wooden_table_asset, wooden_table_start_pose
        
        else:
            table_dims = gymapi.Vec3(1, 1, self.table_thickness)
            table_start_pose = gymapi.Transform()
            table_start_pose.p = gymapi.Vec3(0.6, 0, self.table_heights[0] - self.table_thickness/2)
            table_start_poses = [table_start_pose] * self.num_envs
            table_asset = self.gym.create_box(
                self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
            )
            return table_asset, table_start_poses, None, None, None, None, None, None


    def reset_idx(self, env_ids, object_init_pose=None, **kwargs):
        ## randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ## reset table heights
        if not self.render_cfg["appearance_realistic"]:
            self.table_heights[env_ids] = torch_rand_float(
                self.table_height_range[0], self.table_height_range[1], (len(env_ids),1), device=self.device
            ).view(-1)
            self.root_state_tensor[self.table_indices[env_ids], 2] = self.table_heights[env_ids] - self.table_thickness/2

        ## reset object
        # apply random rotation
        rand_rot_axis = np.random.randn(len(env_ids), 3)
        if self.reset_random_rot == "z":
            rand_rot_axis[:] = np.array([0, 0, 1])
        rand_rot_axis = to_torch(rand_rot_axis / np.linalg.norm(rand_rot_axis, axis=1, keepdims=True), device=self.device)
        rand_angle = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device)
        if self.reset_random_rot == "fixed":
            rand_angle[:] = 0.0
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = (
            quat_from_angle_axis(rand_angle[:,-1], rand_rot_axis)
        )
        # sample random xyz
        samples = self.reset_position_range[:, 0] + (self.reset_position_range[:, 1] - self.reset_position_range[:, 0]) * torch.rand(len(env_ids), 3).to(self.device)
        samples[:, 2] += self.table_heights[env_ids] # add table height
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = samples
        self.root_state_tensor[self.object_indices[env_ids], 7:] = 0
        # if use predefined object pose
        if object_init_pose is not None:
            self.root_state_tensor[self.object_indices[env_ids], 0:7] = to_torch(object_init_pose, device=self.device)

        ## reset distractor objects
        if not self.use_distractor_objects:
            table_object_indices = torch.cat(
                [self.table_indices[env_ids], self.object_indices[env_ids]], dim=0
            ).to(torch.int32)
        else:
            #print(self.distractor_object_indices, env_ids)
            distractor_object_indices = self.distractor_object_indices[env_ids].view(-1).to(torch.int32)
            rand_rot_axis = np.random.randn(len(env_ids)*self.num_distractor_objects, 3)
            if self.reset_random_rot == "z":
                rand_rot_axis[:] = np.array([0, 0, 1])
            rand_rot_axis = to_torch(rand_rot_axis / np.linalg.norm(rand_rot_axis, axis=1, keepdims=True), device=self.device)
            rand_angle = torch_rand_float(-np.pi, np.pi, (len(env_ids)*self.num_distractor_objects, 1), device=self.device)
            if self.reset_random_rot == "fixed":
                rand_angle[:] = 0.0
            self.root_state_tensor[distractor_object_indices, 3:7] = (
                quat_from_angle_axis(rand_angle[:,-1], rand_rot_axis)
            )
            # sample random xyz
            samples = self.reset_position_range[:, 0] + (self.reset_position_range[:, 1] - self.reset_position_range[:, 0]) * torch.rand(len(env_ids)*self.num_distractor_objects, 3).to(self.device)
            samples[:, 2] += self.table_heights[env_ids].unsqueeze(1).repeat(1, self.num_distractor_objects).reshape(-1) # add table height
            self.root_state_tensor[distractor_object_indices, 0:3] = samples
            self.root_state_tensor[distractor_object_indices, 7:] = 0
            # hide some distractor objects
            mask = torch.rand_like(distractor_object_indices.float()) < self.random_remove_distractor_objects
            remove_distractor_object_indices = distractor_object_indices[mask]
            if remove_distractor_object_indices.shape[0] > 0:
                self.root_state_tensor[remove_distractor_object_indices, 0:2] = to_torch([-0.2, 0], device=self.device)
                self.root_state_tensor[remove_distractor_object_indices, 2] = torch_rand_float(0.1, 0.3, (remove_distractor_object_indices.shape[0], 1), device=self.device).view(-1)
            # total indices to reset
            table_object_indices = torch.cat(
                [self.table_indices[env_ids], self.object_indices[env_ids], distractor_object_indices], dim=0
            ).to(torch.int32)
        
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(table_object_indices),
            len(table_object_indices),
        )

        ## reset robot
        delta_max = self.robot_dof_upper_limits - self.robot_dof_default_pos
        delta_min = self.robot_dof_lower_limits - self.robot_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            torch_rand_float(-1.0,1.0,(len(env_ids),self.num_robot_dofs),device=self.device) + 1.0
        )
        pos = self.robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        if self.reset_hand_dof_pos_full_range:
            pos[:, self.hand_dof_start_idx:] = self.robot_dof_default_pos[self.hand_dof_start_idx:] + \
                rand_delta[:, self.hand_dof_start_idx:]
        self.robot_dof_pos[env_ids, :] = pos
        self.robot_dof_vel[env_ids, :] = self.robot_dof_default_vel
        self.prev_targets[env_ids, : self.num_robot_dofs] = pos.clone()
        self.cur_targets[env_ids, : self.num_robot_dofs] = pos.clone()

        robot_indices = self.robot_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.robot_dof_state),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )

        if self.random_episode_length:
            self.progress_buf[env_ids] = torch.randint(0, 10, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        #self.tracking_timestep[env_ids] = 0
        self.has_hit_table[env_ids] = 0

        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        if not self.random_episode_length:
            ## let object fall for 2 seconds
            ts = int(2/self.dt)
            for t in range(ts):
                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
        
        # reset camera
        if self.use_camera:
            ### domain randomization
            if self.apply_render_randomization:
                if len(self.camera_ids) > 0:
                    if len(self.fixed_camera_ids) > 0:
                        # randomize fixed camera extrinsics
                        camera_positions = self.camera_pad_start_states[:, env_ids, 0:3] + \
                            torch_rand_float(self.render_randomization_params['camera_pos'][0], 
                                            self.render_randomization_params['camera_pos'][1], 
                                            (len(self.fixed_camera_ids) * len(env_ids), 3),
                                            device=self.device).reshape(len(self.fixed_camera_ids), len(env_ids), 3)
                        camera_quaternions = self.camera_pad_start_states[:, env_ids, 3:7] + \
                            torch_rand_float(self.render_randomization_params['camera_quat'][0], 
                                            self.render_randomization_params['camera_quat'][1],
                                            (len(self.fixed_camera_ids) * len(env_ids), 4),
                                            device=self.device).reshape(len(self.fixed_camera_ids), len(env_ids), 4)
                        for i in range(len(self.fixed_camera_ids)):
                            self.root_state_tensor[self.camera_pad_indices[i][env_ids], 0:3] = camera_positions[i]
                            self.root_state_tensor[self.camera_pad_indices[i][env_ids], 3:7] = camera_quaternions[i]
                        indices = torch.cat([d[env_ids] for d in self.camera_pad_indices]).reshape(-1).to(torch.int32)
                        #print(indices)
                        self.gym.set_actor_root_state_tensor_indexed(
                            self.sim,
                            gymtorch.unwrap_tensor(self.root_state_tensor),
                            gymtorch.unwrap_tensor(indices),
                            len(indices),
                        )
                        self.gym.simulate(self.sim)

                    # randomize camera intrinsics
                    self.depth_ranges = np.array([self.camera_cfg[f'camera_{i}']['depth_range'] for i in self.camera_ids])
                    self.depth_ranges += np.random.uniform(
                        -self.render_randomization_params['depth_range'],
                        self.render_randomization_params['depth_range'],
                        size=self.depth_ranges.shape
                    )

                # randomize light parameters
                light_intensity_range = self.render_randomization_params['light_intensity']
                light_ambient_range = self.render_randomization_params['light_ambient']
                for i in range(self.render_randomization_params['num_lights']):
                    l_intensity = gymapi.Vec3(*([random.uniform(*light_intensity_range)]*3))
                    l_ambient = gymapi.Vec3(*[random.uniform(*light_ambient_range)]*3)
                    l_direction = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
                    self.gym.set_light_parameters(self.sim, i, l_intensity, l_ambient, l_direction)

                # randomize colors and texture
                self.instructions = []
                for env_id in env_ids:
                    # object and wall colors
                    color_name = random.choice(self.object_color_choices)
                    #if "{COLOR}" in self.instruction_template:
                    #    assert not self.object_random_texture
                    #    self.instructions.append(self.instruction_template.replace("{COLOR}", color_name))
                    if self.use_advanced_instruction:
                        assert not self.object_random_texture
                        self.instructions.append(
                            self.instruction_template.replace("{COLOR}", color_name).replace(
                                "{OBJ}", self.object_name_list[env_id]
                            )
                        )
                        object_color = np.array(COLORS_DICT[color_name])
                    else:
                        self.instructions.append(self.instruction_template)
                        # only add color randomization when not specifying color in instruction
                        object_color = np.array(COLORS_DICT[color_name]) + np.random.uniform(
                            -self.render_randomization_params['color'],
                            self.render_randomization_params['color'],
                            size=3
                        )
                        object_color = np.clip(object_color, 0, 1)
                    if self.object_random_texture:
                        # 67% prob to apply a random texture
                        if np.random.rand() < 0.67:
                            # 50% prob to make the object white
                            if np.random.rand() < 0.5:
                                object_color[:] = 1
                            self.gym.set_rigid_body_texture(
                                self.envs[env_id], self.object_indices[0], 0, 
                                gymapi.MESH_VISUAL, random.choice(self.object_texture_handles)
                            )
                        # else, no texture
                        else:
                            self.gym.set_rigid_body_texture(
                                self.envs[env_id], self.object_indices[0], 0, 
                                gymapi.MESH_VISUAL, self.white_texture
                            )
                    self.gym.set_rigid_body_color(
                        self.envs[env_id], self.object_indices[0], 0,
                        gymapi.MESH_VISUAL, gymapi.Vec3(*object_color)
                    )
                    wall_color = np.array(self.wall_color) + np.random.uniform(
                        -self.render_randomization_params['color'],
                        self.render_randomization_params['color'],
                        size=3
                    )
                    wall_color = np.clip(wall_color, 0, 1)
                    self.gym.set_rigid_body_color(
                        self.envs[env_id], self.wall_indices[0], 0,
                        gymapi.MESH_VISUAL, gymapi.Vec3(*wall_color)
                    )
                    # distractor object colors
                    if self.use_distractor_objects:
                        distractor_object_color_choices = self.object_color_choices.copy()
                        # distractor_object_color_choices.remove(color_name)
                        for i in range(self.num_distractor_objects):
                            distractor_color = np.array(COLORS_DICT[random.choice(distractor_object_color_choices)]) + \
                                np.random.uniform(
                                    -self.render_randomization_params['color'],
                                    self.render_randomization_params['color'],
                                    size=3
                                )
                            distractor_color = np.clip(distractor_color, 0, 1)
                            if self.object_random_texture:
                                # 67% prob to apply a random texture
                                if np.random.rand() < 0.67:
                                    # 50% prob to make the object white
                                    if np.random.rand() < 0.5:
                                        distractor_color[:] = 1
                                    self.gym.set_rigid_body_texture(
                                        self.envs[env_id], self.distractor_object_indices[0][i], 0, 
                                        gymapi.MESH_VISUAL, random.choice(self.object_texture_handles)
                                    )
                                # else, no texture
                                else:
                                    self.gym.set_rigid_body_texture(
                                        self.envs[env_id], self.distractor_object_indices[0][i], 0, 
                                        gymapi.MESH_VISUAL, self.white_texture
                                    )
                            self.gym.set_rigid_body_color(
                                self.envs[env_id], self.distractor_object_indices[0][i], 0,
                                gymapi.MESH_VISUAL, gymapi.Vec3(*distractor_color)
                            )
                    # table texture
                    self.gym.set_rigid_body_texture(self.envs[env_id], self.mat_indices[0], 0, 
                        gymapi.MESH_VISUAL, random.choice(self.background_texture_handles))
                    self.gym.set_rigid_body_texture(self.envs[env_id], self.table_indices[0], 0, 
                        gymapi.MESH_VISUAL, random.choice(self.background_texture_handles))
                    self.gym.set_rigid_body_texture(self.envs[env_id], self.wooden_table_indices[0], 0, 
                        gymapi.MESH_VISUAL, random.choice(self.background_texture_handles))

                # table positions
                noise_table = torch_rand_float(-1, 1, (len(env_ids), 3), device=self.device) * \
                    to_torch(self.render_randomization_params['table_xyz'], device=self.device)
                noise_mat = torch_rand_float(-1, 1, (len(env_ids), 3), device=self.device) * \
                    to_torch(self.render_randomization_params['table_xyz'], device=self.device)
                noise_mat[:, 2] = noise_table[:, 2] # mat and table moves together on z-axis
                self.root_state_tensor[self.table_indices[env_ids], 0:3] = self.table_start_pos[env_ids, 0:3] + noise_table
                self.root_state_tensor[self.mat_indices[env_ids], 0:3] = self.mat_start_pos[env_ids, 0:3] + noise_mat
                table_mat_indices = torch.cat(
                    [self.table_indices[env_ids], self.mat_indices[env_ids]], dim=0
                ).to(torch.int32)
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_state_tensor),
                    gymtorch.unwrap_tensor(table_mat_indices),
                    len(table_mat_indices),
                )
                self.gym.simulate(self.sim)

            
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            for env_id in env_ids:
                for t in self.rgb_tensors:
                    t[env_id] = 0.0
                if "depth" in self.render_data_type or "pcl" in self.render_data_type:
                    for t in self.depth_tensors:
                        t[env_id] = 0.0
                if "seg" in self.render_data_type:
                    for t in self.seg_tensors:
                        t[env_id] = 0
                #self.seg_tensor[env_id] = 0.0
            self.gym.end_access_image_tensors(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.object_init_states[env_ids] = self.root_state_tensor[self.object_indices[env_ids]].clone()
        self.compute_observations()
        self.obs_dict["obs"] = self.obs_buf.to(self.rl_device)
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.object_init_states[env_ids] = self.root_state_tensor[self.object_indices[env_ids]].clone()
        self.cur_ee_targets = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7].clone()
        
        #print(self.instructions)

        self.generate_reaching_plan_idx(env_ids)
        if self.obs_type == "eefpose+objinitpose+pcfeat":
            self.obs_objinitpose = self.obs_dict["obs"][:, 7:14]
            self.obs_eefpose = self.obs_dict["obs"][:, 0:7]
        return self.obs_dict

    # at the beginning of each episode, interpolate between initial ee pose and the first tracking target
    def generate_reaching_plan_idx(self, env_ids, actions=None):
        #print(actions)
        def get_random_value(interval:slice):
            if actions is not None:
                assert actions.shape[-1] >= interval.stop, f"action shape: {actions.shape}, interval: {interval}"
                return actions[env_ids, interval].to(self.device)
            else:
                return torch_rand_float(-1, 1, (self.num_envs, interval.stop-interval.start), device=self.device)
            
        if not self.randomize_tracking_reference:
            self.current_tracking_reference = self.tracking_reference
        else:
            self.current_tracking_reference = deepcopy(self.tracking_reference)
            ### randomize the wrist reference trajectory in the init object coordinate by left-multiplying a random transformation
            # sample rotation, 旋转矩阵左乘demo
            rand_N_3 = get_random_value(slice(3,6))
            rand_quat = quat_from_euler_xyz(
                rand_N_3[:, 0] * self.randomize_tracking_reference_range[3],
                rand_N_3[:, 1] * self.randomize_tracking_reference_range[4],
                rand_N_3[:, 2] * self.randomize_tracking_reference_range[5]
            ).unsqueeze(1).expand(-1, self.T_ref, -1)
            self.current_tracking_reference["wrist_quat"] = quat_mul(rand_quat, self.current_tracking_reference["wrist_quat"])
            self.current_tracking_reference["wrist_initobj_pos"] = quat_apply(rand_quat, self.current_tracking_reference["wrist_initobj_pos"])            
            # sample xyz offset
            rand_N_3 = get_random_value(slice(0,3))
            self.current_tracking_reference["wrist_initobj_pos"] += \
                (rand_N_3 * self.randomize_tracking_reference_range[0:3]).unsqueeze(1).expand(-1, self.T_ref, -1)
            # keep the lifting motion same to the demo
            self.current_tracking_reference["wrist_initobj_pos"][:, self.T_ref_start_lifting:, 0:3] = \
                self.tracking_reference["wrist_initobj_pos"][:, self.T_ref_start_lifting:, 0:3] \
                - self.tracking_reference["wrist_initobj_pos"][:, self.T_ref_start_lifting-1:self.T_ref_start_lifting, 0:3] \
                + self.current_tracking_reference["wrist_initobj_pos"][:, self.T_ref_start_lifting-1: self.T_ref_start_lifting, 0:3]
            # modify grasping pose
            if self.randomize_grasp_pose:
                rand_N_hand = get_random_value(slice(6,6+self.num_active_hand_dofs))
                rand_grasp_pose = self.current_tracking_reference["hand_qpos"][:, self.T_ref_start_lifting-1] + \
                    rand_N_hand * self.randomize_grasp_pose_range
                rand_grasp_pose = torch.clamp(
                    rand_grasp_pose,
                    self.robot_dof_lower_limits[self.active_hand_dof_indices],
                    self.robot_dof_upper_limits[self.active_hand_dof_indices]
                ) # (num_envs, dim)
                hand_ref_seq = self.current_tracking_reference["hand_qpos"] # (num_envs, T_ref, dim)
                hand_ref_seq_t0 = hand_ref_seq[:, 0].unsqueeze(1).repeat(1, self.T_ref_start_lifting-1, 1)
                fraction = (rand_grasp_pose - hand_ref_seq[:, 0]) /\
                    (hand_ref_seq[:, self.T_ref_start_lifting-1] - hand_ref_seq[:, 0] + 1e-6) # (num_envs, dim): (q_grasp' - q0) / (q_grasp - q0)
                self.current_tracking_reference["hand_qpos"][:, 0:self.T_ref_start_lifting-1] = \
                    hand_ref_seq_t0 + \
                    (hand_ref_seq[:, 0:self.T_ref_start_lifting-1] - hand_ref_seq_t0) * \
                    fraction.unsqueeze(1).repeat(1, self.T_ref_start_lifting-1, 1) # pregrasp interpolation: q'(t) = q0 + (q(t)-q0)*fraction 
                self.current_tracking_reference["hand_qpos"][:, self.T_ref_start_lifting-1:] = \
                    rand_grasp_pose.unsqueeze(1).repeat(1, self.T_ref - self.T_ref_start_lifting + 1, 1) # grasp and lift: keep the grasp pose
                self.current_tracking_reference["hand_qpos"] = torch.clamp(
                    self.current_tracking_reference["hand_qpos"],
                    self.robot_dof_lower_limits[self.active_hand_dof_indices],
                    self.robot_dof_upper_limits[self.active_hand_dof_indices]
                )
                #print("Randomized grasp pose diff:", (self.current_tracking_reference["hand_qpos"] - self.tracking_reference["hand_qpos"]).abs().max())


        wrist_pose = self.rigid_body_states.view(-1, 13)[self.eef_idx[env_ids], 0:7] # [B, 7]
        wrist_pose_target = torch.cat([
            self.current_tracking_reference["wrist_initobj_pos"][env_ids, 0] + self.object_init_states[env_ids,0:3],
            self.current_tracking_reference["wrist_quat"][env_ids, 0]
        ], dim=-1) # [B, 7]

        reaching_plan_ee, reaching_plan_timesteps = batch_linear_interpolate_poses(
            wrist_pose, wrist_pose_target, 
            max_trans_step=0.04 * self.cfg["env"]["interpolationStepScale"], 
            max_rot_step=0.1 * self.cfg["env"]["interpolationStepScale"]
        ) # [B, T, 7]
        reaching_plan_ee = reaching_plan_ee[:, 1:min(self.max_episode_length, reaching_plan_ee.shape[1])] # 去头,不超出最大长度
        reaching_plan_timesteps -= 1
        self.reaching_plan_ee[env_ids, :reaching_plan_ee.shape[1]] = reaching_plan_ee
        self.reaching_plan_timesteps[env_ids] = reaching_plan_timesteps
        #print("Reaching plan:", reaching_plan_ee)
        #print(self.reaching_plan_timesteps[env_ids].float().mean(), self.reaching_plan_timesteps[env_ids].float().std(), self.T_ref)

    # planning-based: compute reference action for reaching + tracking
    def compute_reference_actions(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        reaching_plan_timestep_ids = torch.minimum(self.progress_buf, self.reaching_plan_timesteps)
        wrist_pose_reaching_target = self.reaching_plan_ee[env_ids, reaching_plan_timestep_ids] * \
            (self.progress_buf < self.reaching_plan_timesteps).unsqueeze(-1).float()
        
        tracking_timestep_ids = (self.progress_buf - self.reaching_plan_timesteps).clamp(min=0, max=self.T_ref-1)
        wrist_pose_tracking_target = torch.cat([
            self.current_tracking_reference["wrist_initobj_pos"][env_ids, tracking_timestep_ids] \
            + self.object_init_states[:, 0:3],
            self.current_tracking_reference["wrist_quat"][env_ids, tracking_timestep_ids]
        ], dim=-1) * (self.progress_buf >= self.reaching_plan_timesteps).unsqueeze(-1).float() # [B, 7]
        wrist_pose_target = wrist_pose_reaching_target + wrist_pose_tracking_target
        #print("    Wrist pose target:", wrist_pose_target, reaching_plan_timestep_ids)
        hand_qpos_target = self.current_tracking_reference["hand_qpos"][env_ids, tracking_timestep_ids]

        if self.cfg["env"]["limitControlError"]:
            hand_qpos_target = tensor_clamp(
                hand_qpos_target,
                self.robot_dof_pos[:, self.active_hand_dof_indices] - self.cfg["env"]["maxPDErrorHand"],
                self.robot_dof_pos[:, self.active_hand_dof_indices] + self.cfg["env"]["maxPDErrorHand"],
            )
            wrist_pose_target[:, 0:3] = tensor_clamp(
                wrist_pose_target[:, 0:3],
                self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:3] - self.cfg["env"]["maxPDErrorEEPos"],
                self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:3] + self.cfg["env"]["maxPDErrorEEPos"],
            )

        if self.arm_controller == "qpos" and not self.use_relative_control:
            arm_qpos_target = self.compute_arm_ik(
                action=wrist_pose_target, 
                is_delta_pose=False,
            ) + self.robot_dof_pos[:, self.arm_dof_indices]                
            qpos_target = torch.cat([arm_qpos_target, hand_qpos_target], dim=-1)
            action = unscale(
                qpos_target,
                self.robot_dof_lower_limits[self.active_robot_dof_indices],
                self.robot_dof_upper_limits[self.active_robot_dof_indices],
            )
        elif "pose" in self.arm_controller:
            action = self.actions.clone()
            wrist_pose = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7]
            if self.arm_controller == "worlddpose":
                dquat = quat_mul(
                    wrist_pose_target[:, 3:], 
                    quat_conjugate(wrist_pose[:, 3:])
                )
                dangle, daxis = quat_to_angle_axis(dquat)
                dangleaxis = dangle.unsqueeze(-1) * daxis
                action[:, :3] = wrist_pose_target[:, :3] - wrist_pose[:, :3] # set first-3 arm actions to dpos
                action[:, 3:6] = dangleaxis # set next-3 arm actions to dangleaxis
                action[:, 6] = 0.0
            elif self.arm_controller == "eedpose":
                dquat = quat_mul(
                    quat_conjugate(quat_unit(wrist_pose[:, 3:])), 
                    quat_unit(wrist_pose_target[:, 3:])
                )
                dangle, daxis = quat_to_angle_axis(quat_unit(dquat))
                dangle = torch.clamp(dangle, 0., 0.3) # ensure stability
                dangleaxis = dangle.unsqueeze(-1) * daxis
                action[:, :3] = quat_apply(
                    quat_conjugate(wrist_pose[:, 3:]), 
                    wrist_pose_target[:, :3] - wrist_pose[:, :3]
                ) # set first-3 arm actions to local dpos
                action[:, 3:6] = dangleaxis # set next-3 arm actions to local dangleaxis
                action[:, 6] = 0.0
                #print(dangle.max(), dangle.min(), dangle.mean(), dangle.std())
            elif self.arm_controller == "pose":
                action[:, :7] = wrist_pose_target # absolute ee pose
            else:
                raise NotImplementedError
            action[:, self.hand_dof_start_idx:] = unscale(
                hand_qpos_target,
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
        else:
            raise NotImplementedError
        #print(action[:, 0:7].abs().max())
        return action
    

    def step(self, actions):
        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # do linear interpolation to perform position control
        for t_ in range(self.decimation):
            target = self.prev_targets + (t_+1) / self.decimation * (self.cur_targets - self.prev_targets)
            
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(target)
            )
            # step physics and render each frame
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
        
        self.prev_targets[:, self.robot_dof_indices] = self.cur_targets[
            :, self.robot_dof_indices
        ]

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.obs_dict["obs"] = self.obs_buf.to(self.rl_device)
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    def pre_physics_step(self, actions):
        #print(actions)
        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            # last qpos + delta action
            self.cur_targets[:, self.active_hand_dof_indices] = \
                self.prev_targets[:, self.active_hand_dof_indices] + \
                    self.actions[:, self.hand_dof_start_idx:] * self.delta_action_scale[self.hand_dof_start_idx:]
        else:
            # [-1,1] action -> target qpos
            self.cur_targets[:, self.active_hand_dof_indices] = scale(
                self.actions[:, self.hand_dof_start_idx:],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )     
        # clip to satisfy max step size
        self.cur_targets[:, self.active_hand_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.active_hand_dof_indices],
            self.prev_targets[:, self.active_hand_dof_indices]
            - self.act_max_ang_vel_hand * self.dt * self.decimation,
            self.prev_targets[:, self.active_hand_dof_indices]
            + self.act_max_ang_vel_hand * self.dt * self.decimation,
        )
        # set passive joints
        if self.have_passive_joints:
            self.cur_targets[:, self.passive_hand_dof_indices] = \
                self.cur_targets[:, self.mimic_parent_dof_indices] * self.mimic_multipliers
        # clip to joint limits
        self.cur_targets[:, self.hand_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.hand_dof_indices],
            self.robot_dof_lower_limits[self.hand_dof_indices],
            self.robot_dof_upper_limits[self.hand_dof_indices],
        )

        if self.arm_controller == "qpos":
            if self.use_relative_control:
                print("Warning: Currently, relative control for arm is implemented as direct qpos copy!!!")
                self.cur_targets[:, self.arm_dof_indices] = scale(
                    self.actions[:, :self.hand_dof_start_idx],
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
                ## last qpos + delta action
                #self.cur_targets[:, self.arm_dof_indices] = self.prev_targets[:, self.arm_dof_indices] + self.actions[:, :self.hand_dof_start_idx]
            else:
                # [-1,1] action -> target qpos
                self.cur_targets[:, self.arm_dof_indices] = scale(
                    self.actions[:, :self.num_arm_dofs],
                    self.robot_dof_lower_limits[self.arm_dof_indices],
                    self.robot_dof_upper_limits[self.arm_dof_indices],
                )
            
            # clip to satisfy max step size
            self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.arm_dof_indices],
                self.prev_targets[:, self.arm_dof_indices]
                - self.act_max_ang_vel_arm * self.dt * self.decimation,
                self.prev_targets[:, self.arm_dof_indices]
                + self.act_max_ang_vel_arm * self.dt * self.decimation,
            )
            # clip to joint limits
            self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
        elif "pose" in self.arm_controller:
            if self.arm_controller == "pose":
                # absolute pose control
                delta_arm_action = self.compute_arm_ik(self.actions[:, :7], is_delta_pose=False)
            else:
                # delta pose control
                delta_arm_action = self.compute_arm_ik(self.actions[:, :6] * self.delta_action_scale[:6], is_delta_pose=True, is_delta_pose_in_world=("world" in self.arm_controller))
            self.cur_targets[:, self.arm_dof_indices] = self.robot_dof_pos[:, self.arm_dof_indices] + delta_arm_action
            # clip to satisfy max step size
            self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.arm_dof_indices],
                self.prev_targets[:, self.arm_dof_indices]
                - self.act_max_ang_vel_arm * self.dt * self.decimation,
                self.prev_targets[:, self.arm_dof_indices]
                + self.act_max_ang_vel_arm * self.dt * self.decimation,
            )
            # clip to joint limits
            self.cur_targets[:, self.arm_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.compute_observations()
        self.compute_reward()


    def compute_observations(self):
        if self.use_camera:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.rgb_tensors = [torch.stack([i for i in self.camera_rgb_tensor_lists[j]], dim=0) for j in range(len(self.camera_ids))]
            if "depth" in self.render_data_type or "pcl" in self.render_data_type: 
                self.depth_tensors = [-torch.stack([i for i in self.camera_depth_tensor_lists[j]], dim=0) for j in range(len(self.camera_ids))]
            if "seg" in self.render_data_type:
                self.seg_tensors = [torch.stack([i for i in self.camera_seg_tensor_lists[j]], dim=0) for j in range(len(self.camera_ids))]
            self.gym.end_access_image_tensors(self.sim)
            if "depth" in self.render_data_type or "pcl" in self.render_data_type:
                for i in range(len(self.camera_ids)):
                    self.depth_tensors[i] = torch.where(
                        torch.logical_and(self.depth_tensors[i] >= self.depth_ranges[i][0], self.depth_tensors[i] <= self.depth_ranges[i][1]),
                        self.depth_tensors[i],
                        torch.zeros_like(self.depth_tensors[i], device=self.device),
                    )

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        if self.use_distractor_objects:
            self.distractor_object_pos = self.root_state_tensor[self.distractor_object_indices.reshape(-1), 0:3]

        self.palm_state = self.rigid_body_states[:, self.palm_handle][..., :13]
        self.palm_pos = self.palm_state[..., :3]
        self.palm_rot = self.palm_state[..., 3:7]
        self.palm_center_pos = self.palm_pos + quat_apply(
            self.palm_rot, to_torch(self.palm_offset).repeat(self.num_envs, 1)
        )
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][
            ..., :13
        ]
        self.fingertip_pos = self.fingertip_state[..., :3]
        self.fingertip_rot = self.fingertip_state[..., 3:7]

        self.compute_required_observations(self.obs_buf, self.obs_type, self.num_observations)
        
    # compute obs with required contents
    def compute_required_observations(self, obs_buf, obs_type, num_obs):
        obs_end = 0

        if "armdof" in obs_type:
            obs_buf[:, obs_end: obs_end+self.num_arm_dofs] = unscale(
                self.robot_dof_pos[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
            obs_end += self.num_arm_dofs

        if "handdof" in obs_type:
            obs_buf[:, obs_end: obs_end+self.num_active_hand_dofs] = unscale(
                self.robot_dof_pos[:, self.active_hand_dof_indices],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
            obs_end += self.num_active_hand_dofs
        
        if "fulldof" in obs_type:
            obs_buf[:, obs_end: obs_end+self.num_robot_dofs] = unscale(
                self.robot_dof_pos[:, self.robot_dof_indices],
                self.robot_dof_lower_limits[self.robot_dof_indices],
                self.robot_dof_upper_limits[self.robot_dof_indices],
            )
            obs_end += self.num_robot_dofs

        if "eefpose" in obs_type:
            obs_buf[:, obs_end: obs_end+7] = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7]
            obs_end += 7

        if "ftpos" in obs_type: # fingertip positions, N*3
            num_ft_states = self.num_fingers * 3
            obs_buf[:, obs_end: obs_end+num_ft_states] = (
                self.fingertip_pos.reshape(self.num_envs, num_ft_states)
            )
            obs_end += num_ft_states
                
        if "palmpose" in obs_type: # palm pose, N*7
            obs_buf[:, obs_end: obs_end+3] = self.palm_pos
            obs_buf[:, obs_end+3: obs_end+7] = self.palm_rot
            obs_end += 7
        
        if "lastact" in obs_type: # last action
            obs_buf[:, obs_end : obs_end+self.num_actions] = self.actions
            obs_end += self.num_actions
        
        if "objxyz" in obs_type: # object xyz position, N*3
            obs_buf[:, obs_end: obs_end+3] = self.object_pos
            obs_end += 3

        if "objpose" in obs_type: # object pose: pos, rot (7)
            obs_buf[:, obs_end: obs_end+7] = self.object_pose
            obs_end += 7
        
        if "objinitpose" in obs_type:
            obs_buf[:, obs_end: obs_end+7] = self.object_init_states[:, 0:7]
            obs_end += 7
        
        if "objpcl" in obs_type: # object point cloud
            self.transformed_pcl = self.transform_obj_pcl_2_world()
            obs_buf[:, obs_end: obs_end+self.points_per_object*3] = self.transformed_pcl.reshape(self.num_envs,-1)
            obs_end += self.points_per_object*3
        
        assert obs_end == num_obs


    # transform the object pcl within the world coordinate
    def transform_obj_pcl_2_world(self):
        o2w_pos = self.object_pos.clone()
        o2w_pos = o2w_pos.resize(self.num_envs,1,3)
        o2w_quat = self.object_rot.clone()
        o2w_quat = o2w_quat.resize(self.num_envs,1,4)

        append_pos = torch.zeros([self.num_envs, self.points_per_object, 1]).to(self.device)
        pcl = torch.cat([self.obj_pcl_buf, append_pos], 2) # (num_envs, points_per_object, 4)

        o2w_quat = o2w_quat.expand_as(pcl)
        pcl = transform_points(o2w_quat, pcl)
        o2w_pos = o2w_pos.expand_as(pcl)
        pcl = pcl + o2w_pos
        return pcl


    ### convert (delta) target ee pose to delta joint angles of the arm
    def compute_arm_ik(self, action, is_delta_pose=True, is_delta_pose_in_world=True, reference_state=None):
        '''
        action: either 6-dim pos+angle-axis for delta pose, or 7-dim pos+quat for absolute target pose
        is_delta_pose: delta pose or absolute target pose?
        is_delta_pose_in_world: delta pose is in the world frame or in the current end-effector frame?
        reference_state: current end-effector state by default
        '''
        if reference_state is None:
            reference_state = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7]

        # delta action: 3 dim delta position + 3 dim delta angle-axis
        if is_delta_pose:
            delta_action = action
            if is_delta_pose_in_world:
                # delta pose defined in the world frame
                pos_err = delta_action[:, 0:3]
                dtheta = torch.norm(delta_action[:, 3:6], dim=-1, keepdim=True)
                axis = delta_action[:, 3:6] / (dtheta + 1e-4)
                delta_quat = quat_from_angle_axis(dtheta.squeeze().view(-1), axis)
                orn_err = orientation_error(
                    quat_mul(delta_quat, reference_state[:, 3:7]),
                    reference_state[:, 3:7],
                )
            else:
                # delta pose defined in the end-effector frame
                pos_err = quat_apply(reference_state[:, 3:7], delta_action[:, 0:3])
                dtheta = torch.norm(delta_action[:, 3:6], dim=-1, keepdim=True)
                axis = delta_action[:, 3:6] / (dtheta + 1e-4)
                delta_quat = quat_from_angle_axis(dtheta.squeeze().view(-1), axis)
                orn_err = orientation_error(
                    quat_mul(reference_state[:, 3:7], delta_quat),
                    reference_state[:, 3:7],
                )
        # absolute target action: 3 dim position + 4 dim quat
        else:
            pos_err = action[:, 0:3] - reference_state[:, 0:3]
            orn_err = orientation_error(quat_unit(action[:, 3:7]), reference_state[:, 3:7])
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        u = self._control_ik(dpose) # the input dpose of _control_ik is always in the world (base) frame
        return u    
    
    def _control_ik(self, dpose):
        damping = 0.1
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        #print(j_eef_T.shape)
        lmbda = torch.eye(6, device=self.device) * (damping**2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(
            self.num_envs, self.num_arm_dofs
        )
        return u

    def compute_reward(self):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.current_successes[:],
            self.has_hit_table[:],
            reward_info,
        ) = self.reward_function(
            reset_buf = self.reset_buf,
            progress_buf = self.progress_buf,
            successes = self.successes,
            current_successes = self.current_successes,
            has_hit_table = self.has_hit_table,
            max_episode_length = self.max_episode_length,
            table_heights = self.table_heights,
            object_pos = self.object_pos,
            #goal_height = self.goal_height,
            palm_pos = self.palm_center_pos,
            fingertip_pos = self.fingertip_pos,
            num_fingers = self.num_fingers,
            object_init_states = self.object_init_states,
            #tracking_timestep = self.tracking_timestep,
            end_effector_pose = self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7],
            hand_qpos = self.robot_dof_pos[:, self.active_hand_dof_indices],
        )

        self.extras.update(reward_info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["has_hit_table"] = self.has_hit_table
        #print(self.has_hit_table)
    

    # get obs same to the real-world
    def compute_real_observation_dict(self):
        obs_dict = {
            "instruction": self.instructions, # [num_envs] str 
            "right_arm_qpos": self.robot_dof_pos[:, self.arm_dof_indices].cpu().numpy(), # [num_envs, 7]
            "right_arm_eef_pose": self.rigid_body_states.view(-1, 13)[self.eef_idx, 0:7].cpu().numpy(), # [num_envs, 7]
            "right_hand_qpos": self.robot_dof_pos[:, self.active_hand_dof_indices].cpu().numpy(), # [num_envs, 6]
        }

        for i, cam_id in enumerate(self.camera_ids):
            rgb_image = self.rgb_tensors[i][..., :3]
            if "rgb" in self.render_data_type:
                rgb = F.interpolate(
                    rgb_image.permute(0, 3, 1, 2).to(torch.float32),
                    size=tuple(self.render_cfg["resize"]),
                    mode='area'
                ).permute(0,2,3,1) # (num_envs, 256, 256, 3)
                obs_dict[f"camera_{cam_id}.rgb"] = rgb.cpu().numpy().astype(np.uint8)
            if "depth" in self.render_data_type:
                raise NotImplementedError
            if "pcl" in self.render_data_type:
                raise NotImplementedError

        return obs_dict


@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def linear_interpolate_poses(
    pose1: torch.Tensor,  # Shape: [7] (x, y, z, qx, qy, qz, qw)
    pose2: torch.Tensor,  # Shape: [7]
    n_steps: int, # Number of interpolation steps
) -> torch.Tensor:
    # Split into position and quaternion
    p1, q1 = pose1[..., :3], pose1[..., 3:]
    p2, q2 = pose2[..., :3], pose2[..., 3:]
    
    # Generate interpolation steps
    t = torch.linspace(0, 1, n_steps + 1, device=pose1.device)
    
    # Linear interpolation for positions [n_steps+1, B, 3]
    interp_p = p1.unsqueeze(0) + t.view(-1, 1, 1) * (p2.unsqueeze(0) - p1.unsqueeze(0))
    
    # Interpolate rotations (SLERP)
    interp_q = torch.stack([
        slerp(q1, q2, ti.unsqueeze(0))
        for ti in t
    ]).view(n_steps+1, -1, 4)

    interpolated_poses = torch.cat([interp_p, interp_q], dim=-1)
    
    return interpolated_poses[1:]
