# FR3 + Inspire hand
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=fr3_inspire_tac \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_inspire.pkl \
    task.env.trackingReferenceLiftTimestep=13 \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True \
    task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0.2 \
    task.env.observationType="eefpose+objinitpose+objpcl" \
    task.env.episodeLength=40 \
    task.env.enablePointCloud=True \
    train.params.is_vision=True \
    task.env.enableRobotTableCollision=False

# Shadow hand
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=shadow_simple \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl \
    task.env.trackingReferenceLiftTimestep=11 \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True \
    task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0 \
    task.env.observationType="eefpose+objinitpose+objpcl" \
    task.env.episodeLength=40 \
    task.env.enablePointCloud=True \
    train.params.is_vision=True \
    task.env.enableRobotTableCollision=False

# FR3 + Shadow hand
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=fr3_shadow \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl \
    task.env.trackingReferenceLiftTimestep=11 \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True \
    task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0 \
    task.env.observationType="eefpose+objinitpose+objpcl" \
    task.env.episodeLength=40 \
    task.env.enablePointCloud=True \
    train.params.is_vision=True \
    task.env.enableRobotTableCollision=False

# UR5 + Allegro hand
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=ur5_allegro \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_allegro.pkl \
    task.env.trackingReferenceLiftTimestep=11 \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True \
    task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0 \
    task.env.observationType="eefpose+objinitpose+objpcl" \
    task.env.episodeLength=40 \
    task.env.enablePointCloud=True \
    train.params.is_vision=True \
    task.env.enableRobotTableCollision=False

# UR5 + Schunk hand
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=ur5_svh \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_svh.pkl \
    task.env.trackingReferenceLiftTimestep=11 \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True \
    task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0 \
    task.env.observationType="eefpose+objinitpose+objpcl" \
    task.env.episodeLength=40 \
    task.env.enablePointCloud=True \
    train.params.is_vision=True \
    task.env.enableRobotTableCollision=False

# FR3 + gripper
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=fr3_panda_gripper \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_panda_gripper.pkl \
    task.env.trackingReferenceLiftTimestep=11 \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True \
    task.env.resetDofPosRandomInterval=0 \
    task.env.observationType="eefpose+objinitpose+objpcl" \
    task.env.episodeLength=40 \
    task.env.enablePointCloud=True \
    train.params.is_vision=True \
    task.env.resetPositionRange="[[0.4, 0.7], [-0.35, 0.15], [0.1, 0.12]]" \
    task.env.enableRobotTableCollision=False

# FR3 + DClaw gripper
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=fr3_dclaw_gripper \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_dclaw_gripper.pkl \
    task.env.trackingReferenceLiftTimestep=11 \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True \
    task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0 \
    task.env.observationType="eefpose+objinitpose+objpcl" \
    task.env.episodeLength=40 \
    task.env.enablePointCloud=True \
    train.params.is_vision=True \
    task.env.enableRobotTableCollision=False