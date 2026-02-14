# FR3 + Inspire hand
python run_rl_grasp.py task=grasp train=PPOOneStep test=True num_envs=175 \
    task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
    task.env.trackingReferenceFile=tasks/grasp_ref_inspire.pkl task.env.trackingReferenceLiftTimestep=13 \
    task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True checkpoint='ckpt/inspire.pt'

# Shadow hand
python run_rl_grasp.py task=grasp train=PPOOneStep hand=shadow_simple test=True \
    num_envs=175 task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True task.env.resetDofPosRandomInterval=0 \
    task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
    task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl task.env.trackingReferenceLiftTimestep=11 \
    checkpoint='ckpt/shadow.pt'

# FR3 + Shadow hand
python run_rl_grasp.py task=grasp train=PPOOneStep hand=fr3_shadow test=True \
    num_envs=175 task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True task.env.resetDofPosRandomInterval=0 \
    task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
    task.env.trackingReferenceFile=tasks/grasp_ref_shadow.pkl task.env.trackingReferenceLiftTimestep=11 checkpoint='ckpt/fr3_shadow.pt'

# UR5 + Allegro hand
python run_rl_grasp.py task=grasp train=PPOOneStep hand=ur5_allegro test=True num_envs=175 \
    task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True task.env.trackingReferenceFile=tasks/grasp_ref_allegro.pkl \
    task.env.trackingReferenceLiftTimestep=11 task.env.randomizeGraspPose=True task.env.resetDofPosRandomInterval=0 \
    task.env.episodeLength=50 task.env.enablePointCloud=True \
    train.params.is_vision=True checkpoint='ckpt/ur5_allegro.pt'

# UR5 + Schunk hand
python run_rl_grasp.py task=grasp train=PPOOneStep hand=ur5_svh test=True num_envs=175 \
    task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True task.env.trackingReferenceFile=tasks/grasp_ref_svh.pkl \
    task.env.trackingReferenceLiftTimestep=11 task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0 task.env.episodeLength=50 task.env.enablePointCloud=True \
    train.params.is_vision=True checkpoint='ckpt/ur5_svh.pt'

# FR3 + gripper
python run_rl_grasp.py task=grasp train=PPOOneStep hand=fr3_panda_gripper test=True num_envs=175 \
    task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True task.env.trackingReferenceFile=tasks/grasp_ref_panda_gripper.pkl \
    task.env.trackingReferenceLiftTimestep=11 task.env.resetDofPosRandomInterval=0 \
    task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
    task.env.resetPositionRange="[[0.4, 0.7], [-0.35, 0.15], [0.1, 0.12]]" \
    checkpoint='ckpt/fr3_panda_gripper.pt'

# FR3 + DClaw gripper
python run_rl_grasp.py task=grasp train=PPOOneStep hand=fr3_dclaw_gripper test=True \
    num_envs=175 task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
    task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
    task.env.randomizeTrackingReference=True task.env.trackingReferenceFile=tasks/grasp_ref_dclaw_gripper.pkl \
    task.env.trackingReferenceLiftTimestep=11 task.env.randomizeGraspPose=True \
    task.env.resetDofPosRandomInterval=0 task.env.episodeLength=50 task.env.enablePointCloud=True \
    train.params.is_vision=True checkpoint='ckpt/fr3_dclaw.pt'
