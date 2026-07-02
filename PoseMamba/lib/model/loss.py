import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_SKELETON_NAMES, KEYPOINT_INDEX

    _LIMB_IDS = [[KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]] for a, b in BICYCLE_SKELETON_NAMES]
except Exception:
    _LIMB_IDS = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16],
    ]

# Numpy-based errors

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    # print(predicted.shape, target.shape)
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # print(predicted.shape, w.shape)
    assert w.shape[0] == predicted.shape[2] #torch.Size([24, 243, 17, 3]) torch.Size([17])
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs = x[:, :, _LIMB_IDS, :]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var

def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt) # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)

def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = _LIMB_IDS
    angle_id = [[i, i + 1] for i in range(max(0, len(limbs_id) - 1))]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)


# ---------------------------------------------------------------------------
# Bicycle-specific dynamics losses (steering angle, roll angle).
#
# Both quantities are derived directly from the predicted / target 3D keypoint
# tensor and compared in radians via an L1 loss with proper ±pi wrap-around.
# This makes the loss self-consistent: pred and gt steer/roll are produced by
# the same geometric formula, so a zero loss is achieved iff the predicted
# keypoints reproduce the target's bicycle pose up to the angles we care
# about. MuJoCo `dynamics_gt` stored in pickles is intentionally not used here
# (it lives in a different convention / world frame); it stays available for
# validation and offline metrics.
# ---------------------------------------------------------------------------

# Bicycle keypoint index aliases (must match BICYCLE_KEYPOINT_NAMES).
try:
    _KP_BB = KEYPOINT_INDEX["k_bottom_bracket"]
    _KP_SEAT_STAY = KEYPOINT_INDEX["k_seat_stay"]
    _KP_SADDLE = KEYPOINT_INDEX["k_saddle"]
    _KP_UHT = KEYPOINT_INDEX["k_upper_head_tube"]
    _KP_LHT = KEYPOINT_INDEX["k_lower_head_tube"]
    _KP_HB_L = KEYPOINT_INDEX["k_handlebar_left"]
    _KP_HB_MID = KEYPOINT_INDEX["k_handlebar_middle"]
    _KP_HB_R = KEYPOINT_INDEX["k_handlebar_right"]
    _KP_FH_L = KEYPOINT_INDEX["k_front_hub_left"]
    _KP_FH_R = KEYPOINT_INDEX["k_front_hub_right"]
    _KP_FW_BACK = KEYPOINT_INDEX["k_front_wheel_back"]
    _KP_FW_FRONT = KEYPOINT_INDEX["k_front_wheel_front"]
    _KP_RH_L = KEYPOINT_INDEX["k_rear_hub_left"]
    _KP_RH_R = KEYPOINT_INDEX["k_rear_hub_right"]
    _KP_RW_GND = KEYPOINT_INDEX["k_rear_wheel_ground"]
    # Frame-rigid midline keypoints used to fit the sagittal plane.
    # Front-wheel keypoints are excluded because they rotate out of the plane
    # whenever the fork is steered.
    _SAGITTAL_PLANE_IDS = [
        _KP_BB,
        _KP_SEAT_STAY,
        _KP_SADDLE,
        _KP_UHT,
        _KP_LHT,
        _KP_HB_MID,
        _KP_RW_GND,
    ]
except Exception:
    _KP_BB = 0
    _KP_SEAT_STAY = 1
    _KP_SADDLE = 2
    _KP_UHT = 3
    _KP_LHT = 4
    _KP_HB_L = 5
    _KP_HB_MID = 6
    _KP_HB_R = 7
    _KP_FH_L = 8
    _KP_FH_R = 9
    _KP_FW_BACK = 10
    _KP_FW_FRONT = 11
    _KP_RH_L = 13
    _KP_RH_R = 14
    _KP_RW_GND = 15
    _SAGITTAL_PLANE_IDS = [
        _KP_BB, _KP_SEAT_STAY, _KP_SADDLE, _KP_UHT, _KP_LHT, _KP_HB_MID, _KP_RW_GND,
    ]


_DYN_EPS = 1e-6


def _safe_normalize(v, eps=_DYN_EPS):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


def _project_perpendicular_to_axis(v, axis):
    return v - (v * axis).sum(dim=-1, keepdim=True) * axis


def _signed_oriented(vec, ref):
    """Flip ``vec`` so that it points in the same half-space as ``ref``.

    Both inputs are (..., 3). The flip is piecewise constant w.r.t. ``vec`` so
    gradients flow normally except on a measure-zero boundary.
    """
    dot = (vec * ref).sum(dim=-1, keepdim=True)
    sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
    return vec * sign


def _sagittal_plane_normal(kpts):
    """Fit a plane to the bicycle's midline keypoints and return its unit normal.

    Args:
        kpts: tensor of shape (..., J, 3) in any consistent frame.

    Returns:
        Tensor of shape (..., 3). The normal's sign is aligned with the rear
        hub left -> right direction so that "positive lateral" is well defined
        across batches.
    """
    pts = kpts[..., _SAGITTAL_PLANE_IDS, :]  # (..., K, 3)
    centered = pts - pts.mean(dim=-2, keepdim=True)
    cov = centered.transpose(-1, -2) @ centered  # (..., 3, 3)
    # eigh is more stable than svd for symmetric PSD matrices and has
    # well-defined gradients here (the smallest eigenvalue is well separated
    # from the others as long as the plane fit is meaningful).
    _, eigvecs = torch.linalg.eigh(cov)
    n = eigvecs[..., :, 0]  # eigenvector for smallest eigenvalue = plane normal
    ref = kpts[..., _KP_RH_R, :] - kpts[..., _KP_RH_L, :]
    return _safe_normalize(_signed_oriented(n, ref))


def _signed_angle_about_axis(ref, obs, axis):
    """Signed angle from ``ref`` to ``obs`` after projection perpendicular to axis."""
    a = _project_perpendicular_to_axis(ref, axis)
    b = _project_perpendicular_to_axis(obs, axis)
    sin_part = (torch.cross(a, b, dim=-1) * axis).sum(dim=-1)
    cos_part = (a * b).sum(dim=-1)
    angle = torch.atan2(sin_part, cos_part)
    weight = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)
    return angle, weight


def _weighted_circular_mean(angles, weights):
    angles = torch.stack(angles, dim=0)
    weights = torch.stack(weights, dim=0).clamp_min(_DYN_EPS)
    sin_sum = (weights * torch.sin(angles)).sum(dim=0)
    cos_sum = (weights * torch.cos(angles)).sum(dim=0)
    return torch.atan2(sin_sum, cos_sum)


def bicycle_steer_angle_hub(kpts):
    """Signed steering angle from only rear-hub and front-hub transverse axes.

    This is the rake-independent, geometrically exact definition: project the
    frame's transverse axis (rear hub L->R) and the fork's transverse axis
    (front hub L->R) into the plane perpendicular to the head-tube axis, and
    measure the signed rotation between them via ``atan2``.

    Args:
        kpts: tensor of shape (..., J, 3).

    Returns:
        Tensor of shape (...,) with the signed steer angle in radians.
    """
    rh_l = kpts[..., _KP_RH_L, :]
    rh_r = kpts[..., _KP_RH_R, :]
    fh_l = kpts[..., _KP_FH_L, :]
    fh_r = kpts[..., _KP_FH_R, :]
    ht_u = kpts[..., _KP_UHT, :]
    ht_l = kpts[..., _KP_LHT, :]

    e = _safe_normalize(ht_u - ht_l)  # head-tube axis

    v_frame = rh_r - rh_l
    v_fork = fh_r - fh_l

    angle, _ = _signed_angle_about_axis(v_frame, v_fork, e)
    return angle


def bicycle_steer_angle(kpts):
    """Signed steering angle in radians, computed about the head-tube axis.

    The estimator is still rake-independent, but no longer relies on a single
    front-hub vector. It measures the rotation of several fork-rigid directions
    around the head-tube axis and combines them with a circular mean:

    * front-hub left->right axis vs rear-hub left->right axis,
    * handlebar left->right axis vs rear-hub left->right axis,
    * front-wheel back->front axis vs frame forward direction.

    These observables use the extra steering markers while avoiding the older
    frame-plane intersection method, whose line-orientation ambiguity made the
    predicted steer collapse toward a nearly flat signal.

    Args:
        kpts: tensor of shape (..., J, 3).

    Returns:
        Tensor of shape (...,) with the signed steer angle in radians.
    """
    rh_l = kpts[..., _KP_RH_L, :]
    rh_r = kpts[..., _KP_RH_R, :]
    fh_l = kpts[..., _KP_FH_L, :]
    fh_r = kpts[..., _KP_FH_R, :]
    hb_l = kpts[..., _KP_HB_L, :]
    hb_r = kpts[..., _KP_HB_R, :]
    fw_back = kpts[..., _KP_FW_BACK, :]
    fw_front = kpts[..., _KP_FW_FRONT, :]
    ht_u = kpts[..., _KP_UHT, :]
    ht_l = kpts[..., _KP_LHT, :]

    e = _safe_normalize(ht_u - ht_l)

    rear_center = 0.5 * (rh_l + rh_r)
    head_center = 0.5 * (ht_u + ht_l)
    frame_lateral = rh_r - rh_l
    frame_forward = head_center - rear_center

    angles = []
    weights = []

    # Transverse fork axes: same zero-angle convention as the hub-only method.
    angle, weight = _signed_angle_about_axis(frame_lateral, fh_r - fh_l, e)
    angles.append(angle)
    weights.append(weight)

    angle, weight = _signed_angle_about_axis(frame_lateral, hb_r - hb_l, e)
    angles.append(angle)
    # Handlebar keypoints are visually distinctive but farther from the wheel
    # axle, so give them slightly less influence than the hub axle itself.
    weights.append(0.75 * weight)

    # Longitudinal fork axis from wheel markers. This brings the front/back
    # wheel markers into the estimate without mixing transverse and longitudinal
    # directions in the same angle comparison.
    angle, weight = _signed_angle_about_axis(frame_forward, fw_front - fw_back, e)
    angles.append(angle)
    weights.append(0.5 * weight)

    return _weighted_circular_mean(angles, weights)


def bicycle_roll_angle(kpts):
    """Signed roll angle in radians, in the OpenCV camera frame.

    Defined as the tilt of the sagittal plane's normal away from the camera's
    horizontal plane (X-Z plane). Camera Y points down (OpenCV convention),
    so an upright bicycle has its sagittal normal lying in the X-Z plane and
    its roll is 0. Sign matches the direction of the bike's lean about its
    longitudinal axis.

    Args:
        kpts: tensor of shape (..., J, 3) in the (root-relative) camera frame.

    Returns:
        Tensor of shape (...,) with the signed roll angle in radians.
    """
    n = _sagittal_plane_normal(kpts)  # (..., 3)
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    # angle of n above the camera-horizontal plane (X-Z).
    # camera-up = -Y, so a normal that has a -Y component is tilted up.
    horiz = torch.sqrt(nx * nx + nz * nz + _DYN_EPS)
    return torch.atan2(-ny, horiz)


def _wrap_pi(angle):
    """Wrap a (signed) angle difference to (-pi, pi] via atan2(sin, cos)."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def loss_bicycle_steer(predicted, target):
    """L1 error on the steering angle, with proper ±pi wrap-around."""
    assert predicted.shape == target.shape
    s_pred = bicycle_steer_angle(predicted)
    s_gt = bicycle_steer_angle(target)
    return _wrap_pi(s_pred - s_gt).abs().mean()


def loss_bicycle_roll(predicted, target):
    """L1 error on the roll angle, with proper ±pi wrap-around."""
    assert predicted.shape == target.shape
    r_pred = bicycle_roll_angle(predicted)
    r_gt = bicycle_roll_angle(target)
    return _wrap_pi(r_pred - r_gt).abs().mean()


def loss_bicycle_steer_velocity(predicted, target):
    """L1 error on the steering-angle velocity (per-frame finite difference)."""
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.zeros((), device=predicted.device, dtype=predicted.dtype)
    s_pred = bicycle_steer_angle(predicted)
    s_gt = bicycle_steer_angle(target)
    dv_pred = _wrap_pi(s_pred[:, 1:] - s_pred[:, :-1])
    dv_gt = _wrap_pi(s_gt[:, 1:] - s_gt[:, :-1])
    return _wrap_pi(dv_pred - dv_gt).abs().mean()


def loss_bicycle_roll_velocity(predicted, target):
    """L1 error on the roll-angle velocity (per-frame finite difference)."""
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.zeros((), device=predicted.device, dtype=predicted.dtype)
    r_pred = bicycle_roll_angle(predicted)
    r_gt = bicycle_roll_angle(target)
    dv_pred = _wrap_pi(r_pred[:, 1:] - r_pred[:, :-1])
    dv_gt = _wrap_pi(r_gt[:, 1:] - r_gt[:, :-1])
    return _wrap_pi(dv_pred - dv_gt).abs().mean()

