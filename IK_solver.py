import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


class IKSolver:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._arm_dof_indices = None
        self._arm_joint_info = None
        self._init_arm_dofs()

    def _init_arm_dofs(self):
        """Collect DOF indices, qpos info, and joint limits for arm joints (shoulder flexion, abduction, elbow)."""
        arm_joint_names = [
            "left upper arm flexion",
            "left upper arm abduction",
            "left lower arm hinge",
            "right upper arm flexion",
            "right upper arm abduction",
            "right lower arm hinge",
        ]
        dof_indices = []
        joint_info = []  # (jid, jnt_type, qposadr, dofadr, ndof, limited, range_lo, range_hi)
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            jnt_type = self.model.jnt_type[jid]
            qposadr = self.model.jnt_qposadr[jid]
            dofadr = self.model.jnt_dofadr[jid]
            ndof = 3 if jnt_type == mujoco.mjtJoint.mjJNT_BALL else 1
            limited = bool(self.model.jnt_limited[jid])
            range_lo = float(self.model.jnt_range[jid, 0]) if limited else None
            range_hi = float(self.model.jnt_range[jid, 1]) if limited else None
            joint_info.append((jid, jnt_type, qposadr, dofadr, ndof, limited, range_lo, range_hi))
            dof_indices.extend(range(dofadr, dofadr + ndof))
        self._arm_dof_indices = np.array(dof_indices, dtype=np.int32)
        self._arm_joint_info = joint_info

    def solve_hands(
        self,
        max_iter=150,
        step_size=0.5,
        damping=1e-2,
        tol=1e-4,
    ):
        """
        Jacobian-based IK to move hand sites toward handlebar sites.
        Uses damped least-squares (Levenberg–Marquardt style) in joint velocity space.
        """
        left_hand_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left hand site")
        right_hand_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right hand site")
        left_handlebar_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left handlebar site")
        right_handlebar_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right handlebar site")

        nv = self.model.nv
        arm_dofs = self._arm_dof_indices
        n_arm = len(arm_dofs)

        # Preallocate Jacobian buffers (3 x nv each for position)
        jacp_left = np.zeros((3, nv), dtype=np.float64)
        jacp_right = np.zeros((3, nv), dtype=np.float64)

        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)

            # Current hand positions and handlebar targets (world frame)
            left_hand_pos = self.data.site_xpos[left_hand_site].copy()
            right_hand_pos = self.data.site_xpos[right_hand_site].copy()
            left_target = self.data.site_xpos[left_handlebar_site].copy()
            right_target = self.data.site_xpos[right_handlebar_site].copy()

            # Position error (6D)
            error = np.concatenate([left_target - left_hand_pos, right_target - right_hand_pos])

            if np.linalg.norm(error) < tol:
                break

            # Jacobians for both hand sites
            mujoco.mj_jacSite(self.model, self.data, jacp_left, None, left_hand_site)
            mujoco.mj_jacSite(self.model, self.data, jacp_right, None, right_hand_site)

            # Stack to 6 x nv and take arm columns -> 6 x n_arm
            J_full = np.vstack([jacp_left, jacp_right])
            J_arm = J_full[:, arm_dofs]

            # Damped least-squares: dq = (J'J + damping² I)^{-1} J' error
            JtJ = J_arm.T @ J_arm + (damping ** 2) * np.eye(n_arm)
            dq_arm = np.linalg.solve(JtJ, J_arm.T @ error)

            # Apply step in joint space
            self._apply_arm_dq(dq_arm, step_size)

        mujoco.mj_forward(self.model, self.data)

    def _apply_arm_dq(self, dq_arm, step_size):
        """Apply joint-space step dq_arm to qpos (only arm DOFs), enforcing limits."""
        i = 0
        for jid, jnt_type, qposadr, dofadr, ndof, limited, range_lo, range_hi in self._arm_joint_info:
            dq = step_size * dq_arm[i : i + ndof]
            i += ndof

            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                self.data.qpos[qposadr] += dq[0]
                if limited:
                    self.data.qpos[qposadr] = np.clip(
                        self.data.qpos[qposadr], range_lo, range_hi
                    )
            else:  # mjJNT_BALL
                # MuJoCo quat (w,x,y,z); scipy uses (x,y,z,w)
                quat = self.data.qpos[qposadr : qposadr + 4].copy()
                quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
                delta_rot = R.from_rotvec(dq)
                new_rot = R.from_quat(quat_scipy) * delta_rot
                new_quat = new_rot.as_quat()  # (x,y,z,w)
                self.data.qpos[qposadr] = new_quat[3]
                self.data.qpos[qposadr + 1 : qposadr + 4] = new_quat[:3]
                # Ball joints: no standard scalar range in MuJoCo; renormalize quat
                q = self.data.qpos[qposadr : qposadr + 4]
                q[:] = q / np.linalg.norm(q)
