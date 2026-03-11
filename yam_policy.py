import numpy as np
import pinocchio as pin
import cv2
import torch
import os
from ultralytics import YOLOWorld

class YoloPinocchioPolicy:
    """
    A unified LeRobot Policy that completely replaces the ROS 2 architecture.
    It combines Vision (YOLO) and Brain (Pinocchio IK) into a single standard evaluate loop.
    """
    def __init__(self, urdf_path="~/yam_ws/src/yam_botany_urdf/yam_st_urdf_with_linear_gripper.urdf"):
        print("⏳ Loading YOLO-World AI for LeRobot...")
        self.yolo = YOLOWorld("yolov8s-world.pt")
        
        # Define the target words (User can change these!)
        self.pick_class = "bottle"
        self.place_class = "box"
        self.yolo.set_classes([self.pick_class, self.place_class])
        
        print("⏳ Loading Pinocchio Kinematics for LeRobot...")
        expanded_urdf = os.path.expanduser(urdf_path)
        self.model = pin.buildModelFromUrdf(expanded_urdf)
        self.data = self.model.createData()
        self.JOINT_ID = self.model.getFrameId("gripper")
        self.current_q = np.zeros(self.model.nq)
        
        self.target_pos = None
        self.dynamic_drop_zone = None
        self.robot_state = 'TRACKING'
        self.gripper_cmd = 1.0
        self.grasp_timer = 0
        self.drop_timer = 0
        
        # --- CALIBRATION ---
        # Physical Translation and Rotation of Zed-M to Robot Base
        self.OFFSET_X = -0.235  # 23.5 cm behind the robot (-)
        self.OFFSET_Y = 0.283   # 28.3 cm to the left (+)
        self.OFFSET_Z = 0.58    # 58.0 cm high (+)
        self.TILT_ANGLE = 45.0  # Tilted down 45 degrees
        self.TABLE_Z = -0.045   # The robot base is 4.5 cm ABOVE the table.
        print("🚀 Custom YAM Policy is ONLINE.")

    def get_3d_point_heuristic(self, cx, cy, img_h, img_w):
        """Math adapted for the tilted physical ZED-M camera."""
        # 1. Approx ZED Pinhole Intrinsics
        fx, fy = 350.0, 350.0  
        px, py = img_w / 2.0, img_h / 2.0
        
        # 2. Camera Ray (Optical Frame: Z=Forward, X=Right, Y=Down)
        ray_c = np.array([(cx - px) / fx, (cy - py) / fy, 1.0])
        ray_c = ray_c / np.linalg.norm(ray_c)
        
        # 3. Base Camera Rotation Matrix (Optical to Robot Base)
        R_opt2robot = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ])
        
        # 4. Apply the user's 45-degree pitch downward
        pitch_rad = np.radians(self.TILT_ANGLE)
        R_tilt = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
        ])
        
        ray_robot = (R_opt2robot @ R_tilt) @ ray_c
        
        # 5. Ray-Plane Intersection with Table
        cam_pos_robot = np.array([self.OFFSET_X, self.OFFSET_Y, self.OFFSET_Z])
        t = (self.TABLE_Z - cam_pos_robot[2]) / ray_robot[2]
        s
        target_point = cam_pos_robot + t * ray_robot
        return float(target_point[0]), float(target_point[1]), self.TABLE_Z
        
    def select_action(self, batch):
        """
        This is the main function LeRobot calls ~30 times a second.
        It passes in a 'batch' which contains the Zed-M camera images and joint states.
        We must calculate the target joints and return the `action` tensor.
        """
        # 1. Extract the Zed-M camera image from the LeRobot batch (B, C, H, W)
        image_tensor = batch.get('observation.images.topdown')
        
        if image_tensor is not None:
            # Convert PyTorch Tensor back to OpenCV Image for YOLO
            image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            color_frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            img_h, img_w, _ = color_frame.shape
            
            # Run YOLO-World (Vision Node Logic)
            results = self.yolo.predict(color_frame, verbose=False, conf=0.1)
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                object_name = self.yolo.names[class_id]
                cx, cy, _, _ = box.xywh[0]
                cx, cy = int(cx), int(cy)
                
                point_3d = self.get_3d_point_heuristic(cx, cy, img_h, img_w)
                
                if object_name == self.pick_class and self.robot_state in ['TRACKING', 'DESCENDING']:
                    self.target_pos = np.array(point_3d)
                elif object_name == self.place_class:
                    self.dynamic_drop_zone = np.array(point_3d)
        
        # 2. Read true joint angles from the real robot (via LeRobot batch)
        if 'observation.state' in batch:
            state_tensor = batch['observation.state'].squeeze(0).cpu().numpy()
            n = min(len(state_tensor), self.model.nq)
            self.current_q[:n] = state_tensor[:n]

        # 3. Run Pinocchio Inverse Kinematics (Brain Node Logic)
        if self.target_pos is not None:
            pin.forwardKinematics(self.model, self.data, self.current_q)
            pin.updateFramePlacements(self.model, self.data)
            wrist_pose = self.data.oMf[self.JOINT_ID]
            tcp_offset_local = np.array([0.0, 0.0, -0.07]) 
            fingertip_pos = wrist_pose.translation + wrist_pose.rotation @ tcp_offset_local
            
            OPEN_VAL, CLOSE_VAL = 1.0, 0.0
            
            # --- STEP 1: SET THE TARGET BASED ON CURRENT STATE ---
            if self.robot_state == 'TRACKING':
                self.target_pos[2] = self.TABLE_Z + 0.20      
                self.gripper_cmd = OPEN_VAL
            elif self.robot_state == 'DESCENDING':
                self.target_pos[2] = self.TABLE_Z + 0.025    
                self.gripper_cmd = OPEN_VAL
            elif self.robot_state == 'GRASPING':
                self.target_pos[2] = self.TABLE_Z + 0.025    
                self.gripper_cmd = CLOSE_VAL 
            elif self.robot_state == 'LIFTING':
                self.target_pos[2] = self.TABLE_Z + 0.30      
                self.gripper_cmd = CLOSE_VAL
            elif self.robot_state == 'MOVING_TO_DROP':
                if self.dynamic_drop_zone is not None:
                    self.target_pos = self.dynamic_drop_zone.copy()
                    self.target_pos[2] = self.TABLE_Z + 0.15 # Drop from 15cm higher to clear box edges
                self.gripper_cmd = CLOSE_VAL
            elif self.robot_state == 'RELEASING':
                if self.dynamic_drop_zone is not None:
                    self.target_pos = self.dynamic_drop_zone.copy()
                    self.target_pos[2] = self.TABLE_Z + 0.15
                self.gripper_cmd = OPEN_VAL

            # --- STEP 2: THE 1 CM SAFETY FLOOR ---
            # Enforce absolute Z-limit: Do not go below -0.035m (1cm above table)
            if self.target_pos[2] < -0.035:
                self.target_pos[2] = -0.035
                
            error = self.target_pos - fingertip_pos
            xy_error_norm = np.linalg.norm(error[:2]) 
            total_error_norm = np.linalg.norm(error)  

            # State Machine Transitions
            if self.robot_state == 'TRACKING' and xy_error_norm < 0.02:
                print("Target locked! DESCENDING...")
                self.robot_state = 'DESCENDING'
            
            # 🚨 RESTORED 8CM THRESHOLD: The IK physics can't always reach perfect 2cm accuracy mathematically!
            elif self.robot_state == 'DESCENDING' and total_error_norm < 0.08: 
                print("Reached object! GRASPING...")
                self.robot_state = 'GRASPING'
                self.grasp_timer = 0
                
            elif self.robot_state == 'GRASPING':
                self.grasp_timer += 1
                if self.grasp_timer > 70: 
                    print("Got it! LIFTING...")
                    self.robot_state = 'LIFTING'
            elif self.robot_state == 'LIFTING' and total_error_norm < 0.05:
                print("Lifted! MOVING TO DROP ZONE...")
                self.robot_state = 'MOVING_TO_DROP'
            elif self.robot_state == 'MOVING_TO_DROP' and total_error_norm < 0.05:
                print("At drop zone! RELEASING...")
                self.robot_state = 'RELEASING'
                self.drop_timer = 0
            elif self.robot_state == 'RELEASING':
                self.drop_timer += 1
                if self.drop_timer > 50:
                    print("Object dropped! DONE.")
                    self.robot_state = 'DONE'

            if total_error_norm > 0.005:
                J = pin.computeFrameJacobian(self.model, self.data, self.current_q, self.JOINT_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                damping = 0.02
                inv_J = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3))
                step = inv_J @ (error * 0.1)
                self.current_q = pin.integrate(self.model, self.current_q, step)

        # 4. Format output exactly how LeRobot expects it (PyTorch Tensor)
        action_array = np.append(self.current_q[:7], self.gripper_cmd)
        action_tensor = torch.tensor(action_array, dtype=torch.float32).unsqueeze(0)
        
        return {"action": action_tensor}
