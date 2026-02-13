"""
Corner tracking module for LET-NET Python inference.
Converted from tracking.cpp and tracking.h
"""

import numpy as np
import cv2
from typing import List, Tuple


class CornerTracking:
    """Corner tracking class using optical flow and feature extraction"""
    
    def __init__(self):
        self.tracked_points = []
        self.prev_tracked_points = []
        self.prev_score = None
        self.prev_desc = None
        self.tracked_points_history = []
    
    def update(self, score: np.ndarray, desc: np.ndarray):
        """
        Update tracking with new score and descriptor maps
        
        Args:
            score: Score map from neural network (H x W, grayscale)
            desc: Descriptor map from neural network (H x W x 3, BGR)
        """
        if len(self.tracked_points) == 0:  # first frame
            self.tracked_points = self.extract_feature(score)
            self.tracked_points_history = [[pt] for pt in self.tracked_points]
        else:
            # Calculate optical flow
            tracked_points_new, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_desc,
                desc,
                np.array(self.tracked_points, dtype=np.float32),
                None
            )
            
            # Filter tracked points by status
            tracked = []
            tracked_history = []
            for i, st in enumerate(status):
                if st[0]:  # status is returned as [[1]] or [[0]]
                    tracked.append(tracked_points_new[i])
                    self.tracked_points_history[i].append(tuple(tracked_points_new[i]))
                    if len(self.tracked_points_history[i]) > 5:
                        self.tracked_points_history[i].pop(0)
                    tracked_history.append(self.tracked_points_history[i])
            
            # Extract new features
            add = self.extract_feature(score, 20, tracked)
            add_history = [[pt] for pt in add]
            
            # Update tracked points
            self.tracked_points = tracked + add
            self.tracked_points_history = tracked_history + add_history
        
        self.prev_desc = desc.copy()
    
    def show(self, img: np.ndarray):
        """
        Visualize tracked points and their trajectories
        
        Args:
            img: Image to draw on (will be modified in place)
        """
        # Draw tracked points
        for pt in self.tracked_points:
            cv2.circle(img, tuple(map(int, pt)), 2, (0, 255, 0), -1)
        
        # Draw trajectories
        for history in self.tracked_points_history:
            for i in range(1, len(history)):
                pt1 = tuple(map(int, history[i - 1]))
                pt2 = tuple(map(int, history[i]))
                cv2.line(img, pt1, pt2, (0, 0, 255), 1)
        
        cv2.imshow("tracking", img)
    
    def extract_feature(
        self,
        score: np.ndarray,
        ncellsize: int = 20,
        vcurkps: List = None
    ) -> List[Tuple[float, float]]:
        """
        Extract feature points from score map using grid-based NMS
        
        Args:
            score: Score map (H x W, uint8)
            ncellsize: Size of grid cells for feature extraction
            vcurkps: Currently tracked keypoints to avoid
            
        Returns:
            List of detected keypoint coordinates
        """
        if vcurkps is None:
            vcurkps = []
        
        if score.size == 0:
            return []
        
        nrows, ncols = score.shape
        nhalfcell = ncellsize // 4
        
        nhcells = nrows // ncellsize
        nwcells = ncols // ncellsize
        nbcells = nhcells * nwcells
        
        # Initialize occupancy grid
        voccupcells = np.zeros((nhcells + 1, nwcells + 1), dtype=bool)
        
        # Create mask to suppress already tracked points
        mask = np.ones((nrows, ncols), dtype=np.uint8)
        
        for pt in vcurkps:
            px_x, px_y = int(pt[0]), int(pt[1])
            voccupcells[px_y // ncellsize][px_x // ncellsize] = True
            cv2.circle(mask, (px_x, px_y), nhalfcell, 0, -1)
        
        vdetected_px = []
        vvsec_detections_px = []
        
        # Process each cell
        for i in range(nbcells):
            r = i // nwcells
            c = i % nwcells
            
            if voccupcells[r][c]:
                continue
            
            x = c * ncellsize
            y = r * ncellsize
            
            if x + ncellsize < ncols - 1 and y + ncellsize < nrows - 1:
                roi_score = score[y:y+ncellsize, x:x+ncellsize]
                roi_mask = mask[y:y+ncellsize, x:x+ncellsize]
                
                # Find first maximum
                masked_score = roi_score * roi_mask
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_score)
                
                max_px = (max_loc[0] + x, max_loc[1] + y)
                
                if max_val >= 0.2 * 255:  # threshold (score is uint8, 0-255)
                    vdetected_px.append(max_px)
                    cv2.circle(mask, max_px, nhalfcell, 0, -1)
                    
                    # Find second maximum
                    roi_mask = mask[y:y+ncellsize, x:x+ncellsize]
                    masked_score = roi_score * roi_mask
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_score)
                    
                    max_px = (max_loc[0] + x, max_loc[1] + y)
                    
                    if max_val >= 0.2 * 255:
                        vvsec_detections_px.append(max_px)
                        cv2.circle(mask, max_px, nhalfcell, 0, -1)
        
        nbkps = len(vdetected_px)
        nboccup = np.sum(voccupcells)
        
        # Add secondary detections if needed
        if nbkps + nboccup < nbcells:
            nbsec = nbcells - nbkps - nboccup
            for i, sec_kp in enumerate(vvsec_detections_px):
                if i >= nbsec:
                    break
                vdetected_px.append(sec_kp)
        
        return vdetected_px
