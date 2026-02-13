"""
LET-NET Python Inference Demo
Converted from main.cpp to use onnxruntime instead of ncnn
"""

import argparse
import time
import sys
import cv2
import numpy as np
import onnxruntime as ort
from tracking import CornerTracking

# Image dimensions (Note: ONNX model has H/W swapped internally)
IMG_H = 240
IMG_W = 320


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for neural network inference
    
    Args:
        img: Input image (BGR, uint8, shape=(H, W, 3))
        
    Returns:
        Preprocessed image tensor (1, 3, 320, 240) in float32
        Note: H and W are swapped to match ONNX model expectations
    """
    # Normalize to [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Convert BGR to RGB and change to CHW format
    img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    
    # Swap H and W to match ONNX model (which expects (C, 320, 240))
    img_chw_swapped = np.transpose(img_chw, (0, 2, 1))  # (C, H, W) -> (C, W, H)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw_swapped, axis=0)
    
    return img_batch


def postprocess_outputs(score_out: np.ndarray, desc_out: np.ndarray) -> tuple:
    """
    Postprocess network outputs
    
    Args:
        score_out: Score output from network (1, 1, 320, 240)
        desc_out: Descriptor output from network (1, 3, 320, 240)
        Note: Dimensions are swapped in the model
        
    Returns:
        Tuple of (score, descriptor) as uint8 images with correct dimensions
    """
    # Remove batch dimension
    score = score_out[0, 0, :, :]  # (320, 240)
    desc = desc_out[0]  # (3, 320, 240)
    
    # Swap H and W back to correct orientation
    score = np.transpose(score)  # (320, 240) -> (240, 320)
    desc = np.transpose(desc, (0, 2, 1))  # (3, 320, 240) -> (3, 240, 320)
    
    # Convert to HWC format
    desc = np.transpose(desc, (1, 2, 0))  # (3, 240, 320) -> (240, 320, 3)
    
    # Denormalize to [0, 255]
    score_uint8 = (score * 255).astype(np.uint8)
    desc_uint8 = (desc * 255).astype(np.uint8)
    
    # Convert descriptor from RGB to BGR for OpenCV
    desc_bgr = cv2.cvtColor(desc_uint8, cv2.COLOR_RGB2BGR)
    
    return score_uint8, desc_bgr


def main():
    parser = argparse.ArgumentParser(
        description='LET-NET Python Inference Demo'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to video file or first image'
    )
    parser.add_argument(
        'input_path2',
        type=str,
        nargs='?',
        default=None,
        help='Path to second image (if using image mode)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable display windows (save output only)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output images (default: output)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    import os
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Determine if we're processing video or images
    is_video = args.input_path2 is None
    
    # Initialize video capture or load images
    if is_video:
        capture = cv2.VideoCapture(args.input_path)
        if not capture.isOpened():
            print("Error opening video file!")
            return -1
        
        # Initialize video writer
        writer = cv2.VideoWriter(
            'output.avi',
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            10,
            (IMG_W, IMG_H)
        )
    else:
        img1 = cv2.imread(args.input_path)
        img2 = cv2.imread(args.input_path2)
        
        if img1 is None or img2 is None:
            print("Error opening image files!")
            return -1
        
        img1 = cv2.resize(img1, (IMG_W, IMG_H))
        img2 = cv2.resize(img2, (IMG_W, IMG_H))
        images = [img1, img2]
        img_idx = 0
    
    # Load ONNX model
    print(f"Loading model from {args.model_path}")
    try:
        session = ort.InferenceSession(
            args.model_path,
            providers=['CPUExecutionProvider']
        )
        print("Model loaded successfully")
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        print(f"Input name: {input_name}")
        print(f"Output names: {output_names}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return -1
    
    # Initialize tracker
    tracker = CornerTracking()
    
    # Main processing loop
    frame_count = 0
    while True:
        # Get frame
        if is_video:
            ret, frame = capture.read()
            if not ret or frame is None:
                break
        else:
            if img_idx >= len(images):
                break
            frame = images[img_idx]
            img_idx += 1
        
        # Resize frame
        frame = cv2.resize(frame, (IMG_W, IMG_H))
        
        # Preprocess
        t1 = time.time()
        input_tensor = preprocess_image(frame)
        t2 = time.time()
        
        # Run inference
        try:
            outputs = session.run(output_names, {input_name: input_tensor})
            score_out = outputs[0]  # Assuming first output is score
            desc_out = outputs[1]   # Assuming second output is descriptor
        except Exception as e:
            print(f"Error during inference: {e}")
            break
        
        t3 = time.time()
        
        # Postprocess
        score, desc = postprocess_outputs(score_out, desc_out)
        t4 = time.time()
        
        # Update tracker
        tracker.update(score, desc)
        
        # Visualize
        output_frame = frame.copy()
        if args.no_display:
            # Draw on frame without showing
            for pt in tracker.tracked_points:
                cv2.circle(output_frame, tuple(map(int, pt)), 2, (0, 255, 0), -1)
            for history in tracker.tracked_points_history:
                for i in range(1, len(history)):
                    pt1 = tuple(map(int, history[i - 1]))
                    pt2 = tuple(map(int, history[i]))
                    cv2.line(output_frame, pt1, pt2, (0, 0, 255), 1)
            # Save frame
            output_path = f"{args.output_dir}/frame_{frame_count:04d}.png"
            cv2.imwrite(output_path, output_frame)
        else:
            tracker.show(output_frame)
        
        # Write to output video if in video mode
        if is_video:
            writer.write(output_frame)
        
        # Display timing information
        time_preprocess = (t2 - t1) * 1000
        time_inference = (t3 - t2) * 1000
        time_postprocess = (t4 - t3) * 1000
        
        print(f"Frame {frame_count}:")
        print(f"  Preprocess:  {time_preprocess:.2f}ms")
        print(f"  Inference:   {time_inference:.2f}ms")
        print(f"  Postprocess: {time_postprocess:.2f}ms")
        
        frame_count += 1
        
        # Wait for key press (500ms for video, 0 for images)
        if not args.no_display:
            key = cv2.waitKey(500 if is_video else 0)
            if key == 27:  # ESC key
                break
    
    # Cleanup
    if is_video:
        writer.release()
        capture.release()
    
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
