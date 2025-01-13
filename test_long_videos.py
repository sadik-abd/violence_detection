from transforms import *
from custom_swin import SwinTransformer3D
import torch
import time
import cv2
import numpy as np
import os

# Build the pipeline
pipeline = Compose([
    FrameListInit(),
    SampleFrames(clip_len=32, frame_interval=2, num_clips=4, 
                    temporal_jitter=False, twice_sample=False, 
                    out_of_bound_opt='loop', test_mode=True),
    FrameListDecode(convert_bgr_to_rgb=True),
    Resize(scale=(np.inf, 224), keep_ratio=True, interpolation='bilinear', lazy=False),
    ThreeCrop(crop_size=(224, 224)),
    FormatShape(input_format='NCTHW'),
    PackActionInputs(collect_keys=None, 
                        meta_keys=('img_shape', 'img_key', 'video_id', 'timestamp'))
])

model = SwinTransformer3D(pretrained="./weights/violence_swin.pth",pretrained2d=False,patch_size=(2, 4, 4),
                in_chans=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=(8, 7, 7),
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                norm_layer=torch.nn.LayerNorm,
                patch_norm=True,
                frozen_stages=-1)
model.init_weights()
model.eval()
model.to("cuda")

def process_video_sliding_window(video_path, model, pipeline, violence_threshold=0.7, stride_secs=1.0, fps=25):
    # Create output directories if they don't exist
    os.makedirs("violence_outputs", exist_ok=True)
    os.makedirs("non_violence_outputs", exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    window_size_frames = 128 * 2 # we need 128 frames for inference but preprocessing cut it into half with frame interval
    stride_frames = int(stride_secs * video_fps)
    
    frame_buffer = []
    frame_count = 0
    clip_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_buffer.append(frame)
        frame_count += 1
        
        if len(frame_buffer) == window_size_frames:
            # Process current window
            height, width = frame_buffer[0].shape[:2]
            timestamp = (frame_count - window_size_frames) / video_fps
            
            # Generate output filename
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            clip_name = f"{base_filename}_clip_{clip_counter}_time_{timestamp:.2f}.mp4"
            
            # Get model prediction
            t = time.time()
            data = dict(avg_fps=video_fps,frame_list=frame_buffer)
            results = pipeline(data)
            print(f'time taken to preprocess a clip: {(time.time()-t)*1000} ms')
            
            t=time.time()
            with torch.no_grad():
                output = model.predict(results['inputs'].to("cuda"))
            score, pred = output
            prob_violence = score[0, 1].item()
            print(f'time taken to infer a clip: {(time.time()-t)*1000} ms')
            
            # Save to appropriate directory based on threshold
            if prob_violence > violence_threshold:
                output_path = os.path.join("violence_outputs", clip_name)
            else:
                output_path = os.path.join("non_violence_outputs", clip_name)
                
            # Save the actual clip
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),  video_fps, (width, height))
            for f in frame_buffer:
                out.write(f)
            out.release()
            
            # Remove stride frames from the beginning
            frame_buffer = frame_buffer[stride_frames:]
            clip_counter += 1

    
    cap.release()


video_path_list = ["2025-01-13 07-05-13.mkv"]
for vid_path in video_path_list:
    process_video_sliding_window(vid_path, model, pipeline, violence_threshold=0.7)
