import cv2
import numpy as np
import torch

#####################################
# 1. Compose (pipeline container)
#####################################
class Compose:
    """A simple pipeline-like container to call each transform in sequence."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
            if results is None:
                return None
        return results

class FrameListInit:
    """Initialize from an in-memory list of frames (decoded images).

    Required key in results:
        - 'frame_list': list of np.ndarrays of shape (H, W, 3) in BGR or RGB.

    Added/Modified keys:
        - 'total_frames': int
        - 'start_index': int
        - 'img_key': str (optional, if you want to track something)
        - 'video_id': str (optional, if you want to track something)
        - 'timestamp': None
    """

    def __init__(self):
        pass

    def __call__(self, results):
        # Check that 'frame_list' is provided
        frame_list = results.get('frame_list', None)
        if frame_list is None:
            raise ValueError('FrameListInit requires "frame_list" in results dict.')

        # Basic info
        total_frames = len(frame_list)

        # Populate keys for downstream transforms
        results['total_frames'] = total_frames
        results['start_index'] = 0
        # These are optional, but keep them for consistency
        results['img_key'] = results.get('img_key', 'in_memory_frames')
        results['video_id'] = results.get('video_id', 'in_memory')
        results['timestamp'] = None

        return results

class FrameListDecode:
    """Select frames from the in-memory 'frame_list' using 'frame_inds'.

    Required keys in results:
        - 'frame_list': list of decoded frames (np.ndarrays)
        - 'frame_inds': 1D array of indices to pick

    Modified / Added keys:
        - 'imgs': list of selected frames
        - 'original_shape': (height, width)
        - 'img_shape': (height, width)
    """

    def __init__(self, convert_bgr_to_rgb=False):
        self.convert_bgr_to_rgb = convert_bgr_to_rgb

    def __call__(self, results):
        frame_list = results['frame_list']
        frame_inds = results['frame_inds']
        if frame_inds.ndim != 1:
            frame_inds = np.squeeze(frame_inds)
            results['frame_inds'] = frame_inds

        imgs = []
        for idx in frame_inds:
            if idx < 0 or idx >= len(frame_list):
                # Out-of-bound frames can still happen if out_of_bound_opt='repeat_last' or 'loop'
                # but typically SampleFrames handles that. We'll be safe anyway.
                # Append a black frame if truly out-of-bounds
                height, width = frame_list[0].shape[:2]
                imgs.append(np.zeros((height, width, 3), dtype=np.uint8))
                continue

            frame = frame_list[idx]
            if self.convert_bgr_to_rgb:
                # If your in-memory frames are BGR and you need them in RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)

        # Save final frames
        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results
class OpenCVInit:
    """Initialize video capture with OpenCV and prepare some basic keys 
    (e.g., total_frames, fps, start_index, etc.)."""

    def __init__(self, io_backend='disk', num_threads=1):
        # For disk-based reading. num_threads won't affect OpenCV but is
        # kept here just to mirror MMACTION2 signatures.
        self.io_backend = io_backend
        self.num_threads = num_threads

    def __call__(self, results):
        # Expect `filename` or `video_path` key in results
        video_path = results.get('filename', None)
        if video_path is None:
            raise ValueError('OpenCVInit requires "filename" in results dict.')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f'Failed to open video file {video_path}')

        # Get total frames and fps if available
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = None  # Some videos may not report FPS properly

        # Store necessary info in results
        results['video_reader'] = cap
        results['total_frames'] = total_frames
        results['avg_fps'] = fps
        results['start_index'] = 0
        # For meta info usage
        results['img_key'] = video_path
        results['video_id'] = video_path.split('/')[-1]
        results['timestamp'] = None
        return results


#################################################
# 3. SampleFrames (direct port of MMACTION2 code)
#################################################
class SampleFrames:
    """Sample frames from the video based on the MMACTION2 SampleFrames logic."""

    def __init__(
        self,
        clip_len,
        frame_interval=1,
        num_clips=1,
        temporal_jitter=False,
        twice_sample=False,
        out_of_bound_opt='loop',
        test_mode=False,
        keep_tail_frames=False,
        target_fps=None
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames, ori_clip_len):
        """Train-mode offset logic from MMACTION2."""
        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (
                    base_offsets + np.random.uniform(0, avg_interval, self.num_clips)
                ).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips
                )
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips
                    )
                )
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def _get_test_clips(self, num_frames, ori_clip_len):
        """Test-mode offset logic from MMACTION2."""
        if self.clip_len == 1:
            # 2D case
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            # 3D case
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                if self.target_fps is not None:
                    # integer floor
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames, ori_clip_len):
        """Choose clip offsets depending on test vs train."""
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)
        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio):
        """Compute original clip length when adjusting for FPS, if needed."""
        if self.target_fps is not None:
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval
        return ori_clip_len

    def __call__(self, results):
        total_frames = results['total_frames']
        fps = results.get('avg_fps', None)
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps

        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        # Build frame_inds
        if self.target_fps:
            # If fps is adjusted, we use a float linspace over ori_clip_len
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len
            ).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len
            )[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        # Temporal jitter
        if self.temporal_jitter:
            perframe_offsets = np.random.randint(self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        # Reshape to [num_clips, clip_len]
        frame_inds = frame_inds.reshape((-1, self.clip_len))

        # Handle out-of-bounds
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = ~safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds) + (unsafe_inds.T * last_ind).T
            frame_inds = new_inds
        else:
            raise ValueError('Invalid out_of_bound_opt')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index

        # Store in the results dict
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


########################################
# 4. OpenCVDecode (replaces DecordDecode)
########################################
class OpenCVDecode:
    """Decode the video frames using OpenCV based on sampled indices.

    - Required keys: 
        'video_reader' (cv2.VideoCapture), 'frame_inds'
    - Added/Modified keys: 
        'imgs', 'original_shape', 'img_shape'
    """

    def __init__(self, mode='accurate'):
        # mode='efficient' logic is not trivial with OpenCV 
        # (Decord uses keyframe seeking). 
        # We will only mirror 'accurate' (frame-by-frame read).
        assert mode in ['accurate', 'efficient']
        self.mode = mode

    def __call__(self, results):
        
        cap = results.get('video_reader', None)
        if cap is None:
            raise ValueError('OpenCVDecode needs `video_reader` in results dict.')

        frame_inds = results['frame_inds']
        if frame_inds.ndim != 1:
            frame_inds = np.squeeze(frame_inds)
            results['frame_inds'] = frame_inds

        # Decoding frames
        imgs = []
        # We always do "accurate" decoding by reading from the start
        # and seeking forward. This is not super fast, 
        # but matches Decord's "accurate" approach (frame-precise).
        for idx in frame_inds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # Fallback: if reading fails, append a black frame
                height = imgs[0].shape[0] if len(imgs) > 0 else 224
                width = imgs[0].shape[1] if len(imgs) > 0 else 224
                imgs.append(np.zeros((height, width, 3), dtype=np.uint8))
                continue
            # Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)

        # Once done, we can free the capture if you don't plan to reuse it
        cap.release()
        results['video_reader'] = None
        

        # Save frames & shapes
        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results


#################################################
# 5. Resize (mirrors MMACTION2 Resize w/ OpenCV)
#################################################
class Resize:
    """Resize frames to a given scale, keeping ratio if needed.

    - Required keys:
        'imgs', 'img_shape'
    - Modified:
        'imgs', 'img_shape'
    """

    def __init__(self, scale, keep_ratio=True, interpolation='bilinear', lazy=False):
        # MMACTION2 uses mmcv; we'll rely on OpenCV's interpolation flags
        interp_dict = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4,
        }
        if interpolation not in interp_dict:
            raise ValueError(f'Unsupported interpolation {interpolation}')
        self.scale = scale  # (inf, 224) or something similar
        self.keep_ratio = keep_ratio
        self.interpolation = interp_dict[interpolation]
        self.lazy = lazy
        self.mean=np.array([
            123.675,
            116.28,
            103.53,
        ])
        self.std=np.array([
            58.395,
            57.12,
            57.375,
        ])

    def _imresize(self, img, new_w, new_h):
        return cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

    def __call__(self, results):
        imgs = results['imgs']
        img_h, img_w = results['img_shape']
        if isinstance(self.scale, tuple):
            # (short_side, long_side) or (inf, something)
            w_scale, h_scale = self.scale
            if w_scale == np.inf:
                # scale so that the shorter side = h_scale
                # which is actually the second tuple element
                # (inf, 224) means: shorter side = 224
                short_side = min(img_w, img_h)
                scale_ratio = h_scale / short_side
                new_w = int(img_w * scale_ratio)
                new_h = int(img_h * scale_ratio)
            else:
                # direct (new_w, new_h)
                new_w, new_h = self.scale
        else:
            # float scale factor
            new_w = int(img_w * self.scale)
            new_h = int(img_h * self.scale)

        if self.keep_ratio and (w_scale != np.inf):
            # Just in case we want to maintain ratio in a different scenario
            # This mimics mmcv.rescale_size with keep_ratio
            # but we’ll skip an intricate check for simplicity.
            pass

        # Resize each frame
        resized = [self._imresize(img, new_w, new_h) for img in imgs]
        
        normalized_tensors = []
        for img in resized:
            normalized_tensor = (img - self.mean[None, None, :]) / self.std[None, None, :]
            normalized_tensors.append(normalized_tensor)
        results['imgs'] = normalized_tensors
        results['img_shape'] = (new_h, new_w)
        return results


####################################################
# 6. ThreeCrop (mirrors MMACTION2 ThreeCrop logic)
####################################################
class ThreeCrop:
    """Crop images into three crops along the shorter side."""

    def __init__(self, crop_size=(224, 224)):
        # If given an int, convert to tuple. Here we assume it's a tuple.
        self.crop_size = crop_size

    def __call__(self, results):
        imgs = results['imgs']
        img_h, img_w = imgs[0].shape[:2]
        crop_w, crop_h = self.crop_size

        # Confirm that either the width matches crop_w or height matches crop_h
        assert (crop_h == img_h) or (crop_w == img_w), (
            "ThreeCrop expects either the image height or width to match the crop size."
        )

        if crop_h == img_h:
            # Horizontal crops
            diff = img_w - crop_w
            if diff < 0:
                raise ValueError(f'Image width {img_w} < crop width {crop_w}')
            step = diff // 2
            offsets = [(0, 0), (2 * step, 0), (step, 0)]
        else:
            # Vertical crops
            diff = img_h - crop_h
            if diff < 0:
                raise ValueError(f'Image height {img_h} < crop height {crop_h}')
            step = diff // 2
            offsets = [(0, 0), (0, 2 * step), (0, step)]

        new_imgs = []
        for (x_off, y_off) in offsets:
            # Crop each frame
            cropped = [
                img[y_off : y_off + crop_h, x_off : x_off + crop_w] for img in imgs
            ]
            new_imgs.extend(cropped)

        # Overwrite
        results['imgs'] = new_imgs
        # The last crop’s shape is the same for all of them
        results['img_shape'] = (crop_h, crop_w)
        return results


###########################################################
# 7. FormatShape (mirrors MMACTION2 FormatShape->NCTHW)
###########################################################
class FormatShape:
    """Rearrange frames from [M, H, W, C] into [N, C, T, H, W], 
    where M = num_crops * num_clips * clip_len.
    """

    def __init__(self, input_format='NCTHW'):
        if input_format != 'NCTHW':
            raise NotImplementedError('Currently only supports NCTHW format.')
        self.input_format = input_format

    def __call__(self, results):
        
        imgs = results['imgs']  # list of np.ndarrays: shape [H,W,C]
        # Convert to a single 4D array: [M, H, W, C]
        # M = (num_crops * num_clips * clip_len)
        imgs = np.array(imgs, np.float32)  
        # If shape = (M, H, W, C)

        num_clips = results['num_clips']
        clip_len = results['clip_len']

        # Reshape: [N_crops, N_clips, clip_len, H, W, C]
        # But we don't explicitly store N_crops. Instead, we do M // (num_clips * clip_len)
        M = imgs.shape[0]
        # In MMACTION2, if you have three crops, M is 3 * num_clips * clip_len
        # Let n_crops = M / (num_clips*clip_len)
        n_crops = M // (num_clips * clip_len)

        imgs = imgs.reshape((n_crops, num_clips, clip_len, *imgs.shape[1:]))

        # Now transpose to [n_crops, num_clips, C, clip_len, H, W]
        imgs = imgs.transpose((0, 1, 5, 2, 3, 4))

        # Flatten out n_crops * num_clips => dimension N:
        # final shape: [N, C, T, H, W]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        results['imgs'] = imgs  # shape [N, C, T, H, W]
        results['input_shape'] = imgs.shape
        return results


###########################################################
# 8. PackActionInputs (similar to MMACTION2, minimal version)
###########################################################
class PackActionInputs:
    """Pack the final data into a dict with 'inputs' and some metainfo."""

    def __init__(self,
                 collect_keys=None,
                 meta_keys=('img_shape', 'img_key', 'video_id', 'timestamp')):
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        packed_results = {}

        # 1) Gather input tensor
        if self.collect_keys is not None:
            # If you specifically want certain keys (e.g. 'imgs')
            packed_results['inputs'] = {}
            for key in self.collect_keys:
                packed_results['inputs'][key] = torch.from_numpy(
                    np.ascontiguousarray(results[key])
                )
        else:
            # By default, gather 'imgs'
            if 'imgs' in results:
                # shape [N, C, T, H, W]
                arr = results['imgs']
                packed_results['inputs'] = torch.from_numpy(np.ascontiguousarray(arr))
            else:
                raise ValueError('No "imgs" key found in results to pack!')

        # 2) Collect meta information
        metainfo = {}
        for k in self.meta_keys:
            if k in results:
                metainfo[k] = results[k]

        packed_results['data_samples'] = {
            'metainfo': metainfo
            # If you want to store bounding boxes, labels, etc., 
            # you would mimic the MMACTION2 approach with an ActionDataSample
        }

        return packed_results


##################################################
# Putting it all together in a pipeline
##################################################
if __name__ == "__main__":
    # Example usage:
    video_path = "../crowded.avi"

    # Build the pipeline
    pipeline = Compose([
        OpenCVInit(io_backend='disk', num_threads=1),
        SampleFrames(clip_len=32, frame_interval=2, num_clips=4, 
                     temporal_jitter=False, twice_sample=False, 
                     out_of_bound_opt='loop', test_mode=True),
        OpenCVDecode(mode='accurate'),
        Resize(scale=(np.inf, 224), keep_ratio=True, interpolation='bilinear', lazy=False),
        ThreeCrop(crop_size=(224, 224)),
        FormatShape(input_format='NCTHW'),
        PackActionInputs(collect_keys=None, 
                         meta_keys=('img_shape', 'img_key', 'video_id', 'timestamp'))
    ])

    # Prepare initial dict
    data = dict(filename=video_path)

    # Run the pipeline
    results = pipeline(data)

    # 'results' is now a dict containing:
    #   {
    #     'inputs': <torch.Tensor of shape [N, C, T, H, W]>,
    #     'data_samples': {
    #         'metainfo': {...}
    #     }
    #   }
    print('Pipeline output keys:', results.keys())
    if 'inputs' in results:
        print('inputs.shape:', results['inputs'].shape)
    if 'data_samples' in results:
        print('metainfo:', results['data_samples']['metainfo'])
    from custom_swin import SwinTransformer3D
    model = SwinTransformer3D(pretrained="../weights/violence_detect/new_swin_fixed.pth",pretrained2d=False,patch_size=(2, 4, 4),
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
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        print(model.predict(results['inputs'].to("cuda")))