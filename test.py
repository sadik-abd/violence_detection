from transforms import *
from custom_swin import SwinTransformer3D
import torch
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


print('Pipeline output keys:', results.keys())
if 'inputs' in results:
    print('inputs.shape:', results['inputs'].shape)
if 'data_samples' in results:
    print('metainfo:', results['data_samples']['metainfo'])
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
model.to("cuda")
model.eval()
with torch.no_grad():
    print(model.predict(results['inputs'].to("cuda")))