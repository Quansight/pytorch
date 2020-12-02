import operator_benchmark as op_bench
import torch

"""Microbenchmarks for the interpolate op.

python -m pt.interpolate_test --ai_pep_format --tag_filter=all > bench_interpolate.log

1) upsample_bilinear2d_cpu to support uint8
  - downsampling in/out sizes: 300,400,500 -> 256, 1024 -> 512, 128 -> 32
  - upsampling in/out sizes: 500 -> 800 (opt 300 -> 512, 64 -> 256)
  - channels-last format with float32, uint8

2) Compare with OpenCV and PIL if available
"""

HAS_OPENCV = True
try:
    import numpy as np
    import cv2
except ImportError:
    HAS_OPENCV = False

HAS_PILLOW = True
try:
    import numpy as np
    import PIL
except ImportError:
    HAS_PILLOW = False


interpolate_2d_downsample_configs = op_bench.cross_product_configs(
    in_size=[300, 400, 500],
    out_size=[256],
    dtype=[torch.uint8, torch.float32],
    channels_last=[False, True],
    with_uint8_float32_cast=[True, ],
    mode=["nearest", "bilinear"],
    tags=["downsample"]
)

interpolate_2d_large_downsample_configs = op_bench.cross_product_configs(
    in_size=[1024, ],
    out_size=[512, ],
    dtype=[torch.uint8, torch.float32],
    channels_last=[False, True],
    with_uint8_float32_cast=[True, ],
    mode=["nearest", "bilinear"],
    tags=["downsample large"]
)

interpolate_2d_small_downsample_configs = op_bench.cross_product_configs(
    in_size=[128, ],
    out_size=[32, ],
    dtype=[torch.uint8, torch.float32],
    channels_last=[False, True],
    with_uint8_float32_cast=[True, ],
    mode=["nearest", "bilinear"],
    tags=["downsample small"]
)

interpolate_2d_upsample_configs = op_bench.cross_product_configs(
    in_size=[500, ],
    out_size=[800, ],
    dtype=[torch.uint8, torch.float32],
    channels_last=[False, True],
    with_uint8_float32_cast=[True, ],
    mode=["nearest", "bilinear"],
    tags=["upsample"]
)

interpolate_2d_small_upsample_configs = op_bench.cross_product_configs(
    in_size=[300, ],
    out_size=[456, ],
    dtype=[torch.uint8, torch.float32],
    channels_last=[False, True],
    with_uint8_float32_cast=[True, ],
    mode=["nearest", "bilinear"],
    tags=["upsample small"]
)

interpolate_2d_large_upsample_configs = op_bench.cross_product_configs(
    in_size=[64, ],
    out_size=[256, ],
    dtype=[torch.uint8, torch.float32],
    channels_last=[False, True],
    with_uint8_float32_cast=[True, ],
    mode=["nearest", "bilinear"],
    tags=["upsample large"]
)

# all_configs = interpolate_2d_downsample_configs
all_configs = interpolate_2d_downsample_configs + \
    interpolate_2d_large_downsample_configs + \
    interpolate_2d_small_downsample_configs + \
    interpolate_2d_upsample_configs + \
    interpolate_2d_large_upsample_configs + \
    interpolate_2d_small_upsample_configs


class InterpolateBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_size, out_size, dtype, channels_last, with_uint8_float32_cast, mode):
        
        x = torch.randint(0, 256, size=(1, 3, in_size, in_size), dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        align_corners = False if mode != "nearest" else None

        self.inputs = {
            "x": x,
            "out_size": out_size,
            "mode": mode,
            "align_corners": align_corners,
            "with_uint8_float32_cast": with_uint8_float32_cast
        }
        self.set_module_name("interpolate")

    def forward(self, x, out_size, mode, align_corners, with_uint8_float32_cast):
        if with_uint8_float32_cast and mode != "nearest" and x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        return torch.nn.functional.interpolate(
            x,
            size=[out_size, out_size],
            mode=mode,
            align_corners=align_corners
        )


op_bench.generate_pt_test(all_configs, InterpolateBenchmark)

if HAS_OPENCV:
    print("opencv version:", cv2.__version__)
    print("numpy version:", np.__version__)

    cv2_resize_2d_downsample_configs = op_bench.cross_product_configs(
        in_size=[300, 400, 500],
        out_size=[256],
        mode=["nearest", "bilinear"],
        tags=["cv2 downsample"]
    )

    cv2_resize_2d_large_downsample_configs = op_bench.cross_product_configs(
        in_size=[1024, ],
        out_size=[512, ],
        mode=["nearest", "bilinear"],
        tags=["cv2 downsample large"]
    )

    cv2_resize_2d_small_downsample_configs = op_bench.cross_product_configs(
        in_size=[128, ],
        out_size=[32, ],
        mode=["nearest", "bilinear"],
        tags=["cv2 downsample small"]
    )

    cv2_resize_2d_upsample_configs = op_bench.cross_product_configs(
        in_size=[500, ],
        out_size=[800, ],
        mode=["nearest", "bilinear"],
        tags=["cv2 upsample"]
    )

    cv2_resize_2d_small_upsample_configs = op_bench.cross_product_configs(
        in_size=[300, ],
        out_size=[456, ],
        mode=["nearest", "bilinear"],
        tags=["cv2 upsample small"]
    )

    cv2_resize_2d_large_upsample_configs = op_bench.cross_product_configs(
        in_size=[64, ],
        out_size=[256, ],
        mode=["nearest", "bilinear"],
        tags=["cv2 upsample large"]
    )

    # all_opencv_configs = cv2_resize_2d_downsample_configs
    all_opencv_configs = cv2_resize_2d_downsample_configs + \
        cv2_resize_2d_large_downsample_configs + \
        cv2_resize_2d_small_downsample_configs + \
        cv2_resize_2d_upsample_configs + \
        cv2_resize_2d_small_upsample_configs + \
        cv2_resize_2d_large_upsample_configs


    class OpencvResizeBenchmark(op_bench.TorchBenchmarkBase):
        def init(self, in_size, out_size, mode):
            
            x = np.random.randint(0, 256, size=(in_size, in_size, 3), dtype='uint8')

            interpolation_map = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
            }

            self.inputs = {
                "x": x,
                "out_size": out_size,
                "interpolation": interpolation_map[mode],
            }
            self.set_module_name("cv2-resize")

        def forward(self, x, out_size, interpolation):
            return cv2.resize(x, dsize=(out_size, out_size), interpolation=interpolation)
    

    op_bench.generate_pt_test(all_opencv_configs, OpencvResizeBenchmark)

if HAS_PILLOW:
    from PIL import Image

    print("PIL version:", PIL.__version__)
    print("numpy version:", np.__version__)    

    pillow_resize_2d_downsample_configs = op_bench.cross_product_configs(
        in_size=[300, 400, 500],
        out_size=[256],
        mode=["nearest", "bilinear"],
        tags=["pillow downsample"]
    )

    pillow_resize_2d_large_downsample_configs = op_bench.cross_product_configs(
        in_size=[1024, ],
        out_size=[512, ],
        mode=["nearest", "bilinear"],
        tags=["pillow downsample large"]
    )

    pillow_resize_2d_small_downsample_configs = op_bench.cross_product_configs(
        in_size=[128, ],
        out_size=[32, ],
        mode=["nearest", "bilinear"],
        tags=["pillow downsample small"]
    )

    pillow_resize_2d_upsample_configs = op_bench.cross_product_configs(
        in_size=[500, ],
        out_size=[800, ],
        mode=["nearest", "bilinear"],
        tags=["pillow upsample"]
    )

    pillow_resize_2d_small_upsample_configs = op_bench.cross_product_configs(
        in_size=[300, ],
        out_size=[456, ],
        mode=["nearest", "bilinear"],
        tags=["pillow upsample small"]
    )

    pillow_resize_2d_large_upsample_configs = op_bench.cross_product_configs(
        in_size=[64, ],
        out_size=[256, ],
        mode=["nearest", "bilinear"],
        tags=["pillow upsample large"]
    )

    # all_pillow_configs = pillow_resize_2d_downsample_configs
    all_pillow_configs = pillow_resize_2d_downsample_configs + \
        pillow_resize_2d_large_downsample_configs + \
        pillow_resize_2d_small_downsample_configs + \
        pillow_resize_2d_upsample_configs + \
        pillow_resize_2d_small_upsample_configs + \
        pillow_resize_2d_large_upsample_configs


    class PillowResizeBenchmark(op_bench.TorchBenchmarkBase):
        def init(self, in_size, out_size, mode):
            
            x = Image.fromarray(np.random.randint(0, 256, size=(in_size, in_size, 3), dtype='uint8'))

            interpolation_map = {
                "nearest": PIL.Image.NEAREST,
                "bilinear": PIL.Image.BILINEAR,
            }

            self.inputs = {
                "x": x,
                "out_size": out_size,
                "interpolation": interpolation_map[mode],
            }
            self.set_module_name("pillow-resize")

        def forward(self, x, out_size, interpolation):
            return x.resize((out_size, out_size), interpolation)
    

    op_bench.generate_pt_test(all_pillow_configs, PillowResizeBenchmark)

if __name__ == '__main__':
    print("torch version:", torch.__version__, torch.__file__)
    print(torch.version.debug, torch.version.cuda, torch.version.git_version)
    op_bench.benchmark_runner.main()
