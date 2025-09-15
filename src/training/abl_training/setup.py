from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='lsr_cpp',
    ext_modules=[
        CUDAExtension(
            name='lsr_cpp',
            sources=[
                os.path.join(this_dir, 'abl_training/losses/lsr_cpp/csrc/lsr_kernel.cu'),
            ],
            include_dirs=[
                os.path.join(this_dir, 'abl_training/losses/lsr_cpp/csrc'),
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
