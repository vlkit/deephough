from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='deephough',
    version='0.1.0',
    author='Qi Han and Kai Zhao',
    url='https://github.com/vlkit/deephough',
    license='MIT',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='deephough.hough_ext',
            sources=[
                'deephough/deep_hough_cuda.cpp',
                'deephough/deep_hough_cuda_kernel.cu'
            ], include_dirs=['deephough'],
        extra_compile_args={'nvcc': ['-arch=sm_60']})
    ],
    cmdclass={'build_ext': BuildExtension},
    package_data={'deephough': ['deep_hough_cuda.cpp', 'deep_hough_cuda_kernel.cu']},
)
