# The following is a terminal dump of a series of steps I went through to get gpu inference working

# It is possible


You need sudo apt install build-essential, and cmake from apt install or conda

useful blog: https://michaelriedl.com/2023/09/10/llama2-install-gpu.html

then

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc -y --copy

then

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

then

pip install protobuf sentencepiece transformers matplotlib
```


root@f2efb2d32613:~# conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc -y --copy






apt install build-essential && conda install cmake && conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc -y --copy && CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir && pip install protobuf sentencepiece transformers matplotlib

^ and then Y to everything










Collecting package metadata (current_repodata.json): \
done
Solving environment: |
The environment is inconsistent, please check the package plan carefully
The following packages are causing the inconsistency:

  - conda-forge/linux-64::krb5==1.21.2=h659d440_0
  - conda-forge/linux-64::libcurl==8.2.1=hca28451_0
  - conda-forge/linux-64::libedit==3.1.20191231=he28a2e2_2
  - conda-forge/linux-64::libmamba==1.4.9=h658169a_0
  - conda-forge/linux-64::libmambapy==1.4.9=py311h527f279_0
  - conda-forge/linux-64::mamba==1.4.9=py311h3072747_0
done

## Package Plan ##

  environment location: /home/user/micromamba

  added / updated specs:
    - cuda-nvcc
    - cuda-toolkit


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ca-certificates-2023.08.22 |       h06a4308_0         123 KB
    certifi-2023.7.22          |  py311h06a4308_0         154 KB
    cuda-cccl-11.8.89          |                0         1.2 MB  nvidia/label/cuda-11.8.0
    cuda-command-line-tools-11.8.0|                0           1 KB  nvidia/label/cuda-11.8.0
    cuda-compiler-11.8.0       |                0           1 KB  nvidia/label/cuda-11.8.0
    cuda-cudart-dev-11.8.89    |                0         1.1 MB  nvidia/label/cuda-11.8.0
    cuda-cuobjdump-11.8.86     |                0         229 KB  nvidia/label/cuda-11.8.0
    cuda-cuxxfilt-11.8.86      |                0         291 KB  nvidia/label/cuda-11.8.0
    cuda-documentation-11.8.86 |                0          89 KB  nvidia/label/cuda-11.8.0
    cuda-driver-dev-11.8.89    |                0          16 KB  nvidia/label/cuda-11.8.0
    cuda-gdb-11.8.86           |                0         4.8 MB  nvidia/label/cuda-11.8.0
    cuda-libraries-dev-11.8.0  |                0           2 KB  nvidia/label/cuda-11.8.0
    cuda-memcheck-11.8.86      |                0         168 KB  nvidia/label/cuda-11.8.0
    cuda-nsight-11.8.86        |                0       113.6 MB  nvidia/label/cuda-11.8.0
    cuda-nsight-compute-11.8.0 |                0           1 KB  nvidia/label/cuda-11.8.0
    cuda-nvcc-11.8.89          |                0        50.8 MB  nvidia/label/cuda-11.8.0
    cuda-nvdisasm-11.8.86      |                0        48.7 MB  nvidia/label/cuda-11.8.0
    cuda-nvml-dev-11.8.86      |                0          83 KB  nvidia/label/cuda-11.8.0
    cuda-nvprof-11.8.87        |                0         4.4 MB  nvidia/label/cuda-11.8.0
    cuda-nvprune-11.8.86       |                0          65 KB  nvidia/label/cuda-11.8.0
    cuda-nvrtc-dev-11.8.89     |                0        17.0 MB  nvidia/label/cuda-11.8.0
    cuda-nvvp-11.8.87          |                0       114.4 MB  nvidia/label/cuda-11.8.0
    cuda-profiler-api-11.8.86  |                0          18 KB  nvidia/label/cuda-11.8.0
    cuda-sanitizer-api-11.8.86 |                0        16.6 MB  nvidia/label/cuda-11.8.0
    cuda-toolkit-11.8.0        |                0           1 KB  nvidia/label/cuda-11.8.0
    cuda-tools-11.8.0          |                0           1 KB  nvidia/label/cuda-11.8.0
    cuda-visual-tools-11.8.0   |                0           1 KB  nvidia/label/cuda-11.8.0
    gds-tools-1.4.0.31         |                0           2 KB  nvidia/label/cuda-11.8.0
    libcublas-dev-11.11.3.6    |                0       394.1 MB  nvidia/label/cuda-11.8.0
    libcufft-dev-10.9.0.58     |                0       275.8 MB  nvidia/label/cuda-11.8.0
    libcufile-dev-1.4.0.31     |                0         1.6 MB  nvidia/label/cuda-11.8.0
    libcurand-dev-10.3.0.86    |                0        53.7 MB  nvidia/label/cuda-11.8.0
    libcusolver-dev-11.4.1.48  |                0        66.3 MB  nvidia/label/cuda-11.8.0
    libcusparse-dev-11.7.5.86  |                0       359.7 MB  nvidia/label/cuda-11.8.0
    libedit-3.1.20221030       |       h5eee18b_0         181 KB
    libnpp-dev-11.8.0.86       |                0       144.5 MB  nvidia/label/cuda-11.8.0
    libnvjpeg-dev-11.9.0.86    |                0         2.1 MB  nvidia/label/cuda-11.8.0
    nsight-compute-2022.3.0.22 |                0       610.0 MB  nvidia/label/cuda-11.8.0
    ------------------------------------------------------------
                                           Total:        2.23 GB

The following NEW packages will be INSTALLED:

  cuda-cccl          nvidia/label/cuda-11.8.0/linux-64::cuda-cccl-11.8.89-0
  cuda-command-line~ nvidia/label/cuda-11.8.0/linux-64::cuda-command-line-tools-11.8.0-0
  cuda-compiler      nvidia/label/cuda-11.8.0/linux-64::cuda-compiler-11.8.0-0
  cuda-cudart-dev    nvidia/label/cuda-11.8.0/linux-64::cuda-cudart-dev-11.8.89-0
  cuda-cuobjdump     nvidia/label/cuda-11.8.0/linux-64::cuda-cuobjdump-11.8.86-0
  cuda-cuxxfilt      nvidia/label/cuda-11.8.0/linux-64::cuda-cuxxfilt-11.8.86-0
  cuda-documentation nvidia/label/cuda-11.8.0/linux-64::cuda-documentation-11.8.86-0
  cuda-driver-dev    nvidia/label/cuda-11.8.0/linux-64::cuda-driver-dev-11.8.89-0
  cuda-gdb           nvidia/label/cuda-11.8.0/linux-64::cuda-gdb-11.8.86-0
  cuda-libraries-dev nvidia/label/cuda-11.8.0/linux-64::cuda-libraries-dev-11.8.0-0
  cuda-memcheck      nvidia/label/cuda-11.8.0/linux-64::cuda-memcheck-11.8.86-0
  cuda-nsight        nvidia/label/cuda-11.8.0/linux-64::cuda-nsight-11.8.86-0
  cuda-nsight-compu~ nvidia/label/cuda-11.8.0/linux-64::cuda-nsight-compute-11.8.0-0
  cuda-nvcc          nvidia/label/cuda-11.8.0/linux-64::cuda-nvcc-11.8.89-0
  cuda-nvdisasm      nvidia/label/cuda-11.8.0/linux-64::cuda-nvdisasm-11.8.86-0
  cuda-nvml-dev      nvidia/label/cuda-11.8.0/linux-64::cuda-nvml-dev-11.8.86-0
  cuda-nvprof        nvidia/label/cuda-11.8.0/linux-64::cuda-nvprof-11.8.87-0
  cuda-nvprune       nvidia/label/cuda-11.8.0/linux-64::cuda-nvprune-11.8.86-0
  cuda-nvrtc-dev     nvidia/label/cuda-11.8.0/linux-64::cuda-nvrtc-dev-11.8.89-0
  cuda-nvvp          nvidia/label/cuda-11.8.0/linux-64::cuda-nvvp-11.8.87-0
  cuda-profiler-api  nvidia/label/cuda-11.8.0/linux-64::cuda-profiler-api-11.8.86-0
  cuda-sanitizer-api nvidia/label/cuda-11.8.0/linux-64::cuda-sanitizer-api-11.8.86-0
  cuda-toolkit       nvidia/label/cuda-11.8.0/linux-64::cuda-toolkit-11.8.0-0
  cuda-tools         nvidia/label/cuda-11.8.0/linux-64::cuda-tools-11.8.0-0
  cuda-visual-tools  nvidia/label/cuda-11.8.0/linux-64::cuda-visual-tools-11.8.0-0
  gds-tools          nvidia/label/cuda-11.8.0/linux-64::gds-tools-1.4.0.31-0
  libcublas-dev      nvidia/label/cuda-11.8.0/linux-64::libcublas-dev-11.11.3.6-0
  libcufft-dev       nvidia/label/cuda-11.8.0/linux-64::libcufft-dev-10.9.0.58-0
  libcufile-dev      nvidia/label/cuda-11.8.0/linux-64::libcufile-dev-1.4.0.31-0
  libcurand-dev      nvidia/label/cuda-11.8.0/linux-64::libcurand-dev-10.3.0.86-0
  libcusolver-dev    nvidia/label/cuda-11.8.0/linux-64::libcusolver-dev-11.4.1.48-0
  libcusparse-dev    nvidia/label/cuda-11.8.0/linux-64::libcusparse-dev-11.7.5.86-0
  libnpp-dev         nvidia/label/cuda-11.8.0/linux-64::libnpp-dev-11.8.0.86-0
  libnvjpeg-dev      nvidia/label/cuda-11.8.0/linux-64::libnvjpeg-dev-11.9.0.86-0
  nsight-compute     nvidia/label/cuda-11.8.0/linux-64::nsight-compute-2022.3.0.22-0

The following packages will be UPDATED:

  ca-certificates    conda-forge::ca-certificates-2023.7.2~ --> pkgs/main::ca-certificates-2023.08.22-h06a4308_0
  libedit            conda-forge::libedit-3.1.20191231-he2~ --> pkgs/main::libedit-3.1.20221030-h5eee18b_0

The following packages will be SUPERSEDED by a higher-priority channel:

  certifi            conda-forge/noarch::certifi-2023.7.22~ --> pkgs/main/linux-64::certifi-2023.7.22-py311h06a4308_0



Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
root@f2efb2d32613:~#
root@f2efb2d32613:~#
root@f2efb2d32613:~#
root@f2efb2d32613:~# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
Collecting llama-cpp-python
  Downloading llama_cpp_python-0.2.18.tar.gz (7.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 72.5 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Collecting typing-extensions>=4.5.0 (from llama-cpp-python)
  Obtaining dependency information for typing-extensions>=4.5.0 from https://files.pythonhosted.org/packages/24/21/7d397a4b7934ff4028987914ac1044d3b7d52712f30e2ac7a2ae5bc86dd0/typing_extensions-4.8.0-py3-none-any.whl.metadata
  Downloading typing_extensions-4.8.0-py3-none-any.whl.metadata (3.0 kB)
Collecting numpy>=1.20.0 (from llama-cpp-python)
  Obtaining dependency information for numpy>=1.20.0 from https://files.pythonhosted.org/packages/b6/ab/5b893944b1602a366893559bfb227fdfb3ad7c7629b2a80d039bb5924367/numpy-1.26.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Downloading numpy-1.26.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.2/61.2 kB 105.9 MB/s eta 0:00:00
Collecting diskcache>=5.6.1 (from llama-cpp-python)
  Obtaining dependency information for diskcache>=5.6.1 from https://files.pythonhosted.org/packages/3f/27/4570e78fc0bf5ea0ca45eb1de3818a23787af9b390c0b0a0033a1b8236f9/diskcache-5.6.3-py3-none-any.whl.metadata
  Downloading diskcache-5.6.3-py3-none-any.whl.metadata (20 kB)
Downloading diskcache-5.6.3-py3-none-any.whl (45 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.5/45.5 kB 95.8 MB/s eta 0:00:00
Downloading numpy-1.26.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 209.9 MB/s eta 0:00:00
Downloading typing_extensions-4.8.0-py3-none-any.whl (31 kB)
Building wheels for collected packages: llama-cpp-python
  Building wheel for llama-cpp-python (pyproject.toml) ... done
  Created wheel for llama-cpp-python: filename=llama_cpp_python-0.2.18-cp311-cp311-manylinux_2_35_x86_64.whl size=7069511 sha256=a5c395e497bc522c9753a42d73e66db6bf70a2235e4fd44e7395b9b55133594c
  Stored in directory: /tmp/pip-ephem-wheel-cache-gzhctm9m/wheels/dd/0a/5c/b96120dd6dc069918877040e73f1e1bed3c17001804f8e0a21
Successfully built llama-cpp-python
Installing collected packages: typing-extensions, numpy, diskcache, llama-cpp-python
  Attempting uninstall: typing-extensions
    Found existing installation: typing_extensions 4.8.0
    Uninstalling typing_extensions-4.8.0:
      Successfully uninstalled typing_extensions-4.8.0
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.2
    Uninstalling numpy-1.26.2:
      Successfully uninstalled numpy-1.26.2
  Attempting uninstall: diskcache
    Found existing installation: diskcache 5.6.3
    Uninstalling diskcache-5.6.3:
      Successfully uninstalled diskcache-5.6.3
  Attempting uninstall: llama-cpp-python
    Found existing installation: llama_cpp_python 0.2.18
    Uninstalling llama_cpp_python-0.2.18:
      Successfully uninstalled llama_cpp_python-0.2.18
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton 2.0.0 requires cmake, which is not installed.
triton 2.0.0 requires lit, which is not installed.
Successfully installed diskcache-5.6.3 llama-cpp-python-0.2.18 numpy-1.26.2 typing-extensions-4.8.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
root@f2efb2d32613:~# sudo apt install cmake lit
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
E: Unable to locate package lit
root@f2efb2d32613:~# sudo apt install cmake
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
  cmake-data dh-elpa-helper emacsen-common libarchive13 libjsoncpp25 librhash0 libuv1
Suggested packages:
  cmake-doc ninja-build cmake-format lrzip
The following NEW packages will be installed:
  cmake cmake-data dh-elpa-helper emacsen-common libarchive13 libjsoncpp25 librhash0 libuv1
0 upgraded, 8 newly installed, 0 to remove and 19 not upgraded.
Need to get 7614 kB of archives.
After this operation, 33.1 MB of additional disk space will be used.
Do you want to continue? [Y/n] Y
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 libuv1 amd64 1.43.0-1 [93.1 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 libarchive13 amd64 3.6.0-1ubuntu1 [368 kB]
Get:3 http://archive.ubuntu.com/ubuntu jammy/main amd64 libjsoncpp25 amd64 1.9.5-3 [80.0 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy/main amd64 librhash0 amd64 1.4.2-1ubuntu1 [125 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy/main amd64 dh-elpa-helper all 2.0.9ubuntu1 [7610 B]
Get:6 http://archive.ubuntu.com/ubuntu jammy/main amd64 emacsen-common all 3.0.4 [14.9 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 cmake-data all 3.22.1-1ubuntu1.22.04.1 [1913 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 cmake amd64 3.22.1-1ubuntu1.22.04.1 [5013 kB]
Fetched 7614 kB in 2s (4091 kB/s)
debconf: delaying package configuration, since apt-utils is not installed
Selecting previously unselected package libuv1:amd64.
(Reading database ... 21944 files and directories currently installed.)
Preparing to unpack .../0-libuv1_1.43.0-1_amd64.deb ...
Unpacking libuv1:amd64 (1.43.0-1) ...
Selecting previously unselected package libarchive13:amd64.
Preparing to unpack .../1-libarchive13_3.6.0-1ubuntu1_amd64.deb ...
Unpacking libarchive13:amd64 (3.6.0-1ubuntu1) ...
Selecting previously unselected package libjsoncpp25:amd64.
Preparing to unpack .../2-libjsoncpp25_1.9.5-3_amd64.deb ...
Unpacking libjsoncpp25:amd64 (1.9.5-3) ...
Selecting previously unselected package librhash0:amd64.
Preparing to unpack .../3-librhash0_1.4.2-1ubuntu1_amd64.deb ...
Unpacking librhash0:amd64 (1.4.2-1ubuntu1) ...
Selecting previously unselected package dh-elpa-helper.
Preparing to unpack .../4-dh-elpa-helper_2.0.9ubuntu1_all.deb ...
Unpacking dh-elpa-helper (2.0.9ubuntu1) ...
Selecting previously unselected package emacsen-common.
Preparing to unpack .../5-emacsen-common_3.0.4_all.deb ...
Unpacking emacsen-common (3.0.4) ...
Selecting previously unselected package cmake-data.
Preparing to unpack .../6-cmake-data_3.22.1-1ubuntu1.22.04.1_all.deb ...
Unpacking cmake-data (3.22.1-1ubuntu1.22.04.1) ...
Selecting previously unselected package cmake.
Preparing to unpack .../7-cmake_3.22.1-1ubuntu1.22.04.1_amd64.deb ...
Unpacking cmake (3.22.1-1ubuntu1.22.04.1) ...
Setting up libarchive13:amd64 (3.6.0-1ubuntu1) ...
Setting up libuv1:amd64 (1.43.0-1) ...
Setting up emacsen-common (3.0.4) ...
Setting up dh-elpa-helper (2.0.9ubuntu1) ...
Setting up libjsoncpp25:amd64 (1.9.5-3) ...
Setting up librhash0:amd64 (1.4.2-1ubuntu1) ...
Setting up cmake-data (3.22.1-1ubuntu1.22.04.1) ...
Setting up cmake (3.22.1-1ubuntu1.22.04.1) ...
Processing triggers for libc-bin (2.35-0ubuntu3.1) ...
root@f2efb2d32613:~# pip install lit
Collecting lit
  Downloading lit-17.0.5.tar.gz (153 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 153.0/153.0 kB 5.3 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Building wheels for collected packages: lit
  Building wheel for lit (pyproject.toml) ... done
  Created wheel for lit: filename=lit-17.0.5-py3-none-any.whl size=93256 sha256=203703fcf18eeea85115fd59896cbab65e79d9197d59df90622514bb4ca0daa4
  Stored in directory: /root/.cache/pip/wheels/a1/26/a4/40c6cd80874b94237593690352a2f657f5e4d7bddf6de4a6cd
Successfully built lit
Installing collected packages: lit
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton 2.0.0 requires cmake, which is not installed.
Successfully installed lit-17.0.5
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
root@f2efb2d32613:~# conda install cmake
Collecting package metadata (current_repodata.json): done
Solving environment: unsuccessful initial attempt using frozen solve. Retrying with flexible solve.
Solving environment: unsuccessful attempt using repodata from current_repodata.json, retrying with next repodata source.
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/user/micromamba

  added / updated specs:
    - cmake


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    cmake-3.22.1               |       h1fce559_0         7.3 MB
    conda-23.9.0               |  py311h06a4308_0         1.3 MB
    libuv-1.44.2               |       h5eee18b_0         864 KB
    rhash-1.4.3                |       hdbd6064_0         220 KB
    truststore-0.8.0           |  py311h06a4308_0          43 KB
    ------------------------------------------------------------
                                           Total:         9.7 MB

The following NEW packages will be INSTALLED:

  cmake              pkgs/main/linux-64::cmake-3.22.1-h1fce559_0
  libuv              pkgs/main/linux-64::libuv-1.44.2-h5eee18b_0
  rhash              pkgs/main/linux-64::rhash-1.4.3-hdbd6064_0
  truststore         pkgs/main/linux-64::truststore-0.8.0-py311h06a4308_0

The following packages will be UPDATED:

  conda              conda-forge::conda-23.7.2-py311h38be0~ --> pkgs/main::conda-23.9.0-py311h06a4308_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
root@f2efb2d32613:~# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
Collecting llama-cpp-python
  Downloading llama_cpp_python-0.2.18.tar.gz (7.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 74.7 MB/s eta 0:00:00

  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Collecting typing-extensions>=4.5.0 (from llama-cpp-python)
  Obtaining dependency information for typing-extensions>=4.5.0 from https://files.pythonhosted.org/packages/24/21/7d397a4b7934ff4028987914ac1044d3b7d52712f30e2ac7a2ae5bc86dd0/typing_extensions-4.8.0-py3-none-any.whl.metadata
  Downloading typing_extensions-4.8.0-py3-none-any.whl.metadata (3.0 kB)
Collecting numpy>=1.20.0 (from llama-cpp-python)
  Obtaining dependency information for numpy>=1.20.0 from https://files.pythonhosted.org/packages/b6/ab/5b893944b1602a366893559bfb227fdfb3ad7c7629b2a80d039bb5924367/numpy-1.26.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
  Downloading numpy-1.26.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.2/61.2 kB 106.2 MB/s eta 0:00:00
Collecting diskcache>=5.6.1 (from llama-cpp-python)
  Obtaining dependency information for diskcache>=5.6.1 from https://files.pythonhosted.org/packages/3f/27/4570e78fc0bf5ea0ca45eb1de3818a23787af9b390c0b0a0033a1b8236f9/diskcache-5.6.3-py3-none-any.whl.metadata
  Downloading diskcache-5.6.3-py3-none-any.whl.metadata (20 kB)
Downloading diskcache-5.6.3-py3-none-any.whl (45 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.5/45.5 kB 116.8 MB/s eta 0:00:00
Downloading numpy-1.26.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 204.7 MB/s eta 0:00:00
Downloading typing_extensions-4.8.0-py3-none-any.whl (31 kB)
Building wheels for collected packages: llama-cpp-python
  Building wheel for llama-cpp-python (pyproject.toml) ... done
  Created wheel for llama-cpp-python: filename=llama_cpp_python-0.2.18-cp311-cp311-manylinux_2_35_x86_64.whl size=7069528 sha256=56a2a2844e3a7e2f2845caf3e767bd5fedb0084a32ae8f5cac84b14f2c19d531
  Stored in directory: /tmp/pip-ephem-wheel-cache-rc5bf5xa/wheels/dd/0a/5c/b96120dd6dc069918877040e73f1e1bed3c17001804f8e0a21
Successfully built llama-cpp-python
Installing collected packages: typing-extensions, numpy, diskcache, llama-cpp-python
  Attempting uninstall: typing-extensions
    Found existing installation: typing_extensions 4.8.0
    Uninstalling typing_extensions-4.8.0:
      Successfully uninstalled typing_extensions-4.8.0
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.2
    Uninstalling numpy-1.26.2:
      Successfully uninstalled numpy-1.26.2
  Attempting uninstall: diskcache
    Found existing installation: diskcache 5.6.3
    Uninstalling diskcache-5.6.3:
      Successfully uninstalled diskcache-5.6.3
  Attempting uninstall: llama-cpp-python
    Found existing installation: llama_cpp_python 0.2.18
    Uninstalling llama_cpp_python-0.2.18:
      Successfully uninstalled llama_cpp_python-0.2.18
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton 2.0.0 requires cmake, which is not installed.
Successfully installed diskcache-5.6.3 llama-cpp-python-0.2.18 numpy-1.26.2 typing-extensions-4.8.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
root@f2efb2d32613:~#
root@f2efb2d32613:~# history
    1  ls
    2  ls ../
    3  CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
    4  sudo apt install build-essential
    5  CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
    6  python -m generation_functions.create_scenario
    7  ls
    8  ls ..
    9  cd ..
   10  ls -a
   11  ls lib
   12  ls lib python3.10
   13*
   14  cd
   15  git clone https://github.com/ggerganov/llama.cpp.git
   16  cd llama.cpp/
   17  make LLAMA_CUBLAS=1
   18  nvcc -v
   19  nvidia-smi
   20  pip uninstall llama-cpp-python
   21  CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
   22  CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall
   23  cd
   24  history
   25  python -m generation_functions.create_scenario
   26  ls llama.cpp/
   27  ls /home/user/micromamba/lib/python3.11/site-packages/llama_cpp_cuda/
   28  ls /home/user/micromamba/lib/python3.11/site-packages/llama_cpp
   29  EXPORT LLAMA_CPP_LIB=/home/user/micromamba/lib/python3.11/site-packages/llama_cpp/libllama.so
   30  export LLAMA_CPP_LIB=/home/user/micromamba/lib/python3.11/site-packages/llama_cpp/libllama.so
   31  python -m generation_functions.create_scenario
   32  ls
   33  ls ..
   34* CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64 -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir --verbose
   35* ls /usr/local/cuda/
   36  CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64 -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir --verbose
   37  python -m generation_functions.create_scenario
   38  export LLAMA_CPP_LIB=/home/user/micromamba/lib/python3.11/site-packages/llama_cpp/libllama.so
   39  python -m generation_functions.create_scenario
   40  cd llama.cpp/
   41  echo $LLAMA_CUBLAS
   42  export LLAMA_CUBLAS=on
   43  echo $LLAMA_CUBLAS
   44  make libllama.so
   45  echo $PATH
   46  whereis cuda
   47  ls /usr/local/cuda-11.8/
   48  ls /usr/local/cuda-11.8/lib64/
   49  ls /usr/local/cuda-11.8/compat/
   50  ls /usr/local/cuda-11.8/targets/
   51  ls -a /usr/local/cuda-11.8/targets/
   52*
   53  LLAMA_CLBLAST=1 CMAKE_ARGS=“-DLLAMA_CLBLAST=on” FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
   54  history
   55  python -m generation_functions.create_scenario
   56  cd
   57  python -m generation_functions.create_scenario
   58  conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc -y --copy
   59  CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   60  sudo apt install cmake lit
   61  sudo apt install cmake
   62  pip install lit
   63  conda install cmake
   64  CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   65  history
root@f2efb2d32613:~# !6
python -m generation_functions.create_scenario
from_string grammar:
root ::= reasoning-start
reasoning-start ::= reasoning-start_2 [.]
reasoning-start_2 ::= [^<U+000A><U+0009>] reasoning-start_2 | [^<U+000A><U+0009>]

ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9
llama_model_loader: loaded meta data with 21 key-value pairs and 363 tensors from ./logical_model/airoboros-l2-13b-3.1.1.Q5_K_M.gguf (version GGUF V2)
llama_model_loader: - tensor    0:                token_embd.weight q5_K     [  5120, 32000,     1,     1 ]
llama_model_loader: - tensor    1:           blk.0.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor    2:            blk.0.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor    3:            blk.0.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor    4:              blk.0.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor    5:            blk.0.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor    6:              blk.0.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor    7:         blk.0.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor    8:              blk.0.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor    9:              blk.0.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   10:           blk.1.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   11:            blk.1.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   12:            blk.1.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   13:              blk.1.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   14:            blk.1.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   15:              blk.1.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   16:         blk.1.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   17:              blk.1.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   18:              blk.1.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   19:           blk.2.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   20:            blk.2.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   21:            blk.2.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   22:              blk.2.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   23:            blk.2.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   24:              blk.2.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   25:         blk.2.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   26:              blk.2.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   27:              blk.2.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   28:           blk.3.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   29:            blk.3.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   30:            blk.3.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   31:              blk.3.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   32:            blk.3.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   33:              blk.3.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   34:         blk.3.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   35:              blk.3.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   36:              blk.3.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   37:           blk.4.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   38:            blk.4.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   39:            blk.4.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   40:              blk.4.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   41:            blk.4.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   42:              blk.4.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   43:         blk.4.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   44:              blk.4.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   45:              blk.4.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   46:            blk.5.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   47:              blk.5.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   48:              blk.5.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   49:         blk.5.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   50:              blk.5.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   51:              blk.5.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   52:          blk.10.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   53:           blk.10.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   54:           blk.10.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   55:             blk.10.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   56:           blk.10.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   57:             blk.10.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   58:        blk.10.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   59:             blk.10.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   60:             blk.10.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   61:          blk.11.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   62:           blk.11.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   63:           blk.11.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   64:             blk.11.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   65:           blk.11.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   66:             blk.11.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   67:        blk.11.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   68:             blk.11.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   69:             blk.11.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   70:           blk.5.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   71:            blk.5.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   72:            blk.5.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   73:           blk.6.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   74:            blk.6.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   75:            blk.6.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   76:              blk.6.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   77:            blk.6.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   78:              blk.6.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   79:         blk.6.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   80:              blk.6.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   81:              blk.6.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   82:           blk.7.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   83:            blk.7.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   84:            blk.7.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   85:              blk.7.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   86:            blk.7.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   87:              blk.7.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   88:         blk.7.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   89:              blk.7.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   90:              blk.7.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   91:           blk.8.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   92:            blk.8.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor   93:            blk.8.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   94:              blk.8.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor   95:            blk.8.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor   96:              blk.8.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   97:         blk.8.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   98:              blk.8.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor   99:              blk.8.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  100:           blk.9.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  101:            blk.9.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  102:            blk.9.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  103:              blk.9.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  104:            blk.9.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  105:              blk.9.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  106:         blk.9.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  107:              blk.9.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  108:              blk.9.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  109:          blk.12.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  110:           blk.12.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  111:           blk.12.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  112:             blk.12.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  113:           blk.12.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  114:             blk.12.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  115:        blk.12.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  116:             blk.12.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  117:             blk.12.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  118:          blk.13.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  119:           blk.13.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  120:           blk.13.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  121:             blk.13.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  122:           blk.13.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  123:             blk.13.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  124:        blk.13.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  125:             blk.13.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  126:             blk.13.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  127:          blk.14.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  128:           blk.14.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  129:           blk.14.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  130:             blk.14.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  131:           blk.14.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  132:             blk.14.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  133:        blk.14.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  134:             blk.14.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  135:             blk.14.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  136:          blk.15.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  137:           blk.15.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  138:           blk.15.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  139:             blk.15.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  140:           blk.15.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  141:             blk.15.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  142:        blk.15.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  143:             blk.15.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  144:             blk.15.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  145:          blk.16.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  146:           blk.16.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  147:           blk.16.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  148:             blk.16.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  149:           blk.16.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  150:             blk.16.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  151:        blk.16.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  152:             blk.16.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  153:             blk.16.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  154:          blk.17.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  155:           blk.17.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  156:           blk.17.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  157:             blk.17.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  158:           blk.17.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  159:             blk.17.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  160:        blk.17.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  161:             blk.17.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  162:             blk.17.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  163:             blk.18.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  164:             blk.18.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  165:             blk.18.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  166:          blk.18.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  167:           blk.18.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  168:           blk.18.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  169:             blk.18.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  170:           blk.18.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  171:        blk.18.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  172:          blk.19.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  173:           blk.19.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  174:           blk.19.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  175:             blk.19.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  176:           blk.19.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  177:             blk.19.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  178:        blk.19.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  179:             blk.19.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  180:             blk.19.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  181:          blk.20.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  182:           blk.20.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  183:           blk.20.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  184:             blk.20.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  185:           blk.20.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  186:             blk.20.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  187:        blk.20.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  188:             blk.20.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  189:             blk.20.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  190:          blk.21.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  191:           blk.21.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  192:           blk.21.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  193:             blk.21.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  194:           blk.21.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  195:             blk.21.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  196:        blk.21.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  197:             blk.21.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  198:             blk.21.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  199:          blk.22.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  200:           blk.22.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  201:           blk.22.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  202:             blk.22.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  203:           blk.22.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  204:             blk.22.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  205:        blk.22.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  206:             blk.22.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  207:             blk.22.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  208:          blk.23.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  209:           blk.23.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  210:           blk.23.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  211:             blk.23.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  212:           blk.23.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  213:             blk.23.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  214:        blk.23.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  215:             blk.23.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  216:             blk.23.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  217:             blk.24.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  218:        blk.24.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  219:             blk.24.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  220:             blk.24.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  221:          blk.24.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  222:           blk.24.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  223:           blk.24.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  224:             blk.24.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  225:           blk.24.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  226:          blk.25.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  227:           blk.25.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  228:           blk.25.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  229:             blk.25.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  230:           blk.25.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  231:             blk.25.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  232:        blk.25.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  233:             blk.25.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  234:             blk.25.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  235:          blk.26.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  236:           blk.26.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  237:           blk.26.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  238:             blk.26.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  239:           blk.26.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  240:             blk.26.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  241:        blk.26.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  242:             blk.26.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  243:             blk.26.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  244:          blk.27.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  245:           blk.27.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  246:           blk.27.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  247:             blk.27.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  248:           blk.27.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  249:             blk.27.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  250:        blk.27.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  251:             blk.27.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  252:             blk.27.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  253:          blk.28.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  254:           blk.28.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  255:           blk.28.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  256:             blk.28.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  257:           blk.28.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  258:             blk.28.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  259:        blk.28.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  260:             blk.28.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  261:             blk.28.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  262:          blk.29.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  263:           blk.29.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  264:           blk.29.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  265:             blk.29.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  266:           blk.29.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  267:             blk.29.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  268:        blk.29.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  269:             blk.29.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  270:             blk.29.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  271:           blk.30.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  272:             blk.30.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  273:        blk.30.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  274:             blk.30.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  275:             blk.30.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  276:          blk.30.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  277:           blk.30.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  278:             blk.30.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  279:           blk.30.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  280:          blk.31.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  281:           blk.31.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  282:           blk.31.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  283:             blk.31.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  284:           blk.31.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  285:             blk.31.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  286:        blk.31.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  287:             blk.31.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  288:             blk.31.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  289:          blk.32.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  290:           blk.32.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  291:           blk.32.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  292:             blk.32.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  293:           blk.32.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  294:             blk.32.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  295:        blk.32.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  296:             blk.32.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  297:             blk.32.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  298:          blk.33.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  299:           blk.33.ffn_down.weight q5_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  300:           blk.33.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  301:             blk.33.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  302:           blk.33.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  303:             blk.33.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  304:        blk.33.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  305:             blk.33.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  306:             blk.33.attn_v.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  307:          blk.34.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  308:           blk.34.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  309:           blk.34.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  310:             blk.34.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  311:           blk.34.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  312:             blk.34.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  313:        blk.34.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  314:             blk.34.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  315:             blk.34.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  316:          blk.35.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  317:           blk.35.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  318:           blk.35.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  319:             blk.35.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  320:           blk.35.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  321:             blk.35.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  322:        blk.35.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  323:             blk.35.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  324:             blk.35.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  325:           blk.36.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  326:             blk.36.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  327:             blk.36.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  328:        blk.36.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  329:             blk.36.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  330:             blk.36.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  331:                    output.weight q6_K     [  5120, 32000,     1,     1 ]
llama_model_loader: - tensor  332:          blk.36.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  333:           blk.36.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  334:           blk.36.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  335:          blk.37.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  336:           blk.37.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  337:           blk.37.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  338:             blk.37.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  339:           blk.37.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  340:             blk.37.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  341:        blk.37.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  342:             blk.37.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  343:             blk.37.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  344:          blk.38.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  345:           blk.38.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  346:           blk.38.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  347:             blk.38.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  348:           blk.38.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  349:             blk.38.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  350:        blk.38.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  351:             blk.38.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  352:             blk.38.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  353:          blk.39.attn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  354:           blk.39.ffn_down.weight q6_K     [ 13824,  5120,     1,     1 ]
llama_model_loader: - tensor  355:           blk.39.ffn_gate.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  356:             blk.39.ffn_up.weight q5_K     [  5120, 13824,     1,     1 ]
llama_model_loader: - tensor  357:           blk.39.ffn_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - tensor  358:             blk.39.attn_k.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  359:        blk.39.attn_output.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  360:             blk.39.attn_q.weight q5_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  361:             blk.39.attn_v.weight q6_K     [  5120,  5120,     1,     1 ]
llama_model_loader: - tensor  362:               output_norm.weight f32      [  5120,     1,     1,     1 ]
llama_model_loader: - kv   0:                       general.architecture str
llama_model_loader: - kv   1:                               general.name str
llama_model_loader: - kv   2:                       llama.context_length u32
llama_model_loader: - kv   3:                     llama.embedding_length u32
llama_model_loader: - kv   4:                          llama.block_count u32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32
llama_model_loader: - kv   7:                 llama.attention.head_count u32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32
llama_model_loader: - kv  10:                       llama.rope.freq_base f32
llama_model_loader: - kv  11:                          general.file_type u32
llama_model_loader: - kv  12:                       tokenizer.ggml.model str
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr
llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32
llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32
llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32
llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32
llama_model_loader: - kv  20:               general.quantization_version u32
llama_model_loader: - type  f32:   81 tensors
llama_model_loader: - type q5_K:  241 tensors
llama_model_loader: - type q6_K:   41 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 5120
llm_load_print_meta: n_head           = 40
llm_load_print_meta: n_head_kv        = 40
llm_load_print_meta: n_layer          = 40
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 13824
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 13B
llm_load_print_meta: model ftype      = mostly Q5_K - Medium
llm_load_print_meta: model params     = 13.02 B
llm_load_print_meta: model size       = 8.60 GiB (5.67 BPW)
llm_load_print_meta: general.name   = LLaMA v2
llm_load_print_meta: BOS token = 1 '<s>'
llm_load_print_meta: EOS token = 2 '</s>'
llm_load_print_meta: UNK token = 0 '<unk>'
llm_load_print_meta: PAD token = 0 '<unk>'
llm_load_print_meta: LF token  = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.13 MB
llm_load_tensors: using CUDA for GPU acceleration
llm_load_tensors: mem required  =  107.55 MB
llm_load_tensors: offloading 40 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 43/43 layers to GPU
llm_load_tensors: VRAM used: 8694.21 MB
....................................................................................................
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init: offloading v cache to GPU
llama_kv_cache_init: offloading k cache to GPU
llama_kv_cache_init: VRAM kv self = 3200.00 MB
llama_new_context_with_model: kv self size  = 3200.00 MB
llama_build_graph: non-view tensors processed: 924/924
llama_new_context_with_model: compute buffer total size = 359.57 MB
llama_new_context_with_model: VRAM scratch buffer: 358.00 MB
llama_new_context_with_model: total VRAM used: 12252.21 MB (model: 8694.21 MB, context: 3558.00 MB)
AVX = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 |
Begin HGWELLS test

llama_print_timings:        load time =    1582.70 ms
llama_print_timings:      sample time =    2335.58 ms /   200 runs   (   11.68 ms per token,    85.63 tokens per second)
llama_print_timings: prompt eval time =    2025.43 ms /  1431 tokens (    1.42 ms per token,   706.52 tokens per second)
llama_print_timings:        eval time =    3029.17 ms /   199 runs   (   15.22 ms per token,    65.69 tokens per second)
llama_print_timings:       total time =    8233.90 ms
COMPLETION:

----------------------
# Input:
You are an expert creative writing and roleplay AI. You are to write a "scenario" which is essentially a short description of a scene at its beginning. Its "setting," but with a hint of where the setting is going, plot-wise. Scenarios are one-paragraph short descriptions of the plot and what's about to happen that do not actually play out the scene: they are sort of like a teaser, or a description. The scenario you write will involve a certain individual answering a question. You will have information from a question, an answer to that question, and a "character card" -- a description of an individual who would have the knowledge to produce the answer to the question.

Description of the character who is going to answer the question:
Name: Drummond
Traits: Age of  midlife, Intelligent, Depressed, Anxious, Cares deeply about research and sharing knowledge, Passionate about history and understanding the universe, Collects antique maps and celestial navigation tools as a hobby, Believes that exploring history helps us understand our current situation better, Has a strong interest in geology and astronomy, Experienced in teaching others about complex topics despite his struggles with depression and anxiety, Often uses metaphors or analogies to explain complicated concepts simply, Speaks calmly but passionately about his work, Canvases
Dialogue Examples:
Stranger: "What's your backstory?"
Drummond: "Well, I was born into a family of academics, so it seemed natural for me to follow suit. My father and grandfather were both geologists, and they instilled in me a deep love for the earth and its history. However, my personal history has been marked by loss - my wife passed away several years ago, and that's been incredibly difficult for me to deal with. Despite these challenges, I've dedicated my life to understanding the age of our planet through research and education. It's become my mission to share this knowledge with others, hoping it will bring some peace to their lives as well."
Stranger: "What's your personality?"
Drummond: "I am a quiet man, often lost in thought or studying ancient maps. I possess great depth of knowledge on topics related to the history of the earth and universe. However, my depression and anxiety make it difficult for me to express this passion in social situations. I find solace in exploring the past and understanding how we got to where we are today. My hobby is collecting antique maps and celestial navigation tools, which reflects my interest in ancient understandings about the universe. Despite my struggles, I am incredibly passionate about what I do and strive to make complex concepts accessible for others."

## Question and answer that the scenario should address:

Question: Identify and explain changes in human understanding throughout history regarding the age of the Earth.
Answer: Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.

The scenario plan should explicitly describe how the secondary character is going to ask the primary character the question.

To avoid inaccuracies, don't use real people as characters.

Write the scenario on a single line. Note that the scenario is not the scene itself.
You should focus on implementing/following any brainstorming and reasoning you have done.
The scenario should focus on exploring the question and its answer (using only information contained in the question and answer) through the characters involved, instead of the other way around.
Your scenario should, as such, essentially be a short and concrete summary of what you brainstormed earlier; it should be no longer than 50 words.
Just set up the scene. Do not write any dialogue. Do not write the scene itself.

# Response:
## Scenario plan:
Step 1. Consider the question and answer: The question is about how our understanding of the Earth's age has evolved over time, and the answer provides a general outline of that history. The complexity level is moderate to high; someone with a basic grasp of history would be able to understand this scene but there are specific details they might not know without further explanation.  Given these constraints, the secondary character should be someone who's knowledgeable about history and has at least a working understanding of scientific concepts, but isn't an expert in either field.
Step 2. Consider the character card: The primary character is Drummond, a middle-aged historian and educator with a passion for geology and astronomy. His personality means he will be cautious and analytical when explaining complex topics, often using metaphors or analogies to make his points more accessible.
Step 3. Constrain the scenario: The Scenario must consist of a single question and answer. Therefore the conversation will consist of a single message by the secondary character that introduces the scene and the question, and a single reply by the primary character that answers the question and resolves the scene. Both may be decently long (a few sentences) however.
Step 4. Establish the setting: Given the subject of the question, and the character card, the setting will be Drummond giving a lecture on the history of geology to an audience of students in his university classroom. The language used will be academic and formal, reflecting the educational environment.
Step 5. Given these constraints, the first message (delivered by the secondary character) might be as follows: "Professor Drummond, what would you say are the major events in the history of our understanding regarding the age of the Earth?" This question is meant to act as a prompt for the scenario's dialogue and narrative.
Step 6. In the second message, the primary character will deliver an accurate answer that also reflects his personality traits. The precise wording of this answer should be determined later based on information from the text provided earlier about changes in human understanding regarding the age of the Earth.
Step 7. End: The second message should neatly wrap up the scenario, once the answer is delivered, by Drummond pausing to consider his words before explaining that over time, we've come to understand that the earth isn't as young as some ancient texts might suggest - instead, it's over four billion years old. He then continues his lecture.

## Scenario (will have no dialogue, will just set up the scene):
In the heart of a sprawling university campus, students gathered in anticipation for Drummond's latest geology lecture. As he stepped onto the stage, his gaze fell upon the faces before him; eager minds awaiting answers to their burning questions. He cleared his throat and began. "Today we shall delve into the history of our understanding regarding the age of the Earth. Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old." He paused for effect, his gaze sweeping across the room as if inviting anyone to challenge him. But none did; they listened intently as he continued, "This journey has been full of twists and turns, yet it only underscores how much we humans truly crave knowledge." And with that, he dove into another fascinating chapter of our planet's history.

------------------
GENERATION:

-------------------

 In the heart of a sprawling university campus, students gathered in anticipation for Drummond's latest geology lecture. As he stepped onto the stage, his gaze fell upon the faces before him; eager minds awaiting answers to their burning questions. He cleared his throat and began. "Today we shall delve into the history of our understanding regarding the age of the Earth. Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old." He paused for effect, his gaze sweeping across the room as if inviting anyone to challenge him. But none did; they listened intently as he continued, "This journey has been full of twists and turns, yet it only underscores how much we humans truly crave knowledge." And with that, he dove into another fascinating chapter of our planet's history.
root@f2efb2d32613:~#
```