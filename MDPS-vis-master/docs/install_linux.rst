Linux 환경 설치 가이드
==============================================


이 문서는 리눅스 환경에서 필요한 소프트웨어 설치 방법을 안내한다. 주요 설치 항목으로는 NVIDIA 드라이버, CUDA, Anaconda 가상환경, 그리고 필수 Python 라이브러리가 포함된다.

1. `NVIDIA 드라이버 및 CUDA 설치`_

2. `Anaconda 설치`_

3. `가상 환경 생성 및 사용`_

4. `라이브러리 설치 및 입력 파일 저장`_

NVIDIA 드라이버 및 CUDA 설치
****************************************

NVIDIA 드라이버 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **(1) NVIDIA 드라이버 설치 여부 확인**

   리눅스 터미널에 아래 명령어를 입력하여 NVIDIA 드라이버가 설치되어 있는지 확인한다.

   .. code-block:: bash
     
       $ nvidia-smi

   **`nvidia-smi`** 명령어는 NVIDIA System Management Interface를 실행시키며, GPU의 상태, 현재 사용 중인 드라이버 버전, 메모리 사용량 등의 정보를 제공한다. 이미 NVIDIA 드라이버가 설치되어 있다면, 이 명령어를 실행했을 때 GPU에 관한 상세 정보가 출력된다. 

   .. note:: 본 프로그램은 CUDA 11.2~11.8 사이 버전을 사용하므로 NVIDIA 드라이버 450.80.02 이상 버전을 사용해야 한다.

   .. image:: img/linux_guide1.png

   NVIDIA 드라이버가 설치되어 있지 않거나 450.80.02 아래 버전의 NVIDIA 드라이버가 설치되어 있는 경우 아래의 설치 과정을 진행해야 한다. 

   **(2) 기존 NVIDIA 드라이버 및 CUDA 삭제**

   .. code-block:: bash

      $ sudo apt-get purge nvidia*
      $ sudo apt-get autoremove
      $ sudo apt-get autoclean
      $ sudo rm -rf /usr/local/cuda*
      $ sudo apt update
      $ sudo apt upgrade -y

   **(3) NVIDIA 드라이버 설치**

   다음 명령어를 입력하여 설치 가능한 NVIDIA 드라이버 목록을 확인한다.

   .. code-block:: bash

       ubuntu-drivers devices

   출력된 드라이버 목록 중 450.80.02 이상 버전의 드라이버를 설치한다. 

   .. code-block:: bash

      // 예) 525 버전의 드라이버 설치
      $ sudo apt install nvidia-driver-525  

   **(4) 추가 패키지 설치 및 재부팅**

   .. code-block:: bash

      $ sudo apt-get install dkms nvidia-modprobe
      $ sudo apt update
      $ sudo apt upgrade
      $ sudo reboot


CUDA 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. note:: 
      CUDA는 11.2 ~ 11.8 버전으로 설치해야 한다.

   **(1) CUDA Toolkit Archive 웹사이트 접속**

   https://developer.nvidia.com/cuda-toolkit-archive

   **(2) CUDA 버전 선택**

   11.2 ~ 11.8 사이의 CUDA 버전의 링크를 클릭하여 해당 버전의 다운로드 페이지로 이동한다.

   **(3) 운영 체제 및 설치 패키지 선택**

   다운로드 페이지에서, 운영 체제로 Linux를 선택하고, 아키텍처(x86_64, arm64 등), 배포판(Ubuntu, CentOS 등), 버전 및 설치 유형을 사용자 환경에 맞게 결정한다.

   .. image:: img/linux_guide2.png

   **(4) 다운로드 명령어 실행**

   선택한 설치 패키지에 따라, 페이지에서 제공하는 명령어를 복사한다.

   리눅스 터미널을 열고 복사한 명령어를 붙여넣어 실행하여 설치를 완료한다.

   .. image:: img/linux_guide3.png 

   .. code-block:: bash

      $ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
      $ sudo sh cuda_11.8.0_520.61.05_linux.run

   **(5) 환경 변수 설정**

   설치가 완료된 후, CUDA를 사용하기 위해 환경 변수를 설정해야 한다. **`~/.bashrc`** 파일을 수정하여 다음 환경 변수를 추가한다.

   .. code-block::

       export PATH=/usr/local/cuda-[설치한 CUDA 버전]/bin:$PATH
       export LD_LIBRARY_PATH=/usr/local/cuda-[설치한 CUDA 버전]/lib64:$LD_LIBRARY_PATH

   .. image:: img/linux_guide4.png 
   
   |
   .. note::
      예를 들어, CUDA 11.8를 설치했다면 `[설치한 CUDA 버전]`을 `11.8`로 대체한다.

   **(6) CUDA 설치 확인**

   환경 변수 설정을 완료한 후에는 변경사항을 적용하기 위해 **`source ~/.bashrc`** 명령을 실행하거나, 새 터미널 세션을 시작한다.

   설치된 CUDA의 버전을 확인하기 위해 다음 명령어를 터미널에서 실행한다.

   .. code-block:: bash

      $ nvcc --version

   이 명령어는 설치된 CUDA Toolkit의 버전 정보를 출력한다. 출력된 정보에는 CUDA의 버전, 빌드 번호, 릴리스 날짜 등이 포함되어 있으며, 이를 통해 시스템에 CUDA가 성공적으로 설치된 것을 확인할 수 있다.

Anaconda 설치
****************************************

   **(1) Anaconda 설치 파일 다운로드**

   `Anaconda 공식 웹사이트 <https://www.anaconda.com/download>`_ 에서 Linux용 아나콘다 설치 파일을 찾아 다운로드한다.

   **(2) 설치 스크립트 실행**

   터미널을 열고 다운로드한 설치 스크립트가 있는 디렉토리로 이동한 후 다음 명령어를 실행하여 설치 스크립트에 실행 권한을 부여한다.

   .. code-block:: bash

      $ chmod +x Anaconda3-202X.XX-Linux-x86_64.sh

   이후 스크립트를 실행하여 아나콘다를 설치한다.

   .. code-block:: bash

      $ ./Anaconda3-202X.XX-Linux-x86_64.sh

   **(3) 환경 변수 설정**

   **`~/.bashrc`** 파일에 아나콘다의 bin 디렉토리를 PATH에 추가한다.

   .. code-block::

      export PATH=/home/[사용자명]/anaconda3/bin:$PATH

   **(4) Anaconda 설치 확인**

   환경 변수 설정을 완료한 후에는 변경사항을 적용하기 위해 **`source ~/.bashrc`** 명령을 실행하거나, 새 터미널 세션을 시작한다.

   설치가 정상적으로 완료되었는지 확인하기 위해, 터미널에서 다음 명령어를 실행한다.

   .. code-block:: bash

      $ conda --version


가상 환경 생성 및 사용
****************************************

   **(1) 가상 환경 생성**

   Python 3.11을 사용하는 가상 환경을 생성하기 위해, 터미널에서 다음 명령어를 실행한다.

   .. code-block:: bash

      $ conda create -n [가상환경이름] python=3.11.0

   **(2) 가상 환경 활성화**

   생성한 가상 환경을 활성화하기 위해, 다음 명령어를 실행한다.

   .. code-block:: bash

      // 가상 환경 실행 (비활성화: conda deactivate [가상환경이름])
      $ conda activate [가상환경이름]


라이브러리 설치 및 입력 파일 저장
****************************************

   **(1) 라이브러리 설치**

   가상환경 상에서 MDPS-VIS 디렉토리로 이동한 뒤 다음 명령어를 실행하여 가상환경에 필요한 라이브러리를 설치한다.

   .. code-block:: bash

      $ pip install -r requirements.txt

   **(2) 입력 파일 저장**

   패키지 설치가 완료되면, 본 프로그램을 가상 환경 내에서 사용할 수 있다.

   예시 코드를 실행하기 위해 MIDPS-VIS 폴더 안에 input 폴더를 만든 뒤, 1.0_old_dark.mp4 파일을 넣고 `프로그램 실행 가이드 <./tutorial.html>`_ 에 따라 프로그램을 실행한다.