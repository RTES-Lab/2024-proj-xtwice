Windows 환경 설치 가이드
==============================================

| 이 문서는 Windows 환경에서 필요한 소프트웨어 설치 방법을 안내한다. 주요 설치 항목으로는 NVIDIA 드라이버, CUDA, Anaconda 가상환경, 그리고 필수 Python 라이브러리가 포함된다.

1. `NVIDIA 드라이버 및 CUDA 설치`_

2. `Anaconda 설치`_

3. `가상 환경 생성 및 사용`_

4. `라이브러리 설치 및 입력 파일 저장`_

NVIDIA 드라이버 및 CUDA 설치
****************************************

   +) **NVIDIA 드라이버가 450.80.02보다 높으며 CUDA 버전이 11.2~11.8인 경우** 드라이버와 CUDA 설치를 생략하고 `Anaconda 설치`_ 과정부터 진행한다.

   **(1) NVIDIA 드라이버 설치**

   
   터미널에서 ``nvidia-smi`` 명령어를 입력하여 NVIDIA 드라이버 버전이 450.80.02 이상인지 확인한다. 
   450.80.02 버전보다 낮은 경우 제어판에서 NVIDIA 관련 프로그램을 모두 삭제한 후, `NVIDIA 드라이버 다운로드 페이지 <https://www.nvidia.co.kr/Download/index.aspx?lang=kr>`_ 에서 GPU 모듈에 맞는 드라이버를 설치한다.

   .. image:: img/cuda_install4.png

   |
   NVIDIA 드라이버 버전이 450.80.02 이상이면 ``nvcc --version`` 명령어를 입력하여 CUDA 버전을 확인한다. 
   명령어가 정상적으로 실행되지 않거나 CUDA의 버전이 11.2~11.8이 아닌 경우 역시 제어판에서 NVIDIA 관련 프로그램을 모두 삭제한 후, 드라이버와 CUDA를 설치한다.

   .. image:: img/cuda_install5.png

   |
   **(2) CUDA 설치**

   `GPU 아키텍처별 CUDA 버전 지원 <https://www.wikiwand.com/en/CUDA#GPUs_supported>`_ 페이지에서 1) GPU 모델에 따른 **Compute Capability** 와 2) **CUDA SDK 버전** 을 확인한다.
   아래 예시는 NVIDIA RTX A4000 GPU의 Compute Capability와 CUDA SDK 버전을 확인하는 예시이다.


   1\) GPU 모델에 따른 **Compute Capability** 를 확인

   .. image:: img/cuda_install1.png

   | 
   | 2\) **CUDA SDK 버전** 확인

   .. image:: img/cuda_install2.png

   확인한 SDK 버전 범위에서 프로그램과 호환 가능한 **CUDA SDK 버전(11.2~11.8)** 을 설치해야 한다.
   `CUDA 다운로드 페이지 <https://developer.nvidia.com/cuda-toolkit-archive>`_ 에서 버전에 맞는 CUDA를 설치한다.

   .. image:: img/cuda_install3.png

   |
   | CUDA를 설치한 후 ``nvcc --version`` 명령어를 입력하여 11.2~11.8 버전의 CUDA가 설치되었는 지 확인한다.

   .. image:: img/cuda_install5.png

Anaconda 설치
****************************************

   `Anaconda 공식 홈페이지 <https://www.anaconda.com/download>`_ 에서 Anaconda를 다운로드한다.

   .. image:: img/anaconda_page.png

   |
   | 설치 파일을 더블클릭하여 설치를 진행한다.

   .. warning::
      설치 경로에 한글이 포함되어 있으면 안 됨.

   .. image:: img/anaconda_install.png


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