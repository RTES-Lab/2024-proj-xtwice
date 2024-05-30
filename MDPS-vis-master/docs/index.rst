.. MDPS visualization documentation master file, created by
   sphinx-quickstart on Tue Feb 13 12:30:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MDPS visualization software
==============================================

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

본 프로그램은 아래 그림과 같이 마커가 부착된 R-MDPS 전면부 촬영 영상으로부터 각 마커가 위치한 지점의 변위를 추출한다.

.. image:: img/mdps_view.png

정확한 변위 추출을 위해 phase-based magnification (PBM) 영상확대 기법을 사용할 수 있고, 
영상확대 연산은 GPU를 통해 가속화된다.
프로그램의 요구 사양은 다음과 같다.

* OS: Windows 10 이상 또는 Ubuntu 18.04 이상
* CPU: 인텔 i7 이상급 프로세서
* RAM: 16 GB 이상급
* GPU: 8GB 이상 메모리를 가진 NVIDIA GPU
* CUDA 버전: 11.2~11.8
* Python 버전: 3.11.0
* Python 라이브러리 및 라이브러리별 버전: requirements.txt 참조

.. warning::
   NVIDIA GPU가 없는 PC에서는 프로그램을 동작시킬 수 없음.

.. toctree::
   :maxdepth: 2
   :caption: Installation Guide:

   install_window
   install_linux

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   tutorial

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
