Introduction
============

.. image:: https://img.shields.io/pypi/v/pytorch-tabnet2.svg
   :target: https://pypi.org/project/pytorch-tabnet2/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/pytorch-tabnet2.svg
   :target: https://pypi.org/project/pytorch-tabnet2/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=ubuntu
   :alt: Ubuntu

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=windows
   :alt: Windows

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=apple
   :alt: MacOS

.. image:: https://codecov.io/gh/DanielAvdar/tabnet/branch/main/graph/badge.svg
   :alt: Coverage

.. image:: https://img.shields.io/github/last-commit/DanielAvdar/tabnet/main
   :alt: Last Commit



TabNet is an attentive, interpretable deep learning architecture for tabular data, implemented in PyTorch. This project is a maintained fork of the original DreamQuark TabNet, with improvements for metrics, GPU support, and usability. It is suitable for classification, regression, and multitask learning on tabular datasets.

Installation
------------

Install TabNet using pip:

.. code-block:: bash

   pip install pytorch-tabnet2


Original Repository
--------------------

This project is a maintained fork of the original DreamQuark TabNet implementation:

- `dreamquark-ai/tabnet <https://github.com/dreamquark-ai/tabnet>`_

Key Features
-------------

- Supports classification, regression, multitask, and unsupervised pretraining
- GPU acceleration and efficient data handling
- Interpretable feature masks and explanations
- Flexible API for research and production

For more details, see the original paper: https://arxiv.org/pdf/1908.07442.pdf

Project Changes from the Original Implementation
-------------------------------------------------

Key Changes from Original
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Removed the PyTorch DataLoader, which previously accessed each datapoint individually and limited vectorization, resulting in slow performance. Data is now processed in a more efficient, vectorized manner.
- Replaced sklearn metrics with torcheval, enabling fast, GPU-accelerated metric computation without the need to move data to the CPU or convert to numpy.
- Shifted data weighting from the sampling/data loading stage to the loss function and metric calculations, providing more flexibility and efficiency.

Key Improvements
~~~~~~~~~~~~~~~~~

- Added comprehensive unittests, achieving over 90% code coverage for improved reliability and maintainability.
- Significantly reduced training time on both CPU and GPU, primarily due to the removal of the DataLoader and improved vectorization.
- Enabled real-time validation metric calculation on the GPU during training, leveraging torcheval for efficient, on-device evaluation.
- Added type annotations throughout the codebase for improved code clarity and static analysis.
- Added support for newer versions of Python: 3.10, 3.11, 3.12, and 3.13.
