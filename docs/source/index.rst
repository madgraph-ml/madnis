MadNIS documentation
====================

MadNIS is a Python library for neural importance sampling based on PyTorch.

You can find instructions on how to install and use MadNIS on the following pages:

* :doc:`installation`
* :doc:`first-steps`
* :doc:`multi-channel`
* :doc:`integration-settings`
* :doc:`flows`

The following pages contain a detailed reference of all the classes and methods in the madnis
package:

* :doc:`madnis.integrator`
* :doc:`madnis.nn`

Citation
--------

If you use this code or parts of it, please cite:

.. code-block:: bib

    @article{Heimel:2023ngj,
      author = "Heimel, Theo and Huetsch, Nathan and Maltoni, Fabio and Mattelaer, Olivier and Plehn, Tilman and Winterhalder, Ramon",
      title = "{The MadNIS reloaded}",
      eprint = "2311.01548",
      archivePrefix = "arXiv",
      primaryClass = "hep-ph",
      reportNumber = "IRMP-CP3-23-56, MCNET-23-12",
      doi = "10.21468/SciPostPhys.17.1.023",
      journal = "SciPost Phys.",
      volume = "17",
      number = "1",
      pages = "023",
      year = "2024"}

.. toctree::
   :maxdepth: 1
   :caption: Usage
   :hidden:

   installation
   first-steps
   multi-channel
   integration-settings
   flows

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   madnis.integrator
   madnis.nn
