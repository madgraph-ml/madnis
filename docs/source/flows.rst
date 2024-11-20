Using the MadNIS flow library
=============================

In addition to neural importance sampling, the normalizing flow implementation in the MadNIS package
can also be used in other applications. The following examples show how unconditional and
conditional MadNIS flows can be trained on a toy training dataset.

Simple flow training
--------------------

As an example distribution, we will look at a two-dimensional Gaussian mixture model with two peaks
at :math:`(-1,-1)` and :math:`(1,1)` and standard deviations 1 and 0.5. We can generate our
training dataset with the code

.. code-block:: python

    data = torch.cat((
        torch.randn((5000, 2)) - 1, 0.5 * torch.randn((5000, 2)) + 1
    ), dim=0)

Now we can build the MadNIS flow and train it on the data. In this case we will use a Gaussian
latent space. Another option would be a uniform latent space, in which case the distribution would
be restricted to the unit hypercube. As a loss function, we use the negative log-likelihood.

.. code-block:: python

    from madnis.nn import Flow
    flow = Flow

In the last step, we can draw samples from our learned distribution and confirm that the
normalizing flow has indeed learned the distribution of the training data by making histograms.

.. code-block:: python



Conditional flow training
-------------------------

MadNIS flows can also be used to learn conditional distributions. We again use the Gaussian mixture
model from the previous section, but apply a Gaussian smearing with standard deviation 0.1 as a
second step, giving us pairs of points :math:`(x, y)`. We can now use conditional flows to, for
example, solve the inverse problem, i.e. learn the distribution :math:`p(x|y)`.

.. code-block:: python

    from madnis.nn import Flow


Hyperparameters
---------------
