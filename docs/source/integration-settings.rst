Integration settings
====================

In the last two tutorial, we have discussed how the MadNIS integrator can be used for single- and
multi-channel integration. This page describes further options to adapt the integrator for
different applications.

Changing the integration domain
-------------------------------

The default integration domain in MadNIS is the unit hypercube :math:`[0,1]^d`. To choose different
finite intervals for the integration domain, a custom
:py:class:`Integrand <madnis.integrator.Integrand>` object can be created. For instance, to
integrate the function from the first tutorial over the interval :math:`[0,2]^4`, we can
construct the integrator as follows:

.. code-block:: python

    from madnis.integrator import Integrator, Integrand
    f = Integrand(lambda x: (2 * x).prod(dim=1), input_dim=4, bounds=[[0.0, 2.0]] * 4)
    integrator = Integrator(f)

For an infinite integration domain we can change the normalizing flow in our integrator to use a
normal latent space instead of the default uniform latent space. For example, to integrate the
function :math:`f(x,y) = \exp(- x^2 - y^2)`, we can define our integrator as

.. code-block:: python

    integrator = Integrator(
        lambda x: x.square().sum(dim=1).neg().exp(),
        dims=2,
        flow_kwargs={"uniform_latent": False},
    )

Note that this only works well if the means and standard deviations of our integrand are roughly
zero and one, respectively. Otherwise, the integrand should be shifted and rescaled. Also note that,
internally, the integration interval is limited to :math:`[-10, 10]` if the flow is built this way.
An alternative way to integrate over the whole space of real numbers is to apply a logit
transformation and the corresponding Jacobian in the integrand function.

Network architecture
--------------------

The default values for the normalizing flow hyperparameters in MadNIS will work well for many
simple, low-dimensional integrands. For more complex functions, it can be necessary to build a
larger network. This can be done with the ``flow_kwargs`` argument of the
:py:class:`Integrator <madnis.integrator.Integrator>` class. They are passed on to the constructor
of the :py:class:`Flow <madnis.nn.Flow>` class. By default, the flow is built with a sufficient
number of coupling blocks such that every component is conditioned on every other component at least
once. Hence, it is normally not necessary to set the number of coupling blocks by hand. In the
following example, we change the settings for the depth and number of hidden nodes of the flow
sub-networks.

.. code-block:: python

    integrator = Integrator(
        lambda x: (2 * x).prod(dim=1),
        dims = 4,
        flow_kwargs={"layers": 4, "units": 64},
    )

Similarly, the ``cwnet_kwargs`` parameter can be used to change the hyperparameters of the network
used for the trained channel weights.

For even more flexibility with the network architecture, the ``flow`` and ``cwnet`` arguments can
be used to replace the normalizing flows and channel weight networks used by default in MadNIS.
The interface that objects passed to the ``flow`` arguments have to support is specified by the
abstract base class :py:class:`Distribution <madnis.integrator.Distribution>`. A class used as
channel weight network should have a ``forward`` function that accepts tensors of shape
``(batch_size, remapped_dim)`` and returns tensors of shape ``(batch_size, channel_count)``. The
output of the network is then added to the logarithm of the prior channel weights if they were
provided. After that, the normalized channel weights are computed.

Training hyperparameters
------------------------

There are several hyperparameters that affect the network training that can be set when the
constructor of :py:class:`Integrator <madnis.integrator.Integrator>` is called. The loss function
can be changed using the ``loss`` argument. By default, the KL divergence will be used for
single-channel integration. The integral variance and reverse KL divergence are available as
alternative options. The same options are available for multi-channel training, however only the
variance loss allows for the simultaneous optimization of channel mappings and weights.

Further important training parameters are the batch size (``batch_size`` argument) and the learning
rate (``learning_rate`` argument). To enable training with a variable learning rate, a learning rate
scheduler has to be constructed. This can be done by defining a function that returns the scheduler
with the optimizer as a parameter. For instance, the following code sets cosine annealing as the
learning rate scheduling.

.. code-block:: python

    from torch.optim.lr_scheduler import CosineAnnealingLR
    integrator = Integrator(
        ..., # other arguments
        scheduler = lambda opt: CosineAnnealingLR(opt, n_steps) # number of training iterations
    )

If a learning rate scheduler is given, the learning rate used for the current training iteration
will be given in the :py:class:`TrainingStatus <madnis.integrator.TrainingStatus>` object.
Similarly, we can also set the optimizer by passing a function that constructs the optimizer given
the trainable parameters. For instance, to use the ``SGD`` optimizer instead of ``Adam``, we can use

.. code-block:: python

    from torch.optim import SGD
    integrator = Integrator(
        ..., # other arguments
        optimizer = lambda params: SGD(params, lr=1e-3)
    )

VEGAS pre-training
------------------

VEGAS is a commonly used algorithm for importance sampling that works by assuming a factorized
distribution, i.e. no correlations between different dimensions. It then models the one-dimensional distributions using variable-width bins with uniform probabilities. This makes training VEGAS much
faster than the training of a neural network using stochastic gradient descent. We can use that to
our advantage in MadNIS by training a VEGAS grid first and then using it to initialize our
normalizing flow, using the :py:class:`VegasPreTraining <madnis.integrator.VegasPreTraining>` class.
It is constructed using an :py:class:`Integrator <madnis.integrator.Integrator>` instance and uses
the same integrand, sample buffer and integration cache. It relies on the ``vegas`` package which is
an optional dependency of MadNIS. The pre-training can be performed with the following code:

.. code-block:: python

    from madnis.integrator import VegasPreTraining

    # Construct integrator here

    vegas = VegasPreTraining(integrator, bins=64, damping=0.8)
    vegas.train([1000,2000,4000])
    vegas.initialize_integrator()

    # Regular MadNIS training here

The only two hyperparameters are the number of bins for the VEGAS grid and the damping parameter
that influences the VEGAS convergence (high: fast adaption, low: stable convergence). The parameters
of the train method specify the number of samples in each VEGAS iteration (per channel in the
multi-channel case). The last line initializes the normalizing flow in the integrator.

Similar to the :py:class:`Integrator <madnis.integrator.Integrator>` class, the
:py:class:`VegasPreTraining <madnis.integrator.VegasPreTraining>` class also has methods ``sample``,
``integrate``, ``integration_metrics`` and ``unweighting_metrics`` which directly draw samples using
the VEGAS grid. This allows us to compare the VEGAS performance and the normalizing flow performance
directly.

Dealing with zeros
------------------

The MadNIS integrator has two parameters that control how it treats samples with an integrand value
of zero. The parameter ``batch_size_threshold`` controls the minimum amount of samples with non-zero
values per batch relative to the total batch size during the training. More samples are generated
until the number is above this threshold. In addition, the parameter ``drop_zero_integrands``
removes samples with zero-integrand value from the optimization. Depending on the loss function,
this may change the optimization objective. It is especially useful in situations where the
integrand evaluation more expensive for samples with non-zero weight. In that case, improving the
training for samples with non-zero weights at the cost of a lower cut efficiency can be beneficial
for the overall performance.

Device and data type
--------------------

The device and data type used for training and sampling can be set using the ``device`` and
``dtype`` arguments of the :py:class:`Integrator <madnis.integrator.Integrator>` constructor.
As the class inherits from ``torch.nn.Module``, the ``to`` function can be used alternatively
to change the device or data type.


Storing and loading trained models
----------------------------------

The :py:class:`Integrator <madnis.integrator.Integrator>` class is a ``torch.nn.Module``. The
functions ``torch.save`` and ``torch.load`` can therefore be used to store and load trained models.
The saved state includes all network parameters and the integration history, but not the buffered
training samples.

.. code-block:: python

    # save integrator
    torch.save(integrator.state_dict(), "integrator.pth")
    # load integrator
    integrator.load_state_dict(torch.load("integrator.pth"))
