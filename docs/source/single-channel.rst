Integration settings
====================

In the last tutorial, we have discussed the most basic options when building a MadNIS integrator.
This page describes further options to adapt the integrator for different applications.

Changing the integration domain
-------------------------------

The default integration domain in MadNIS is the unit hypercube :math:`[0,1]^d`. To choose different
finite intervals for the integration domain, a custom
:py:class:`Integrand <madnis.integrator.Integrand>` object can be created. For instance, to
integrate the function from the previous tutorial over the interval :math:`[0,2]^4`, we can
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

Training hyperparameters
------------------------

- loss
- batch size
- learning rate
- optimizer
- scheduler

Dealing with zeros
------------------

- drop zero integrand
- batch size threshold

Device and data type
--------------------
