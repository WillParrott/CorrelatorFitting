
Introduction
************

Hello.

This is a small module that finds sensible priors for correlation functions (to be fit using the ``corrfitter`` package)
in an automatic and pure-Bayesian way.

Traditionally, one would design priors for a correlator fit using some combination of QCD-inspired rules of thumb 
("the spectrum is separated by about Lambda_QCD" ), and empirical-Bayes type arguments 
(like eyeing the effective mass of correlators averaged over all configurations). While empirical Bayes approaches are often
fine since they approximate a pure Bayesian result, they do *technically* involve double-counting the information avaliable to us,
and this makes some people feel on-edge.

This module produces priors in a pure Bayesian way. With a gv.Dataset.Dataset full of correlators ``dset``, you can run

``prior, new_dset = CorrBayes.get_prior( dset, 1, nexp )``

This line will shave off a (randomly chosen) single datapoint for each key (e.g. a correlator for each key on a single configuration),
and return ``new_dset``, the same as ``dset`` but with that single point taken out. Then that point is used to deduce sensible priors
for a fit using ``new_dset``. No information from ``new_dset`` has been used to determine these priors, hence no double-counting of
information.

If you are also doing fits to 3-point correlators, this can be handled too. All you need to do is pass an argument into get_priors called
"currents", this must be a list of strings giving the name you gave the current in your datatag convention (See tag conventions section).

The second argument of ``get_prior`` in the above code segment gives the number of points to be used for working out priors (therefore also the number of points shaved off the datset). One is fine in many cases, but if your correlators are noisy (e.g. vector mesons), 
then cranking this up to 10 or 20 would make the process more stable.

I should also mention; this only works for single source/sink combinations, e.g. if you're planning on using a matrix of smearings,
this won't work properly. If you want to use this, let me know and I'll adapt the code so it can handle that kind of thing.
