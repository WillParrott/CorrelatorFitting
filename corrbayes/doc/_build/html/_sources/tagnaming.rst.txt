
Tag Naming Convention
=====================

This module requires a certain convention of datatags in the gv.Dataset.Dataset that is fed in.
For 2-point correlators, it requires 

2-point correlator keys must have the format “meson.ss”, 
where “meson” labels the meson, and the “s“‘s labels the source/sink combination, 
e.g. etac.ll is a local-local eta_c correlator.

3-point correlator keys must have the format “meson1.J.meson2_T{T}.ss”,
where J labels the current, meson1/2 labels the states on either side, {T} is the source/sink temporal
separation, and the "s"'s label the source/sink combination.
e.g. B.V0.Pi_T15.ll is a sensible name for a B -> pi local-local correlator with a temporal vector current,
temporal spacing of 15.

