# Learning to Pivot with Adversarial Networks
https://arxiv.org/abs/1611.01046

* Gilles Louppe
* Michael Kagan
* Kyle Cranmer 

Many inference problems involve data generation processes that are not uniquely specified or are uncertain in some way. In a scientific context, the presence of several plausible data generation processes is often associated to the presence of systematic uncertainties. Robust inference is possible if it is based on a pivot -- a quantity whose distribution is invariant to the unknown value of the (categorical or continuous) nuisance parameters that parametrizes this family of generation processes. In this work, we introduce a flexible training procedure based on adversarial networks for enforcing the pivotal property on a predictive model. We derive theoretical results showing that the proposed algorithm tends towards a minimax solution corresponding to a predictive model that is both optimal and independent of the nuisance parameters (if that models exists) or for which one can tune the trade-off between power and robustness. Finally, we demonstrate the effectiveness of this approach with a toy example and an example from particle physics.

---

Please cite using the following BibTex entry:

```
@article{louppe2016pivot,
           author = {{Louppe}, G. and {Kagan}, M. and {Cranmer}, K.},
            title = "{Learning to Pivot with Adversarial Networks}",
          journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
           eprint = {1611.01046},
     primaryClass = "stat.ML",
             year = 2016,
            month = nov,
}
```
