# Fisher Forecast for the Galaxy Bispectrum of upcoming Large-Scale-Structure including velocity via kSZ
### Summary
In this project, I performed a fisher forecast for upcoming Large-Scale Structure surveys. The calculation is computationally very expensive because one of the operations scales like :math:`$\mathcal{O}(N^6)$`. Even with optimisation using Cython and parallelisation of the code, the code was not feasible to run on 100+ cores. I found a statistical property in the calculation that allowed me to approximate this operation. With this approximation, the scaling reduces to :math:`$\mathcal{O}(N)$` making the calculation feasible. 

### Details
For more details, I refer to the resulting [publication](https://arxiv.org/abs/2303.05535).
