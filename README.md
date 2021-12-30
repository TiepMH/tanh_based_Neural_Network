# This toy example aims to use ***_Deep_ _Learning_*** for finding a near-optimal power-allocation function.

### 1. Stochastic optimization problem is summarized as follows:
<p align="center">
  <img src="https://github.com/TiepMH/tanh_based_Neural_Network/blob/main/figs/Prob_statement.png" width="50%" height="50%">
</p>

<div align="justify">
  
### A deep ***neural network*** (DNN) is designed to solve the above problem. The DNN takes channel realizations as the input and then outputs some power-allocation coefficient to satisfy the proposed optimization problem. The output is a function of the input. This means that the power-allocation coefficient is a function of the channels. Thus, the training process is equivalent to the process of finding an optimal/near-optimal function. NOTE: The DNN learns to optimize a ***function*** but not just a single variable.

</div>

---

### 2. Illustration:

  + ### 2a) The average cost:
  
<img src="https://github.com/TiepMH/tanh_based_Neural_Network/blob/main/figs/Cost_vs_epoch.png" width="50%" height="50%">


  + ### 2b) The average DCMC capacity of the legitimate user and the average DCMC capacity of the eavesdropper:
  
<img src="https://github.com/TiepMH/tanh_based_Neural_Network/blob/main/figs/Cap_vs_epoch.png" width="50%" height="50%">

  + ### 2c) The SEP of the legitimate user and the SEP of the eavesdropper:
  
<img src="https://github.com/TiepMH/tanh_based_Neural_Network/blob/main/figs/SER_vs_epoch.png" width="50%" height="50%">
