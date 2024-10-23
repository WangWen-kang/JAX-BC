# JAX_BC
An extended patch to JAX-Fluids for differentiable boundary conditions

The current JAX-Fluids does not have built-in differentiable boundary conditions. In this patch, we add several differentiable boundary conditions that are highly customizable to the original JAX-Fluids.

Please cite the following paper if you find this code useful

Wenkang Wang, Xu Chu, Optimized Flow Control based on Automatic-Differentiation in Compressible Turbulent Channel Flows(in preparation),2024.

## Differentiable Boundary conditions in this patch
1. Opposition control

A wall normal velocity is applied on the upper and lower wall. This control strategy is applied with the objective of introducing a counteracting wall-normal velocity at the boundary, designed to oppose the near-wall turbulence structures.

2. Permeable wall 
The wall normal velocity on the lower and upper walls are modeled as a linear function of pressure fluctuations.


## Installation

1. download the code of this patch:

```bash
git clone https://github.com/WangWen-kang/JAX_BC.git
```

2. download the JAX-Fluids code 

```bash
git clone https://github.com/tumaer/JAXFLUIDS.git
```

3. replace the source code of JAX-Fluids with this patch

```bash
cp -r PathToYourFolder/JAX_BC/jaxfluids PathToYourFolder/JAXFLUIDS/src/
```

4. install JAX-Fluids

```bash
pip install --upgrade "jax[cuda12]" # for gpu
cd PathToYourFolder/JAXFLUIDS/
pip install -e .
```