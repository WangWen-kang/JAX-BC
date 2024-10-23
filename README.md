# JAX_BC
An extended patch to JAX-Fluids for differentiable boundary conditions

The current JAX-Fluids does not have built-in differentiable boundary conditions. In this patch, we add several differentiable boundary conditions that are highly customizable to the original JAX-Fluids.

Please cite the following paper if you find this code useful

Wenkang Wang, Xu Chu, Optimized Flow Control based on Automatic-Differentiation in Compressible Turbulent Channel Flows(in preparation),2024.

## Differentiable BCs in this patch
### 1. Opposition control

A wall normal velocity is applied on the upper and lower wall. This control strategy is applied with the objective of introducing a counteracting wall-normal velocity at the boundary, designed to oppose the near-wall turbulence structures.

### 2. Permeable wall 

The wall normal velocity on the lower and upper walls are modeled as a linear function of pressure fluctuations.


## Installation

### 1. Download the code of this patch:

```bash
git clone https://github.com/WangWen-kang/JAX_BC.git
```

### 2. Download the JAX-Fluids code 

```bash
git clone https://github.com/tumaer/JAXFLUIDS.git
```

### 3. Replace the source code of JAX-Fluids with this patch

```bash
cp -r PathToYourFolder/JAX_BC/jaxfluids PathToYourFolder/JAXFLUIDS/src/
```

### 4. Install JAX-Fluids

```bash
pip install --upgrade "jax[cuda12]" # for gpu
cd PathToYourFolder/JAXFLUIDS/
pip install -e .
```

Now the JAX-Fluids will be able to implement the differentiable BCs.

## Quick Start
### In the setup json file 
The BCs in current patch follows the same rules with the built-in BCs.
An example of case setup json file for BC setting is  

```
    "boundary_conditions": {
        "east": {"type": "PERIODIC"},
        "west": {"type": "PERIODIC"},
        "north": {
            "type": "ISOTHERMALOPPOSITIONUPWALL",
            "wall_velocity_callable": {
                "u": 0.0,
                "v": 0.0,
                "w": 0.0
            },
            "wall_temperature_callable": 1.0
        },
        "south": {
            "type": "ISOTHERMALOPPOSITIONLOWWALL",
            "wall_velocity_callable": {
                "u": 0.0,
                "v": 0.0,
                "w": 0.0
            },
            "wall_temperature_callable": 1.0
        },
        "top": {"type": "PERIODIC"},
        "bottom": {"type": "PERIODIC"}
    },
```
This code snippet sets the the south and north boundary as isothermal permeable wall.

For opposition control with isothermal condition, one can use "ISOTHERMALJJPOROUSUPWALL" and "ISOTHERMALJJPOROUSLOWWALL".  

One can also remove the "ISOTHERMAL" from the keyword to implement an adiabatic boundary condition.

### In the main file

In the main file we have to prepare a dictionary that contains all the the boundary condition parameters.
The dict variable should have the following format:

```
my_dict={
    'upper':jnp.array,
    'lower':jnp.array
}
```

The values of keys `'upper'` and `'lower'` correspond to the parameters of upper and lower walls, which can either be a number or an array.

The next step is to pass the dictionary to the solver. Just simple pass `my_dict` to the key `ml_parameters_dict` 

In regular simulation, one can use

```
_, _, forcing_parameters = initialization_manager.initialization(ml_parameters_dict=my_dict0)
sim_manager.simulate(simulation_buffers, time_control_variables,forcing_parameters,ml_parameters_dict=my_dict)
```

In feed forward simulation, one can use 

```
data_series,_ = sim_manager.feed_forward(
                        batch_primes_init=my_prime_init, physical_timestep_size=time_step,\
                        t_start=0.0, outer_steps=1000,inner_steps=1,is_scan=True,forcing_parameters=forcing_parameters,ml_parameters_dict=my_dict)    
```