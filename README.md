Silk: a CUDA drifter searcher
=============================

**Silk** is a CGoL drifter search program inspired by Mitchell Riley's
[Barrister](https://github.com/mvr/barrister). We port Barrister's
underlying algorithm, modulo a few tweaks, to CUDA in order to benefit
from the much greater parallelism present on GPUs.

![Silk logo](./docs/silk.jpg)

General idea
------------

In a stable pattern in Conway's Game of Life (one that doesn't change
from one generation to the next), each cell must be of one of eight
different 'types':

 - dead with 0 live neighbours;
 - dead with 1 live neighbour;
 - dead with 2 live neighbours;
 - live with 2 live neighbours;
 - live with 3 live neighbours;
 - dead with 4 live neighbours;
 - dead with 5 live neighbours;
 - dead with 6 live neighbours;

which are abbreviated to d0, d1, d2, l2, l3, d4, d5, and d6. We
follow the design of Barrister by having a Boolean variable per cell
for each of these 8 types, specifying whether or not we know that
the cell is definitely not of that type. These are packed into
bitplanes, allowing logical operations to be performed in parallel
at every location in the plane using bitwise instructions.


