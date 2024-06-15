Silk: a CUDA drifter searcher
=============================

**Silk** is a CGoL drifter search program inspired by Mitchell Riley's
[Barrister](https://github.com/mvr/barrister). We port Barrister's
underlying algorithm, modulo a few tweaks, to CUDA in order to benefit
from the much greater parallelism present on GPUs.

General idea
------------

The input to Silk is an initial **perturbation** $`P`$ together with a
partially unknown **stable background** $`S`$. The background must be
stable in the sense that $`f(S) = S`$ where $`f`$ is the CGoL rule.
The current generation is the XOR $`P \oplus S`$ of the perturbation
and the background, so the perturbation itself evolves as:

$`P \mapsto f(P \oplus S) \oplus S`$

Silk then performs a search over possible backgrounds $`S`$ subject
to various constraints, including bounds on the population and bounding
box of the **active region** (the restiction of $`P`$ to the immediate
neighbourhood of $`S`$). This is implemented as a binary tree search,
where branching decisions are only made if they affect the perturbation
or the restriction of the stable background to the cells that, in at
least one generation, differ from the background. This ensures that
the search does not needlessly dudplicate branches of search space
that differ only in the [stator](https://conwaylife.com/wiki/Stator).

When a candidate solution is found (either because $`P`$ fizzles out
completely, or optionally enters a sufficiently high-period cycle or
becomes well separated from the live cells in $`S`$ for sufficiently
long), it is given to a SAT solver to find a minimal completion of
the stator. We use [CaDiCaL](https://github.com/arminbiere/cadical)
by Armin Biere as our SAT solver of choice, as it is state-of-the-art
and supports incremental solving.

The main tree search runs on the GPU whilst the SAT solvers run on
CPU threads asynchronously, so the completion of stators does not slow
down the main search.

![Silk logo](./docs/silk.jpg)

Tree search methodology
-----------------------

Precursors to Silk, such as Dean Hickerson's **dr** (1997), Mike
Playle's [Bellman](https://conwaylife.com/wiki/Bellman) (2013), and
Mitchell Riley's [Barrister](https://github.com/mvr/barrister) (2022),
performed a depth-first tree search. Depth-first search is inherently
sequential, but a GPU has many independent streaming multiprocessors,
so we implement a parallel variant of depth-first search.

Specifically, we have a priority heap that prioritises nodes with
greater "information content" (something that is monotone-increasing
along any branch of the search tree, so deeper nodes are prioritised
over nodes closer to the root). In each iteration we pop a certain
number $`B`$ of nodes from the heap, process them, and then push any
child nodes back onto the heap. The limit $`B = 1`$ would be depth-
first search, but we use a larger $`B`$ in order to give parallelism.

The heap is Deo and Prasad's vectorised priority heap from their 1992
[paper](https://link.springer.com/article/10.1007/BF00128644), but
implemented in CUDA. We use a vector size of 2048, and the items are
pairs of nodes, so the heap itself stores a multiple of 4096 nodes.
Any leftover nodes are stored in a ring-buffer "staging area" until
we have at least a full vector of 4096 nodes to add to the heap, and
likewise we pop complete vectors (so $`B`$ is constrained to be a
multiple of 4096 as well). In practice we choose $`B`$ as a function
of how full the heap is, to prevent overflowing the available memory:

![Batch size diagram](./docs/batchsize.png)

In particular, when the heap is less than 25% full, we set the batch
size to be the maximum of $`\frac{N}{16} - 4096`$ where $`N`$ denotes
the maximum capacity of the heap. When the heap is more than 75% full,
we set the batch size to be the minimum of 4096. Between those, we
vary it linearly with a gradient of $`-\frac{1}{8}`$ as shown above.

Data structures
---------------

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

As the background is a $`64 \times 64`$ torus, those eight bitplanes
together occupy 4096 bytes. We need another 128 bytes to store the
active perturbation (which is required to fit in a $`32 \times 32`$
box), along with 256 bytes of miscellaneous metadata information,
giving a total of 4480 bytes to represent the pair of problems
obtained when we make a branching decision. (As such, each individual
problem occupies 2240 bytes amortized, and the maximum capacity $`N`$
of the heap is, to a first approximation, the amount of free GPU
memory divided by 2240.)

