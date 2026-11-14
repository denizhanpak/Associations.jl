# [Extending Associations: higher-order and PID hooks](@id extension_api)

The functions and types on this page are not exported, but they are a stable, documented
API for packages that build on the higher-order information measures and the partial
information decomposition (PID), for example to add new redundancy functions or to compute
local (per-sample) versions of the measures. Access them with the module prefix, e.g.
`Associations.subset_entropies`, and extend them with
`function Associations.pid_redundancies(...)`. Breaking changes to them follow the same
versioning rules as the exported API.

## Higher-order measures

Every [`HigherOrderInformationMeasure`](@ref) is a linear combination of joint entropies
of subsets of the inputs. [`Associations.subset_entropies`](@ref) gives those entropies
for an estimator, and [`Associations.entropy_combination`](@ref) combines them into a
measure. `entropy_combination` only adds, subtracts, scales by integers and sums the values
returned by its entropy callable, so a callable returning one vector of local entropies
per subset gives the local values of the measure.

```@docs
Associations.subset_entropies
Associations.entropy_combination
```

## Partial information decomposition

A [`PartialInformationDecomposition`](@ref) is defined by a [`RedundancyFunction`](@ref).
To add a redundancy function, define a subtype of [`RedundancyFunction`](@ref) and extend
[`Associations.pid_redundancies`](@ref) (see "Implementing a new redundancy function" in the
[`RedundancyFunction`](@ref) docstring). Associations.jl then computes the atoms with
[`Associations.mobius_inversion`](@ref), and [`association`](@ref), [`PIDAtom`](@ref) and
[`PIDResult`](@ref) work as for the built-in redundancy functions.

```@docs
Associations.pid_redundancies
Associations.RedundancyLattice
Associations.redundancy_lattice
Associations.lattice_node_vectors
Associations.mobius_inversion
```
