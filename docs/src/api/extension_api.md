# [Extending Associations: higher-order hooks](@id extension_api)

The functions on this page are not exported, but they are a stable, documented API for
packages that build on the higher-order information measures, for example to compute local
(per-sample) versions of the measures. Access them with the module prefix, e.g.
`Associations.subset_entropies`. Breaking changes to them follow the same versioning rules
as the exported API.

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
