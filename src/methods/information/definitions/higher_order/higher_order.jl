export HigherOrderInformationMeasure

"""
    HigherOrderInformationMeasure <: MultivariateInformationMeasure

The supertype for Shannon measures of the shared information among any number of
variables, each of which is a combination of the joint entropies of subsets of
the variables.

## Implementations

- [`TotalCorrelation`](@ref)
- [`DualTotalCorrelation`](@ref)
- [`OInformation`](@ref)
- [`SInformation`](@ref)

See also [`oinformation_gradients`](@ref).

## Inputs

Each input `xᵢ` is one variable. It can be a vector (a scalar variable) or a
`StateSpaceSet` (a group of variables treated as one multivariate variable). At
least two inputs are needed.

## Compatible estimators

- [`JointProbabilities`](@ref): plug-in estimates from one joint probability mass
    function. Every marginal is taken from the same joint table, so all subset
    entropies share one discretization. The joint table is stored densely, so its
    size is the product of the number of outcomes of each variable.
- [`EntropyDecomposition`](@ref) with a [`DiscreteInfoEstimator`](@extref ComplexityMeasures.DiscreteInfoEstimator)
    of [`Shannon`](@extref ComplexityMeasures.Shannon) entropy: each subset is
    discretized and counted separately, which avoids the dense table. Any discrete
    estimator can be used, including the bias-corrected ones (`MillerMadow`, `ChaoShen`,
    `HorvitzThompson`, `Jackknife`, `Schuermann`, `GeneralizedSchuermann`), and the
    `pest` argument takes a probabilities estimator such as `BayesianRegularization`.
    Each subset entropy is estimated from the counts of that subset's *observed* joint
    outcomes; pseudo-count and shrinkage corrections therefore act on the observed
    outcomes only, not on every cell of the product of the marginal alphabets.
- [`EntropyDecomposition`](@ref) with a [`DifferentialInfoEstimator`](@extref ComplexityMeasures.DifferentialInfoEstimator)
    of [`Shannon`](@extref ComplexityMeasures.Shannon) entropy, e.g. `Kraskov`:
    each subset entropy is estimated separately. The biases of the separate
    estimates, which grow with dimension, do not cancel in general.
- [`GaussianDecomposition`](@ref): closed-form Gaussian entropies, optionally after a
    Gaussian copula transform.
"""
abstract type HigherOrderInformationMeasure <: MultivariateInformationMeasure end

min_inputs_vars(::HigherOrderInformationMeasure) = 2
max_inputs_vars(::HigherOrderInformationMeasure) = Inf

# ----------------------------------------------------------------------------------------
# Joint entropies of subsets of the input variables.
# ----------------------------------------------------------------------------------------
# For internal use. `SubsetEntropies` wraps a function `h(S)` that returns the joint
# entropy, in the unit of the measure, of the input variables with indices in `S`.
# Results are cached, because the gradients reuse the same subsets many times.
struct SubsetEntropies{F}
    h::F
    cache::Dict{Vector{Int},Float64}
end
SubsetEntropies(h) = SubsetEntropies(h, Dict{Vector{Int},Float64}())

function (H::SubsetEntropies)(S)
    key = sort!(collect(Int, S))
    isempty(key) && return 0.0
    return get!(() -> H.h(key), H.cache, key)
end

"""
    Associations.subset_entropies(est, x...) → H

Return a callable `H` such that `H(S)` is the estimated joint Shannon entropy of the
inputs `x[S]`, in the unit given by `est.definition.base`.

This function is part of the extension API (see the "Extending Associations" page of the
documentation). It is not exported; access it as `Associations.subset_entropies`.

## Arguments

- `est`: one of
    - [`JointProbabilities`](@ref): every marginal is taken from the joint probability
        mass function of all inputs, which is computed when `subset_entropies` is called;
    - [`EntropyDecomposition`](@ref) with a `DiscreteInfoEstimator` or a
        `DifferentialInfoEstimator` of `Shannon` entropy: each subset is estimated
        separately, when it is first requested;
    - [`GaussianDecomposition`](@ref): closed-form Gaussian entropies from the
        covariance matrix of all inputs.
    `est.definition` may be any `MultivariateInformationMeasure` with a `base` field, such
    as every [`HigherOrderInformationMeasure`](@ref) and
    [`PartialInformationDecomposition`](@ref).
- `x...`: the input variables, each an `AbstractVector` or a `StateSpaceSet`, all with the
    same number of samples.

## The returned callable

`H(S)` takes a collection `S` (a `Tuple`, `Vector` or range) of distinct indices in
`1:length(x)`. The order of the indices does not matter. It returns a `Float64`, the joint
entropy of `x[S]` in the unit `est.definition.base` (bits for the default base 2).
`H(S)` for an empty `S` is `0.0`. Values are cached per subset, so repeated calls are
cheap; `H` is not safe to call concurrently from several tasks.

Only the callable interface of `H` is part of the API; its concrete type is internal.

See also [`Associations.entropy_combination`](@ref).
"""
function subset_entropies end

function subset_entropies(est::JointProbabilities{<:MultivariateInformationMeasure}, x...)
    p = probabilities(est.discretization, x...)
    n = length(x)
    measure = Shannon(; base = est.definition.base)
    h(S) = information(measure, length(S) == n ? p : marginal(p; dims = S))
    return SubsetEntropies(h)
end

function subset_entropies(
        est::EntropyDecomposition{<:MultivariateInformationMeasure, <:DifferentialInfoEstimator{<:Shannon}},
        x...
    )
    X = StateSpaceSet.(x)
    modified_est = estimator_with_overridden_parameters(est.definition, est.est)
    h(S) = information(modified_est, StateSpaceSet(X[S]...))
    return SubsetEntropies(h)
end

function subset_entropies(
        est::EntropyDecomposition{<:MultivariateInformationMeasure, <:DiscreteInfoEstimator{<:Shannon}},
        x...
    )
    # One encoding per input; a multi-column input gives a multi-column encoding.
    E = StateSpaceSet.(codified_marginals(est.discretization, x...))
    modified_est = estimator_with_overridden_parameters(est.definition, est.est)
    o = UniqueElements()
    h(S) = information(modified_est, est.pest, o, StateSpaceSet(E[S]...))
    return SubsetEntropies(h)
end

# ----------------------------------------------------------------------------------------
# Each measure is a combination of subset entropies. `entropy_combination(def, H, V)`
# evaluates the measure on the variables with indices `V`.
# ----------------------------------------------------------------------------------------
"""
    Associations.entropy_combination(definition, H, V) → value

Evaluate the higher-order measure `definition` on the variables with indices `V`, as a
linear combination of the joint entropies `H(S)` of subsets `S ⊆ V`.

This function is part of the extension API (see the "Extending Associations" page of the
documentation). It is not exported; access it as `Associations.entropy_combination`.

## Arguments

- `definition`: a [`TotalCorrelation`](@ref), [`DualTotalCorrelation`](@ref),
    [`OInformation`](@ref) or [`SInformation`](@ref). Its `base` field is not used; the
    unit of the result is the unit of the values returned by `H`.
- `H`: any callable. It is called as `H(S)`, where `S` is a `Tuple` `(i,)` or a
    `Vector{Int}`, always a subset of `V`, with indices in the order they appear in `V`.
    [`Associations.subset_entropies`](@ref) returns such a callable, but `H` can be
    replaced by any callable whose values support `a + b`, `a - b`, multiplication by an
    integer (`k * a`) and `sum` over a generator. For example, a callable that returns, for
    each subset, a `Vector{Float64}` of per-sample (local) entropies
    ``h(x_S) = -\\log p(x_S)`` gives the per-sample (local) values of the measure,
    elementwise.
- `V`: a `Vector{Int}` of distinct variable indices. [`association`](@ref) uses
    `collect(1:n)` for `n` inputs. For `length(V) ≥ 2`, `H` is only called with nonempty
    subsets. For `length(V) < 2` (which [`oinformation_gradients`](@ref) uses), `H` may be
    called with an empty `S` and must then return the additive zero (`0.0`, or a vector of
    zeros for a vector-valued `H`).

## Returns

A value of the same kind as those returned by `H` (a number for a scalar `H`, a vector of
the same length for a vector-valued `H`).

## Example

```julia
using Associations
x, y, z = rand(0:1, 1000), rand(0:1, 1000), rand(0:1, 1000)
est = JointProbabilities(OInformation(), CodifyVariables(UniqueElements()))
H = Associations.subset_entropies(est, x, y, z)
Associations.entropy_combination(OInformation(), H, [1, 2, 3]) # == association(est, x, y, z)
```
"""
function entropy_combination end

const HIGHER_ORDER_ESTIMATORS = Union{
    JointProbabilities{<:HigherOrderInformationMeasure},
    EntropyDecomposition{<:HigherOrderInformationMeasure},
}

function association(est::HIGHER_ORDER_ESTIMATORS, x...)
    return higher_order_association(est, x...)
end

function higher_order_association(est, x...)
    verify_number_of_inputs_vars(est.definition, length(x))
    H = subset_entropies(est, x...)
    return entropy_combination(est.definition, H, collect(1:length(x)))
end

include("TotalCorrelation.jl")
include("DualTotalCorrelation.jl")
include("OInformation.jl")
include("SInformation.jl")
include("oinformation_gradients.jl")
