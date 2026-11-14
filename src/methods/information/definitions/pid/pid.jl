export PartialInformationDecomposition, PIDResult
export RedundancyFunction, IMin, IMMI
export atoms, atom, synergy, redundancy, unique_information

include("lattice.jl")

"""
    RedundancyFunction

The supertype of redundancy functions ``I_\\cap(\\alpha; T)`` that define a
[`PartialInformationDecomposition`](@ref). A redundancy function gives the information that
every element of a collection ``\\alpha = \\{A_1, \\ldots, A_k\\}`` of source sets carries
about the target ``T``.

## Implementations

- [`IMin`](@ref): Williams & Beer's ``I_{min}``.
- [`IMMI`](@ref): the minimum mutual information, ``I_{MMI}``.

## Implementing a new redundancy function

Another package can add a redundancy function without changing Associations.jl:

1. Define a subtype, `struct MyRedundancy <: RedundancyFunction end` (it may have
    fields).
2. Extend [`Associations.pid_redundancies`](@ref) for it and for each estimator type it
    supports, returning the redundancy of every node of the redundancy lattice.

Then `PartialInformationDecomposition(MyRedundancy())` works with [`association`](@ref)
(returning a [`PIDResult`](@ref)), with [`PIDAtom`](@ref) and with
[`SurrogateAssociationTest`](@ref), for every estimator for which a method exists:
[`JointProbabilities`](@ref), [`EntropyDecomposition`](@ref), [`MIDecomposition`](@ref)
or [`GaussianDecomposition`](@ref). For estimators it does not support, add a method that
throws an informative `ArgumentError`; otherwise the user gets a `MethodError`.
The atoms are computed from the redundancies by [`Associations.mobius_inversion`](@ref),
and `show` for a `PIDResult` prints the redundancy function by its type name.

```julia
using Associations
import Associations: RedundancyFunction, pid_redundancies, lattice_node_vectors

struct MyRedundancy <: RedundancyFunction end

function pid_redundancies(::MyRedundancy, est::JointProbabilities, lattice, x...)
    sources, target = x[1:end-1], x[end]
    return map(lattice.nodes) do node
        source_sets = lattice_node_vectors(node) # e.g. [[1], [2, 3]]
        my_redundancy(est, source_sets, sources, target) # a Float64
    end
end
```

See the "Extending Associations" page of the documentation for the full contract.
"""
abstract type RedundancyFunction end

"""
    IMin <: RedundancyFunction
    IMin()

The redundancy function ``I_{min}`` of [Williams2010](@citet), the original redundancy
measure of partial information decomposition.

## Description

For a collection of source sets ``\\alpha = \\{A_1, \\ldots, A_k\\}`` and a target ``T``,

```math
I_{min}(\\alpha; T) = \\sum_{t} p(t) \\min_{A \\in \\alpha} I(T = t; A),
```

where the *specific information* that source set ``A`` carries about the outcome ``t`` is

```math
I(T = t; A) = \\sum_{a} p(a \\mid t) \\log \\frac{p(t \\mid a)}{p(t)}.
```

``I_{min}`` needs the probability of each target outcome, so it is only defined for
discrete data. Estimate it with [`JointProbabilities`](@ref), using any discretization:
continuous data must first be discretized, for example with
`CodifyVariables(ValueBinning(...))` or `CodifyVariables(OrdinalPatterns())`.

``I_{min}`` is known to overestimate redundancy: for two independent sources copied into
the target, ``T = (X_1, X_2)``, it gives 1 bit of redundancy and 1 bit of synergy instead
of 1 bit of unique information from each source.
"""
struct IMin <: RedundancyFunction end

"""
    IMMI <: RedundancyFunction
    IMMI()

The minimum mutual information redundancy function ``I_{MMI}`` [Barrett2015](@cite).

## Description

For a collection of source sets ``\\alpha = \\{A_1, \\ldots, A_k\\}`` and a target ``T``,

```math
I_{MMI}(\\alpha; T) = \\min_{A \\in \\alpha} I(A; T).
```

[Barrett2015](@citet) showed that for jointly Gaussian variables with two sources and a
univariate target, several redundancy functions, including ``I_{min}``, give this value.

``I_{MMI}`` only needs mutual informations, so it works with any estimator that gives them:

- [`JointProbabilities`](@ref) or discrete [`EntropyDecomposition`](@ref) for discrete or
    discretized data;
- [`GaussianDecomposition`](@ref) (optionally with the Gaussian copula) or
    [`EntropyDecomposition`](@ref) with a differential entropy estimator for continuous
    data;
- [`MIDecomposition`](@ref) with any [`MutualInformationEstimator`](@ref) of
    [`MIShannon`](@ref), such as [`KraskovStögbauerGrassberger1`](@ref), or
    [`GaussianMI`](@ref), which gives the same estimate as
    [`GaussianDecomposition`](@ref).

With estimated mutual informations, sampling error can make individual atoms slightly
negative.
"""
struct IMMI <: RedundancyFunction end

"""
    PartialInformationDecomposition <: MultivariateInformationMeasure
    PartialInformationDecomposition(redundancy = IMin(); base = 2)

The partial information decomposition (PID) of [Williams2010](@citet) of the information
that a set of sources carries about a target, using the given [`RedundancyFunction`](@ref).

## Usage

- Use with [`association`](@ref) to compute the decomposition from input data:
    `association(est, s₁, s₂, …, target)`, where `est` is an estimator whose
    `definition` is a `PartialInformationDecomposition`. The **last** input is the target
    and the others are the sources. The result is a [`PIDResult`](@ref).
- Use [`PIDAtom`](@ref) to compute a single atom as a number, for example to test the
    significance of synergy with a [`SurrogateAssociationTest`](@ref).

## Inputs

Between 2 and 5 sources, and one target. Each input can be a vector or a
`StateSpaceSet` (a group of variables treated as one multivariate variable).

## Description

The mutual information ``I(S_1, \\ldots, S_n; T)`` is split into *atoms*, one for each
node ``\\alpha`` of the redundancy lattice. A node is a collection of source sets, none of
which contains another, written e.g. ``\\{1\\}\\{23\\}``. Given a redundancy function
``I_\\cap``, the atoms ``\\Pi`` are defined by

```math
I_\\cap(\\alpha; T) = \\sum_{\\beta \\preceq \\alpha} \\Pi(\\beta; T),
```

and found by Möbius inversion over the lattice. For two sources there are four atoms:

- the redundancy ``\\{1\\}\\{2\\}``, information that either source alone provides;
- the unique information ``\\{1\\}`` and ``\\{2\\}``, provided by one source only;
- the synergy ``\\{12\\}``, provided only by both sources together.

The number of atoms grows quickly with the number of sources: 4, 18, 166 and 7579 for 2,
3, 4 and 5 sources.

## Compatible estimators

| Estimator | [`IMin`](@ref) | [`IMMI`](@ref) |
|---|:---:|:---:|
| [`JointProbabilities`](@ref) | ✓ | ✓ |
| [`EntropyDecomposition`](@ref) | | ✓ |
| [`GaussianDecomposition`](@ref) | | ✓ |
| [`MIDecomposition`](@ref) | | ✓ |

## Examples

```julia
using Associations
x, y = rand(0:1, 10_000), rand(0:1, 10_000)
t = x .⊻ y
est = JointProbabilities(PartialInformationDecomposition(IMin()), CodifyVariables(UniqueElements()))
res = association(est, x, y, t)
synergy(res) # ≈ 1 bit
```
"""
Base.@kwdef struct PartialInformationDecomposition{R<:RedundancyFunction, B} <: MultivariateInformationMeasure
    redundancy::R = IMin()
    base::B = 2
end
function PartialInformationDecomposition(redundancy::RedundancyFunction; base = 2)
    return PartialInformationDecomposition(redundancy, base)
end

min_inputs_vars(::PartialInformationDecomposition) = 3
max_inputs_vars(::PartialInformationDecomposition) = MAX_PID_SOURCES + 1

# ----------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------
"""
    PIDResult

The result of a [`PartialInformationDecomposition`](@ref), returned by
[`association`](@ref). Atoms are named by their lattice node, given as a collection of
source sets, e.g. `[[1], [2, 3]]` for ``\\{1\\}\\{23\\}``.

## Accessors

- [`synergy`](@ref)`(res)`: the atom of all sources together, ``\\{12 \\ldots n\\}``.
- [`redundancy`](@ref)`(res)`: the atom shared by all sources, ``\\{1\\}\\{2\\} \\ldots \\{n\\}``.
- [`unique_information`](@ref)`(res, i)`: the atom ``\\{i\\}``.
- [`atom`](@ref)`(res, α)`: any atom.
- [`atoms`](@ref)`(res)`: every atom, as `node => value` pairs.

The atoms sum to the mutual information between all sources and the target.
"""
struct PIDResult{D<:PartialInformationDecomposition, T<:Real}
    definition::D
    lattice::RedundancyLattice
    redundancies::Vector{T}
    atoms::Vector{T}
end

nsources(res::PIDResult) = res.lattice.nsources

"""
    atoms(res::PIDResult) → Vector{Pair}

All atoms of the decomposition, as `node => value` pairs, where `node` is a vector of
source sets. The nodes are ordered from the bottom of the lattice (redundancy) to the top
(synergy).
"""
function atoms(res::PIDResult)
    L = res.lattice
    return [lattice_node_vectors(L.nodes[j]) => res.atoms[j] for j in L.order]
end

"""
    atom(res::PIDResult, α)

The atom of node `α`, given as a collection of source sets: e.g. `atom(res, [[1], [2]])`
is the redundancy between sources 1 and 2, and `atom(res, [[1, 2]])` their synergy.
"""
atom(res::PIDResult, α) = res.atoms[lattice_node_index(res.lattice, α)]

"""
    synergy(res::PIDResult)

The atom ``\\{12 \\ldots n\\}``: the information about the target that only all the
sources together provide.
"""
synergy(res::PIDResult) = atom(res, [collect(1:nsources(res))])

"""
    redundancy(res::PIDResult)

The atom ``\\{1\\}\\{2\\} \\ldots \\{n\\}``: the information about the target that every
source provides on its own.
"""
redundancy(res::PIDResult) = atom(res, [[i] for i in 1:nsources(res)])

"""
    unique_information(res::PIDResult, i::Integer)

The atom ``\\{i\\}``: the information about the target that only source `i` provides.
"""
unique_information(res::PIDResult, i::Integer) = atom(res, [[i]])

function Base.show(io::IO, ::MIME"text/plain", res::PIDResult)
    n = nsources(res)
    r = nameof(typeof(res.definition.redundancy))
    println(io, "PIDResult: $n sources, redundancy function $r, base $(res.definition.base)")
    L = res.lattice
    nshow = length(L.order) ≤ 30 ? length(L.order) : 20
    for j in L.order[1:nshow]
        println(io, "  ", rpad(lattice_node_string(L.nodes[j]), 14), res.atoms[j])
    end
    if nshow < length(L.order)
        println(io, "  ⋮ ($(length(L.order) - nshow) more atoms; use `atoms(res)`)")
    end
    print(io, "  total (sum of atoms): ", sum(res.atoms))
end

# ----------------------------------------------------------------------------------------
# Redundancy values on every lattice node.
# ----------------------------------------------------------------------------------------
# Mutual information I(X_A; T) for a source subset A (a bitmask), where the target is
# the last input.

# From joint entropies of subsets of the inputs (estimators supported by
# `subset_entropies`): I(X_A; T) = H(X_A) + H(T) - H(X_A, T).
function source_target_mi(est, x...)
    t = length(x)
    H = subset_entropies(est, x...)
    return mask -> begin
        A = mask_to_indices(mask)
        H(A) + H((t,)) - H(vcat(A, t))
    end
end

# From a mutual information estimator.
function source_target_mi(est::MIDecomposition, x...)
    n = length(x) - 1
    mi_est = estimator_with_overridden_parameters(est.definition, est.est)
    S = StateSpaceSet.(x[1:n])
    T = x[end]
    cache = Dict{UInt, Float64}()
    return mask -> get!(cache, mask) do
        association(mi_est, StateSpaceSet(S[mask_to_indices(mask)]...), T)
    end
end

"""
    Associations.pid_redundancies(r::RedundancyFunction, est, lattice::RedundancyLattice, x...) → Vector{<:Real}

Return the redundancy ``I_\\cap(\\alpha; T)`` of every node ``\\alpha`` of `lattice`,
using the redundancy function `r` and the estimator `est`. Element `k` of the result is the
redundancy of the node `lattice.nodes[k]`.

This is the function to extend to add a new [`RedundancyFunction`](@ref) (see the section
"Implementing a new redundancy function" there). It is not exported; access it as
`Associations.pid_redundancies`, and extend it with
`function Associations.pid_redundancies(r::MyRedundancy, est::SomeEstimator, lattice, x...)`.

## Arguments

- `r`: the redundancy function, `est.definition.redundancy`.
- `est`: the estimator given to [`association`](@ref): a [`JointProbabilities`](@ref),
    [`EntropyDecomposition`](@ref), [`MIDecomposition`](@ref) or
    [`GaussianDecomposition`](@ref) whose `definition` is a
    [`PartialInformationDecomposition`](@ref). The unit of the redundancies is
    `est.definition.base` (bits for the default base 2). For example,
    `probabilities(est.discretization, x...)` gives the joint probability mass function
    of a `JointProbabilities` estimator, and [`Associations.subset_entropies`](@ref)`(est, x...)`
    gives joint entropies of subsets of the inputs (the target has index `length(x)`).
- `lattice`: the [`Associations.RedundancyLattice`](@ref) of `length(x) - 1` sources,
    from [`Associations.redundancy_lattice`](@ref). Use
    [`Associations.lattice_node_vectors`](@ref) to read a node as source sets.
- `x...`: the inputs as given to `association`: the sources `x[1:end-1]`, then the target
    `x[end]`. The number of inputs has already been checked (3 to 6).

## Returns

A `Vector{T}`, with a concrete `T <: Real` (normally `Float64`), of length
`length(lattice.nodes)`, in the order of `lattice.nodes`. Associations.jl then computes the
atoms with [`Associations.mobius_inversion`](@ref) and returns a [`PIDResult`](@ref) that
stores both the redundancies and the atoms.

## Built-in methods

- `pid_redundancies(::IMin, est::JointProbabilities, lattice, x...)`; other estimators
    throw an `ArgumentError`.
- `pid_redundancies(::IMMI, est, lattice, x...)` for every PID estimator.

A new method may call these, e.g. `Associations.pid_redundancies(IMMI(), est, lattice, x...)`.
"""
function pid_redundancies end

function pid_redundancies(::IMMI, est, lattice, x...)
    mi = source_target_mi(est, x...)
    return [minimum(mi, node) for node in lattice.nodes]
end

function pid_redundancies(::IMin, est::JointProbabilities, lattice, x...)
    pt, spec = specific_informations(est, x...)
    return map(lattice.nodes) do node
        s = 0.0
        for t in eachindex(pt)
            pt[t] > 0 || continue
            s += pt[t] * minimum(spec(A)[t] for A in node)
        end
        s
    end
end

function pid_redundancies(::IMin, est, lattice, x...)
    throw(ArgumentError(
        "IMin is defined for discrete data only. Use `JointProbabilities` with a " *
        "discretization, e.g. `JointProbabilities(PartialInformationDecomposition(IMin()), " *
        "CodifyVariables(ValueBinning(RectangularBinning(8))))`. " *
        "Got a $(nameof(typeof(est))) estimator."
    ))
end

# The target marginal p(t) and a function returning, for a source subset A (a bitmask),
# the specific informations I(T = t; X_A) for every target outcome t.
function specific_informations(est::JointProbabilities, x...)
    t = length(x)
    p = probabilities(est.discretization, x...)
    base = est.definition.base
    pt = vec(collect(marginal(p; dims = [t]).p))
    cache = Dict{UInt, Vector{Float64}}()
    spec(mask) = get!(cache, mask) do
        dims = vcat(mask_to_indices(mask), t)
        pAT = length(dims) == ndims(p) ? p.p : marginal(p; dims).p
        # Rows: joint outcomes of X_A; columns: outcomes of T (the last dimension).
        M = reshape(collect(pAT), :, length(pt))
        pa = vec(sum(M; dims = 2))
        out = zeros(length(pt))
        for j in eachindex(pt), i in axes(M, 1)
            pat = M[i, j]
            pat > 0 || continue
            out[j] += pat / pt[j] * log(base, pat / (pa[i] * pt[j]))
        end
        out
    end
    return pt, spec
end

function pid_association(est, x...)
    verify_number_of_inputs_vars(est.definition, length(x))
    lattice = redundancy_lattice(length(x) - 1)
    r = pid_redundancies(est.definition.redundancy, est, lattice, x...)
    if length(r) != length(lattice.nodes)
        throw(ArgumentError(
            "`pid_redundancies` for $(nameof(typeof(est.definition.redundancy))) must " *
            "return one value per lattice node ($(length(lattice.nodes))); got $(length(r))."
        ))
    end
    return PIDResult(est.definition, lattice, r, mobius_inversion(lattice, r))
end

const PID_ESTIMATORS = Union{
    JointProbabilities{<:PartialInformationDecomposition},
    EntropyDecomposition{<:PartialInformationDecomposition},
    MIDecomposition{<:PartialInformationDecomposition},
}
# `GaussianDecomposition` is defined later; its method is in GaussianDecomposition.jl.
association(est::PID_ESTIMATORS, x...) = pid_association(est, x...)

include("PIDAtom.jl")
