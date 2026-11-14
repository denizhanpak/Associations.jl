# ----------------------------------------------------------------------------------------
# The redundancy lattice of Williams & Beer (2010). `RedundancyLattice`,
# `redundancy_lattice`, `mobius_inversion` and `lattice_node_vectors` are part of the
# documented extension API; the other functions here are internal.
#
# A node is an antichain α = {A₁, …, Aₖ} of nonempty subsets of the sources, none of which
# contains another. Nodes are ordered by
#     α ≼ β  ⟺  for every B ∈ β there is an A ∈ α with A ⊆ B.
# Given a redundancy function I∩(α; T) on the nodes, the partial information atoms Π are
# its Möbius inversion: I∩(α) = Σ_{β ≼ α} Π(β).
#
# Source subsets are stored as bitmasks: source i is bit i - 1.
# ----------------------------------------------------------------------------------------

const MAX_PID_SOURCES = 5

"""
    Associations.RedundancyLattice

The redundancy lattice of [Williams2010](@citet) for a given number of sources. Obtain one
with [`Associations.redundancy_lattice`](@ref); do not construct or modify it directly.

A node ``\\alpha = \\{A_1, \\ldots, A_k\\}`` is a collection (an antichain) of nonempty
source sets, none of which contains another, e.g. ``\\{1\\}\\{23\\}``. Nodes are ordered by
``\\alpha \\preceq \\beta`` if and only if every ``B \\in \\beta`` contains some
``A \\in \\alpha``.

This type is part of the extension API (see the "Extending Associations" page of the
documentation). It is not exported. The fields below are stable and may be read directly.

## Fields

- `nsources::Int`: the number of sources `n`.
- `nodes::Vector{Vector{UInt}}`: the `m` nodes. Each node is a vector of source-set
    bitmasks, sorted in increasing order, where source `i` is bit `i - 1`
    (`UInt(1) << (i - 1)`); e.g. ``\\{1\\}\\{23\\}`` is `UInt[0b001, 0b110]`. Use
    [`Associations.lattice_node_vectors`](@ref) to convert a node to source indices. A node
    is referred to by its position `k` in `nodes`; the positions are the same every time
    the lattice for `n` sources is built, but carry no other meaning.
- `index::Dict{Vector{UInt}, Int}`: the position `k` of each node, `nodes[index[node]] == node`.
- `below::BitMatrix`: an `m × m` matrix with `below[i, j] == true` if and only if node `i`
    is strictly below node `j` (``\\alpha_i \\prec \\alpha_j``). This is the full order
    relation, not only the covering relation.
- `order::Vector{Int}`: a permutation of `1:m` in which every node comes after all the
    nodes below it. The first entry is the bottom node ``\\{1\\}\\{2\\} \\ldots \\{n\\}``
    and the last the top node ``\\{12 \\ldots n\\}``.
"""
struct RedundancyLattice
    nsources::Int
    # Each node is a sorted vector of source-subset bitmasks.
    nodes::Vector{Vector{UInt}}
    index::Dict{Vector{UInt}, Int}
    # below[i, j] is true if node i is strictly below node j.
    below::BitMatrix
    # Node indices in an order where every node comes after all the nodes below it.
    order::Vector{Int}
end

is_submask(a::UInt, b::UInt) = (a & b) == a
mask_to_indices(mask::UInt) = [i for i in 1:(8 * sizeof(UInt)) if (mask >> (i - 1)) & 1 == 1]
indices_to_mask(idxs) = reduce(|, (UInt(1) << (i - 1) for i in idxs); init = UInt(0))

# All antichains of nonempty subsets of n sources. Subsets are added in increasing mask
# order, so each antichain is generated exactly once.
function lattice_antichains(n::Int)
    subsets = UInt.(1:(2^n - 1))
    result = Vector{Vector{UInt}}()
    current = UInt[]
    function extend!(start)
        for k in start:length(subsets)
            s = subsets[k]
            if all(c -> !is_submask(c, s) && !is_submask(s, c), current)
                push!(current, s)
                push!(result, sort(current))
                extend!(k + 1)
                pop!(current)
            end
        end
    end
    extend!(1)
    return result
end

# α ≼ β in the redundancy lattice.
lattice_precedes(α, β) = all(B -> any(A -> is_submask(A, B), α), β)

function RedundancyLattice(n::Int)
    1 ≤ n ≤ MAX_PID_SOURCES || throw(ArgumentError(
        "The redundancy lattice is implemented for 1 to $MAX_PID_SOURCES sources; got $n."))
    nodes = lattice_antichains(n)
    m = length(nodes)
    below = falses(m, m)
    for j in 1:m, i in 1:m
        if i != j && lattice_precedes(nodes[i], nodes[j])
            below[i, j] = true
        end
    end
    # If α ≺ β, everything below α is also below β, so α has strictly fewer nodes below
    # it. Sorting by that count therefore gives a valid order for the Möbius inversion.
    order = sortperm(vec(sum(below; dims = 1)))
    index = Dict(node => k for (k, node) in enumerate(nodes))
    return RedundancyLattice(n, nodes, index, below, order)
end

const REDUNDANCY_LATTICES = Dict{Int, RedundancyLattice}()
const REDUNDANCY_LATTICES_LOCK = ReentrantLock()

"""
    Associations.redundancy_lattice(n::Int) → RedundancyLattice

The [`Associations.RedundancyLattice`](@ref) of `n` sources, for `1 ≤ n ≤ 5`. It has 1, 4,
18, 166 and 7579 nodes for `n` = 1, …, 5.

Each lattice is built once and cached (safely across threads), so every call with the same
`n` returns the same object; it must not be modified.

This function is part of the extension API (see the "Extending Associations" page of the
documentation). It is not exported; access it as `Associations.redundancy_lattice`.
"""
function redundancy_lattice(n::Int)
    lock(REDUNDANCY_LATTICES_LOCK) do
        get!(() -> RedundancyLattice(n), REDUNDANCY_LATTICES, n)
    end
end

"""
    Associations.mobius_inversion(lattice::RedundancyLattice, r::AbstractVector{T}) → Vector{T}

The partial information atoms ``\\Pi`` from the redundancies `r` of every node of
`lattice`: `r[k]` is the redundancy of `lattice.nodes[k]`, and the result has the atom of
`lattice.nodes[k]` at position `k`. The atoms solve

```math
r_j = \\Pi_j + \\sum_{i \\,:\\, \\alpha_i \\prec \\alpha_j} \\Pi_i
```

for every node ``j``.

`r` must have one-based indices and `length(r) == length(lattice.nodes)`. Its elements may
be of any type that supports `a + b`, `a - b` and `zero(a)`: real numbers, or, e.g., one
`Vector{Float64}` of per-sample values per node. The inversion is linear, so inverting
per-sample (local) redundancies and averaging the resulting atoms gives the inversion of
the averaged redundancies.

One call checks all ``m^2`` pairs of nodes, for `m = length(lattice.nodes)` (7579 for 5
sources), and does one `+` per pair of comparable nodes. With vector-valued elements, a
single call therefore replaces one call per sample.

This function is part of the extension API (see the "Extending Associations" page of the
documentation). It is not exported; access it as `Associations.mobius_inversion`.
"""
function mobius_inversion(lattice::RedundancyLattice, redundancies::AbstractVector{T}) where T
    m = length(redundancies)
    atoms = Vector{T}(undef, m)
    for j in lattice.order
        # `zero(redundancies[j])` rather than `zero(T)`, so that abstract and array element
        # types work.
        s = zero(redundancies[j])
        for i in 1:m
            lattice.below[i, j] && (s += atoms[i])
        end
        atoms[j] = redundancies[j] - s
    end
    return atoms
end

# Convert a user-given antichain, e.g. `[[1], [2, 3]]`, to a node index.
function lattice_node_index(lattice::RedundancyLattice, α)
    n = lattice.nsources
    masks = UInt[]
    for A in α
        idxs = collect(Int, A)
        isempty(idxs) && throw(ArgumentError("Each element of an atom must be a nonempty set of sources."))
        all(i -> 1 ≤ i ≤ n, idxs) || throw(ArgumentError(
            "Source indices must be between 1 and $n; got $idxs."))
        push!(masks, indices_to_mask(idxs))
    end
    node = sort(unique(masks))
    haskey(lattice.index, node) || throw(ArgumentError(
        "$(collect(α)) is not an antichain: no element may contain another."))
    return lattice.index[node]
end

# Literature notation: {1}{23} for the antichain {{1}, {2, 3}}.
lattice_node_string(node) = join("{" * join(mask_to_indices(A)) * "}" for A in node)
"""
    Associations.lattice_node_vectors(node) → Vector{Vector{Int}}

The source sets of a lattice node (an element of `lattice.nodes`, see
[`Associations.RedundancyLattice`](@ref)), each as a sorted vector of source indices,
e.g. `[[1], [2, 3]]` for ``\\{1\\}\\{23\\}``. The sets are in the order of their bitmasks
in `node`.

This function is part of the extension API (see the "Extending Associations" page of the
documentation). It is not exported; access it as `Associations.lattice_node_vectors`.
"""
lattice_node_vectors(node) = [mask_to_indices(A) for A in node]
