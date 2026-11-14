using Test
using Associations
using Random
using Statistics

# Contract tests for the documented extension API (docs/src/api/extension_api.md), which
# packages building on Associations.jl rely on.

rng = Xoshiro(31415)
disc = CodifyVariables(UniqueElements())

# ---------------------------------------------------------------------------------------
# Higher-order measures: `subset_entropies` and `entropy_combination`.
# ---------------------------------------------------------------------------------------
# Plug-in local (per-sample) entropies -log2 p(x_S) of the inputs `x[S]`. Their mean is
# the plug-in entropy of `x[S]`.
function local_entropies(x, S)
    N = length(first(x))
    isempty(S) && return zeros(N)
    keys = [Tuple(x[i][k] for i in sort(collect(S))) for k in 1:N]
    counts = Dict{eltype(keys), Int}()
    for key in keys
        counts[key] = get(counts, key, 0) + 1
    end
    return [-log2(counts[key] / N) for key in keys]
end

@testset "Higher-order hooks" begin
    N = 500
    x = (rand(rng, 0:1, N), rand(rng, 0:2, N), rand(rng, 0:1, N), rand(rng, 0:1, N))
    defs = (TotalCorrelation(), DualTotalCorrelation(), OInformation(), SInformation())
    for def in defs
        est = JointProbabilities(def, disc)
        H = Associations.subset_entropies(est, x...)
        V = collect(1:length(x))
        scalar = Associations.entropy_combination(def, H, V)
        @test scalar ≈ association(est, x...)
        @test H(Int[]) == 0.0
        @test H((2, 1)) == H([1, 2])

        # A vector-valued callable gives the scalar result in every element.
        Hvec(S) = fill(H(S), 3)
        vec_result = Associations.entropy_combination(def, Hvec, V)
        @test vec_result isa Vector{Float64}
        @test length(vec_result) == 3
        @test all(vec_result .≈ scalar)
        # Also on a subset of the variables.
        @test all(Associations.entropy_combination(def, Hvec, [1, 3, 4]) .≈
            Associations.entropy_combination(def, H, [1, 3, 4]))

        # Local entropies give local values of the measure, which average to the global
        # (plug-in) value.
        Hlocal(S) = local_entropies(x, S)
        loc = Associations.entropy_combination(def, Hlocal, V)
        @test length(loc) == N
        @test mean(loc) ≈ scalar
    end

    # With fewer than two variables (as used by `oinformation_gradients`), an empty subset
    # can be requested; a vector-valued callable must return zeros for it.
    Hvec0(S) = local_entropies(x, S)
    @test all(abs.(Associations.entropy_combination(OInformation(), Hvec0, [2])) .< 1e-12)
end

# ---------------------------------------------------------------------------------------
# PID: an external redundancy function, defined through `pid_redundancies`.
# ---------------------------------------------------------------------------------------
# Reproduces I_MMI by calling the built-in method.
struct ToyRedundancy <: Associations.RedundancyFunction end
function Associations.pid_redundancies(::ToyRedundancy, est::JointProbabilities, lattice, x...)
    return Associations.pid_redundancies(IMMI(), est, lattice, x...)
end

# Returns the wrong number of values.
struct BadLengthRedundancy <: Associations.RedundancyFunction end
function Associations.pid_redundancies(::BadLengthRedundancy, est, lattice, x...)
    return zeros(length(lattice.nodes) - 1)
end

@testset "External redundancy function" begin
    toy = JointProbabilities(PartialInformationDecomposition(ToyRedundancy()), disc)
    mmi = JointProbabilities(PartialInformationDecomposition(IMMI()), disc)

    x1, x2 = [0, 0, 1, 1], [0, 1, 0, 1]
    N = 2000
    a, b, c = rand(rng, 0:1, N), rand(rng, 0:2, N), rand(rng, 0:1, N)
    t2 = mod.(a .+ b .+ (rand(rng, N) .< 0.2), 3)
    t3 = mod.(a .+ b .* c .+ (rand(rng, N) .< 0.2), 3)
    datasets = [(x1, x2, x1 .⊻ x2), (a, b, t2), (a, b, c, t3)]
    for x in datasets
        res_toy = association(toy, x...)
        res_mmi = association(mmi, x...)
        @test res_toy isa PIDResult
        @test res_toy.redundancies == res_mmi.redundancies
        @test first.(atoms(res_toy)) == first.(atoms(res_mmi))
        @test all(last.(atoms(res_toy)) .≈ last.(atoms(res_mmi)))
        @test synergy(res_toy) ≈ synergy(res_mmi)
        @test redundancy(res_toy) ≈ redundancy(res_mmi)
    end
    @test synergy(association(toy, x1, x2, x1 .⊻ x2)) ≈ 1.0

    # `PIDAtom` routes to the external method.
    pid = PartialInformationDecomposition(ToyRedundancy())
    for at in (:synergy, :redundancy, :unique => 1, [[1], [2]])
        est_toy = JointProbabilities(PIDAtom(pid, at), disc)
        est_mmi = JointProbabilities(PIDAtom(PartialInformationDecomposition(IMMI()), at), disc)
        @test association(est_toy, a, b, t2) ≈ association(est_mmi, a, b, t2)
    end

    # Printing names the external redundancy function.
    @test occursin("ToyRedundancy", repr(MIME("text/plain"), association(toy, a, b, t2)))
    # Estimators without a method are not silently accepted.
    @test_throws MethodError association(
        GaussianDecomposition(PartialInformationDecomposition(ToyRedundancy())),
        randn(rng, 100), randn(rng, 100), randn(rng, 100))
    # A wrong number of redundancies is caught.
    bad = JointProbabilities(PartialInformationDecomposition(BadLengthRedundancy()), disc)
    @test_throws ArgumentError association(bad, a, b, t2)
end

# ---------------------------------------------------------------------------------------
# PID: the redundancy lattice and the Möbius inversion, applied per sample.
# ---------------------------------------------------------------------------------------
@testset "Lattice and Möbius inversion hooks" begin
    for n in 2:4
        L = Associations.redundancy_lattice(n)
        m = length(L.nodes)
        @test L === Associations.redundancy_lattice(n)
        @test L.nsources == n
        @test size(L.below) == (m, m)
        @test sort(L.order) == 1:m
        @test all(L.index[node] == k for (k, node) in enumerate(L.nodes))
        @test Associations.lattice_node_vectors(L.nodes[first(L.order)]) == [[i] for i in 1:n]
        @test Associations.lattice_node_vectors(L.nodes[last(L.order)]) == [collect(1:n)]

        # Per-sample (local) redundancies: one column per sample.
        Nsamples = 50
        R = randn(rng, m, Nsamples)
        A = reduce(hcat, [Associations.mobius_inversion(L, r) for r in eachcol(R)])
        @test size(A) == (m, Nsamples)
        @test vec(mean(A; dims = 2)) ≈ Associations.mobius_inversion(L, vec(mean(R; dims = 2)))
        # The same in one call, with one vector of per-sample values per node.
        Avec = Associations.mobius_inversion(L, [R[k, :] for k in 1:m])
        @test reduce(hcat, Avec)' ≈ A
        # Abstract and integer element types.
        @test Associations.mobius_inversion(L, Real[r for r in R[:, 1]]) ≈ A[:, 1]
        @test Associations.mobius_inversion(L, ones(Int, m)) ==
            [k == first(L.order) ? 1 : 0 for k in 1:m]
    end
end
