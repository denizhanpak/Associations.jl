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
