

mutable struct ParallelTemperingConfig
    mpi_comm::MPI.Comm
    mpi_rank::Int
    mpi_n_rank::Int
    mpi_rank_tier::Int
    mpi_rank_pID::Int

    n_tier::Int
    n_walker_per_tier::Int

    symmetric::Bool
    checkerboard::Bool
    β::Number
    Δτ::Number
    n_stab::Int
    rng::AbstractRNG
    δG_max::Number
    Nt::Int
    nt::Int
    reg::Number
    avg_N
    U_table
    α_table

    model::String
    tempering_param::String

    hubbard::Bool
    holstein:: Bool
    ssh::Bool
end

allowed_tempering_params = [
    "α",
    "U"
]

allowed_models = [
    "Holstein",
    "Hubbard",
    "HubbardHolstein",
    "SSH",
    "HubbardSSH"
]