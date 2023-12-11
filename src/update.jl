function run_PQT(unit_cell,lattice,model_geometry,tight_binding_model,phonon_modes::Vector{PhononMode{E}},phonon_bonds,
                 α_table,U_table,model,β,Δτ,avg_N,μ,nt,Nt,N_bins,
                 checkerboard,symmetric,n_stab,δG_max,reg,
                 N_burnin,N_updates,Δt,measurement_table,tempering_param;base_seed=8675309,output_root_dir="./",do_shift=true) where {E <: AbstractFloat}

    
    # MPI inits
    MPI.Init()
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = MPI.Comm_rank(mpi_comm)
    mpi_n_rank = MPI.Comm_size(mpi_comm)

    # Catch bad inputs
    (N_updates % N_bins > 0 ) && shut_it_down("N_updates must be a multiple of N_bins")
    (model ∈ allowed_models) || shut_it_down(model, " is not a member of allowed \'model\'\n","Allowed values are ",allowed_models)
    (tempering_param ∈ allowed_tempering_params ) || shut_it_down(tempering_param, " is not a member of allowed \'tempering_param\'\n","Allowed values are ",allowed_tempering_params)

    # setups
    bin_size = div(N_updates, N_bins)
    seed = abs(base_seed+mpi_rank)
    rng = Xoshiro(seed)
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    # get N_tier
    n_tier = get_N_tier(tempering_param,α_table,U_table,mpi_n_rank)
    n_walker_per_tier = div(mpi_n_rank,n_tier)
    mpi_rank_tier = div(mpi_rank,n_walker_per_tier)
    mpi_rank_pID = mpi_rank%n_tier

    config = ParallelTemperingConfig(
        mpi_comm, mpi_rank, mpi_n_rank, mpi_rank_tier, mpi_rank_pID,
        n_tier, n_walker_per_tier,symmetric, checkerboard, β, Δτ,
        n_stab, rng, δG_max, Nt, nt, reg, U_table, α_table, model, tempering_param
    )

    # make sure tempering parameters and model match
    check_model_tempering(config)

    simulation_info = get_sim_info(config,output_root_dir)
    MPI.Barrier(mpi_comm)
    initialize_datafolder(simulation_info)

    # setup_phonons
    setup_phonons(config,phonon_modes,phonon_bonds,unit_cell)

    
    


    return nothing

end



function p0(text...)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(text...)
    end
end

function shut_it_down(text...)
    p0("Error: ",text...)
    MPI.Finalize()
    exit()
end

function get_N_tier(tempering_param,α_table,U_table,MPI_n_ranks)
    N_tier = 0
    if (tempering_param == "α")
        try
            N_tier = size(α_table,1)
        catch
            shut_it_down("α_table must be an array to anneal on the parameter")
        end
    elseif (tempering_param == "U")
        try
            N_tier = size(U_table,1)
        catch
            shut_it_down("U_table must be an array to anneal on the parameter")
        end
    end
    # ensure correct number of MPI ranks
    (MPI_n_ranks % N_tier != 0) && shut_it_down("Number of MPI ranks must be multiple of number of tempering parameters")
    return N_tier
end

function check_model_tempering(config)
    if (config.model == "Holstein") && (config.tempering_param == "U") ||
        (config.model == "Hubbard") && (config.tempering_param == "α") ||
        (config.model == "SSH") && (config.tempering_param == "U") 
        shut_it_down(config.model, " cannot temper on ", config.tempering_param)
    end
end


function get_sim_info(config,output_root_directory)

    prefix = config.model * "_" * config.tempering_param * "_" * string(config.mpi_rank_tier)
    return SimulationInfo(
        filepath = output_root_directory,                     
        datafolder_prefix = prefix,
        pID = config.mpi_rank_pID
    )
end


function setup_phonons(config,phonon_modes,phonon_bonds,unit_cell)
    n_orbital = unit_cell.n
    use_type = (eltype(phonon_bonds))
    shut_it_down(use_type)
    phonons = []
   # phonon_ids = Array{lu.Bond{2}}(undef,n_orbital)
    phonon_couplings = []
    phonon_coupling_ids = []
    
    n_phonon = size(phonons,1)
    n_bonds = size(phonon_bonds,1)
    (size(config.α_table,2) != n_bonds) && shut_it_down("size(α_table,2) must match size(n_bonds,1)")


    return nohing
end