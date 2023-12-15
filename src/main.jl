function run_PQT(model_geometry,tight_binding_model,phonon_modes,
                α_table,U_table,model,β,Δτ,nt,Nt,N_bins,
                checkerboard,symmetric,n_stab,δG_max,reg,
                N_burnin,N_updates,N_burnin_after_swap,Δt,measurement_list,
                tempering_param,N;
                base_seed=8675309,output_root_dir="./",do_shift=true,
                avg_N=nothing,μ=0.0,do_swaps=true,
    ) 


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
    mpi_rank_pID = mpi_rank%n_walker_per_tier
    hubbard = !isnothing(findfirst("Hubbard",model))
    holstein = !isnothing(findfirst("Holstein",model))
    ssh = !isnothing(findfirst("SSH",model))

    config = ParallelTemperingConfig(
        mpi_comm, mpi_rank, mpi_n_rank, mpi_rank_tier, mpi_rank_pID,
        n_tier, n_walker_per_tier,symmetric, checkerboard, β, Δτ,
        n_stab, rng, δG_max, Nt, nt, reg,avg_N, U_table, α_table, model, tempering_param,
        hubbard, holstein, ssh
    )

    # make sure tempering parameters and model match
    check_model_tempering(config)

    simulation_info = get_sim_info(config,output_root_dir)
    MPI.Barrier(mpi_comm)
    initialize_datafolder(simulation_info)

    electron_phonon_model = nothing
    phonon_ids = nothing
    # setup_phonons
    if (config.model != "Hubbard") 
        electron_phonon_model, phonon_ids, id_tuple = setup_phonons!(model_geometry,config,tight_binding_model,phonon_modes)
    end
    MPI.Barrier(mpi_comm)

    # setup parameters for sim
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = config.rng
    )

    if (config.model != "Hubbard") 
        electron_phonon_parameters = ElectronPhononParameters(
            β = config.β, Δτ = config.Δτ,
            electron_phonon_model = electron_phonon_model,
            tight_binding_parameters = tight_binding_parameters,
            model_geometry = model_geometry,
            rng = config.rng
        )
    end
    # interactions = []
    # (model_includes(config,"Holstein")||model_includes(config,"SSH")) && push!(interactions,electron_phonon_model)
    if config.hubbard
        if config.model != "Hubbard"
            # hubbard + phonon
        else
            # Hubbard
        end
    else # just phonon
        interactions = (electron_phonon_model,)
    end


    model_summary(
        simulation_info = simulation_info,
        β = config.β, Δτ = config.Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = interactions
    )

    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "local_acceptance_rate" => 0.0,
        "hmc_acceptance_rate" => 0.0,
        "phonon_acceptance_rate" => 0.0,
        "local_acceptance_rate" => 0.0,
        "tempering_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "seed" => seed,
        "dt" => Δt,
        "Nt" => Nt,
        "nt" => nt,
        "reg" => reg,
        "NaN" => 0.0
    )

    measurement_container = setup_measurements(config,model_geometry,tight_binding_model,electron_phonon_model,measurement_list)
    initialize_measurement_directories( simulation_info = simulation_info, measurement_container = measurement_container )
    MPI.Barrier(mpi_comm)




    # setup DQMC simulation
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    
    initialize!(fermion_path_integral, electron_phonon_parameters)
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)




    G = zeros(eltype(B[1]), size(B[1]))
    logdetG = zero(Float64)
    sgndetG = zero(Float64)
    δG = zero(Float64)
    δθ = zero(Float64)
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)
    G_ττ = similar(G)
    G_τ0 = similar(G)
    G_0τ = similar(G)

    chemical_potential_tuner = nothing
    if (!isnothing(avg_N))
        chemical_potential_tuner = mt.MuTunerLogger(n₀ = avg_N, β = β, V = N, u₀ = 1.0, μ₀ = μ, c = 0.5)
    end

    hmc_updater = HMCUpdater(
    electron_phonon_parameters = electron_phonon_parameters,
    G = G, Nt = Nt, Δt = Δt, nt = nt, reg = reg
    )
    # shut_it_down(simulation_info,"\n\n\n",model_geometry)
    p0("Beginning warms")
    # warms
    for n in 1:N_burnin
        (logdetG, sgndetG, δG, δθ) = do_updates_sym!(
            G,logdetG, sgndetG,δG,δθ,
            additional_info,
            tight_binding_parameters,
            electron_phonon_parameters,
            fermion_path_integral,
            fermion_greens_calculator,
            fermion_greens_calculator_alt,
            B, rng, id_tuple, hmc_updater,
            δG_max,  chemical_potential_tuner,
            true,
            config 
        )

    end # n in 1:N_burnin


    # meas sweeps
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))
    coup_vec = []
    if config.holstein || config.ssh
        push!(coup_vec,electron_phonon_parameters)
    end
    if config.hubbard
        push!(coup_vec,hubbard_parameters)
    end
    coup_tuple = (coup_vec...,)
    shift_val = 0
    p0("Beginning sweeps")
    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in 1:N_bins

        # Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size
            (logdetG, sgndetG, δG, δθ) = do_updates_sym!(
                G,logdetG, sgndetG,δG,δθ,
                additional_info,
                tight_binding_parameters,
                electron_phonon_parameters,
                fermion_path_integral,
                fermion_greens_calculator,
                fermion_greens_calculator_alt,
                B, rng, id_tuple, hmc_updater,
                δG_max, chemical_potential_tuner,
                false,
                config 
            )

            

            (logdetG, sgndetG, δG, δθ) = make_measurements!(
                measurement_container,
                logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = coup_tuple
            )

        end # for n in 1:bin_size


        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
        p0("Finished Bin ", bin, " of ", N_bins)
        #do_tier_swap_updates
        
        if  do_swaps && bin!=N_bins

            (logdetG, sgndetG, shift_val) = temper_sym!(
                G, B, additional_info,
                fermion_greens_calculator, fermion_greens_calculator_alt,
                logdetG, sgndetG, 
                n_tier, shift_val, do_shift,
                fermion_path_integral,
                electron_phonon_parameters,
                config
            )
            for n ∈ 1:N_burnin_after_swap
                
                (logdetG, sgndetG, δG, δθ) = do_updates_sym!(
                        G,logdetG, sgndetG,δG,δθ,
                        additional_info,
                        tight_binding_parameters,
                        electron_phonon_parameters,
                        fermion_path_integral,
                        fermion_greens_calculator,
                        fermion_greens_calculator_alt,
                        B, rng, id_tuple, hmc_updater,
                        δG_max,  chemical_potential_tuner,
                        false,
                        config, sweep=n
                )
                
            end # n in 1:N_burnin_after_swap
        
        end # if do_swaps
        
    end # for bin in 1:N_bins

    save_simulation_info(simulation_info,additional_info)

    MPI.Barrier(mpi_comm)

    # Calculate acceptance rates.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["phonon_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["local_acceptance_rate"] /= (N_updates + N_burnin)
    if (mpi_rank_tier == 0 || mpi_rank_tier == n_tier-1) 
        additional_info["tempering_acceptance_rate"] /= (N_bins-1)
    else
        additional_info["tempering_acceptance_rate"] /= 2.0 * (N_bins-1)
    end
    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG
    if (mpi_rank_pID == 0)
        process_measurements(simulation_info.datafolder, N_bins)
    end
    return nothing
end

