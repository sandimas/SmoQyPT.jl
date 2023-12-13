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
        n_stab, rng, δG_max, Nt, nt, reg, U_table, α_table, model, tempering_param,
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
        electron_phonon_parameters_tmp = ElectronPhononParameters(
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
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "seed" => seed,
        "dt" => Δt,
        "Nt" => Nt,
        "nt" => nt,
        "reg" => reg
    )
    
    measurement_container = setup_measurements(config,model_geometry,tight_binding_model,electron_phonon_model,measurement_list)
    initialize_measurement_directories( simulation_info = simulation_info, measurement_container = measurement_container )
    MPI.Barrier(mpi_comm)
    



    # setup DQMC simulation
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_tmp = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
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
        if config.holstein
            # shut_it_down(typeof(id_tuple),"\n",id_tuple)
            (accepted, logdetG, sgndetG) = reflection_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng, phonon_types = id_tuple
            )
            additional_info["phonon_acceptance_rate"] += accepted
        end
        if config.ssh 
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng, phonon_type_pairs = id_tuple
            )
            additional_info["phonon_acceptance_rate"] += accepted
        end
        
        if config.hubbard
            # TODO
        end
        if config.holstein || config.ssh
                (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng, initialize_force = true
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted
        end
        if (!isnothing(avg_N))
            logdetG, sgndetG = update_chemical_potential!(
                G, logdetG, sgndetG,
                chemical_potential_tuner = chemical_potential_tuner,
                tight_binding_parameters = tight_binding_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B
            )
        end

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
            if config.holstein

                (accepted, logdetG, sgndetG) = reflection_update!(
                    G, logdetG, sgndetG, electron_phonon_parameters,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                    B = B, rng = rng, phonon_types = id_tuple
                )
                additional_info["phonon_acceptance_rate"] += accepted
            end
            if config.ssh 
                (accepted, logdetG, sgndetG) = swap_update!(
                    G, logdetG, sgndetG, electron_phonon_parameters,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                    B = B, rng = rng, phonon_type_pairs = id_tuple
                )
                additional_info["phonon_acceptance_rate"] += accepted
            end
    
            if config.hubbard
                # TODO
            end
            if config.holstein 
                    (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                    G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                    B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng, initialize_force = true
                )
    
                # Record whether the HMC update was accepted or rejected.
                additional_info["hmc_acceptance_rate"] += accepted
            end
            if (!isnothing(avg_N))
                logdetG, sgndetG = update_chemical_potential!(
                    G, logdetG, sgndetG,
                    chemical_potential_tuner = chemical_potential_tuner,
                    tight_binding_parameters = tight_binding_parameters,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    B = B
                )
            end

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
        if do_swaps && bin!=N_bins
            for tier ∈ 0:n_tier-2
                p0("tier ",tier)
                MPI.Barrier(mpi_comm)    
                weights_r = zeros(Float64,4)
                if tier == mpi_rank_tier
                    # sender
                    # 0 - initialization
                    
                    receiver = n_walker_per_tier*(tier+1)  + ((mpi_rank_pID + shift_val) % n_walker_per_tier)
                    
                    G_s = similar(G)
                    B_s = similar(B)
                    initialize!(fermion_path_integral_tmp, electron_phonon_parameters_tmp)
                    # 1 - swap X fields
                    MPI.Send(electron_phonon_parameters.x,config.mpi_comm,dest=receiver)
                    MPI.Recv!(electron_phonon_parameters_tmp.x,config.mpi_comm)
                    # 2- calculate updated state
                    Sb_old = SmoQyDQMC.bosonic_action(electron_phonon_parameters)
                    update!(fermion_path_integral_tmp,electron_phonon_parameters_tmp,electron_phonon_parameters_tmp.x,electron_phonon_parameters.x)
                    Sb_new = SmoQyDQMC.bosonic_action(electron_phonon_parameters_tmp)
                    B_s = initialize_propagators(fermion_path_integral_tmp, symmetric=config.symmetric, checkerboard=config.checkerboard)
                    
                    fermion_greens_calculator_tmp = dqmcf.FermionGreensCalculator(B_s, config.β, config.Δτ, config.n_stab)
                    logdetG_s, sgndetG_s = dqmcf.calculate_equaltime_greens!(G_s, fermion_greens_calculator_tmp)
                    # 3 - receive weights
                    MPI.Recv!(weights_r,config.mpi_comm)
                    # 4 - calculate update 
                    # p0(Sb_new, " ",logdetG_s, " ",weights_r[4], " ",weights_r[3] )
                    # p0(Sb_old, " ",logdetG, " ",weights_r[2], " ",weights_r[1] )
                    
                    lnP = - Sb_new - 2.0* logdetG_s - weights_r[4] - 2.0 * weights_r[3]
                    lnP += Sb_old + 2.0 * logdetG + weights_r[2] + 2.0 * weights_r[1]
                    P_s = [lnP,log(rand(config.rng,Float64))]
                    # 5 send and process update
                    MPI.Isend(P_s,config.mpi_comm,dest=receiver)
                    if P_s[1] > P_s[2]
                        logdetG =logdetG_s
                        sgndetG =sgndetG_s
                        copyto!(G,G_s)
                        copyto!(electron_phonon_parameters.x,electron_phonon_parameters_tmp.x)
                        copyto!(fermion_path_integral.V,fermion_path_integral_tmp.V)
                        fermion_greens_calculator = dqmcf.FermionGreensCalculator( fermion_greens_calculator_tmp)
                        copyto!(B,B_s)
                    end

                end


                if tier+1 == mpi_rank_tier
                    # receiver
                    # 0 - initialization
                    sender = n_walker_per_tier*tier  + ((mpi_rank_pID - shift_val + n_walker_per_tier) % n_walker_per_tier)
                    
                    G_r = similar(G)
                    B_r = similar(B)
                    initialize!(fermion_path_integral_tmp, electron_phonon_parameters_tmp)
                    # 1 - swap X fields
                    MPI.Recv!(electron_phonon_parameters_tmp.x,config.mpi_comm)
                    MPI.Send(electron_phonon_parameters.x,config.mpi_comm,dest=sender)
                    # 2 - calculate updated state
                    weights_r[1] = logdetG
                    weights_r[2] = SmoQyDQMC.bosonic_action(electron_phonon_parameters)
                    update!(fermion_path_integral_tmp,electron_phonon_parameters_tmp,electron_phonon_parameters_tmp.x,electron_phonon_parameters.x)
                    weights_r[4] = SmoQyDQMC.bosonic_action(electron_phonon_parameters_tmp)
                    B_r = initialize_propagators(fermion_path_integral_tmp, symmetric=config.symmetric, checkerboard=config.checkerboard)
                    fermion_greens_calculator_tmp = dqmcf.FermionGreensCalculator(B_r, config.β, config.Δτ, config.n_stab)
                    logdetG_r, sgndetG_r = dqmcf.calculate_equaltime_greens!(G_r, fermion_greens_calculator_tmp)
                    weights_r[3] = logdetG_r

                    # 3 - send weights from child to parent
                    MPI.Send(weights_r,config.mpi_comm,dest=sender)
                    # 4 - nothing
                    # 5 - receive and do update
                    P_r = zeros(Float64,2) 
                    MPI.Recv!(P_r,config.mpi_comm)
                    if P_r[1] > P_r[2]
                        
                        logdetG=logdetG_r
                        sgndetG=sgndetG_r
                        copyto!(G,G_r)
                        copyto!(electron_phonon_parameters.x,electron_phonon_parameters_tmp.x)
                        copyto!(fermion_path_integral.V,fermion_path_integral_tmp.V)
                        fermion_greens_calculator = dqmcf.FermionGreensCalculator( fermion_greens_calculator_tmp)
                        copyto!(B,B_r)
                    
                    end
    
                end
                MPI.Barrier(mpi_comm)
            end
            shift_val = (shift_val + 1) % n_walker_per_tier

            # warms
            for n in 1:N_burnin_after_swap
                if config.holstein
                    # shut_it_down(typeof(id_tuple),"\n",id_tuple)
                    (accepted, logdetG, sgndetG) = reflection_update!(
                        G, logdetG, sgndetG, electron_phonon_parameters,
                        fermion_path_integral = fermion_path_integral,
                        fermion_greens_calculator = fermion_greens_calculator,
                        fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                        B = B, rng = rng, phonon_types = id_tuple
                    )
                    additional_info["phonon_acceptance_rate"] += accepted
                end
                if config.ssh 
                    (accepted, logdetG, sgndetG) = swap_update!(
                        G, logdetG, sgndetG, electron_phonon_parameters,
                        fermion_path_integral = fermion_path_integral,
                        fermion_greens_calculator = fermion_greens_calculator,
                        fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                        B = B, rng = rng, phonon_type_pairs = id_tuple
                    )
                    additional_info["phonon_acceptance_rate"] += accepted
                end
                
                if config.hubbard
                    # TODO
                end
                if config.holstein || config.ssh
                        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                        G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                        fermion_path_integral = fermion_path_integral,
                        fermion_greens_calculator = fermion_greens_calculator,
                        fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                        B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng, initialize_force = true
                    )

                    # Record whether the HMC update was accepted or rejected.
                    additional_info["hmc_acceptance_rate"] += accepted
                end
                if (!isnothing(avg_N))
                    logdetG, sgndetG = update_chemical_potential!(
                        G, logdetG, sgndetG,
                        chemical_potential_tuner = chemical_potential_tuner,
                        tight_binding_parameters = tight_binding_parameters,
                        fermion_path_integral = fermion_path_integral,
                        fermion_greens_calculator = fermion_greens_calculator,
                        B = B
                    )
                end

            end # n in 1:N_burnin_after_swap
            
        end # for bin in 1:N_bins
    end # if do_swaps
    MPI.Barrier(mpi_comm)

    # Calculate acceptance rates.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["phonon_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["local_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG
    if (mpi_rank_pID == 0)
        process_measurements(simulation_info.datafolder, N_bins)
    end
    return nothing
end



function p0(text...)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(text...)
    end
end

function p(i,text...)
    if MPI.Comm_rank(MPI.COMM_WORLD) == i
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




function setup_measurements(config,model_geometry,tight_binding_model,electron_phonon_model,measurement_table)
    measurement_container = initialize_measurement_container(model_geometry, config.β, config.Δτ)
    initialize_measurements!(measurement_container, tight_binding_model)
    (!isnothing(electron_phonon_model)) && initialize_measurements!(measurement_container, electron_phonon_model)

    for (m,measurement) ∈ enumerate(measurement_table)
        pairs = Vector{Tuple{Int64,Int64}}(undef,size(measurement[3]))
        for  y ∈ size(measurement[3])
            pairs[y] = (get_bond_id(model_geometry,measurement[3][y][1]),get_bond_id(model_geometry,measurement[3][y][2]))
        end
        
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = measurement[1],
            time_displaced = measurement[2],
            pairs = pairs
        )


    end
    
    return measurement_container
end





function setup_phonons!(model_geometry,config,tight_binding_model,phonon_modes)
    
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    n_bonds = size(config.α_table,2)
    
    α_tier = 0
    if (config.tempering_param == "α") ; α_tier = config.mpi_rank_tier; end

    phonon_ids = []

    for mode ∈ phonon_modes    
        push!(phonon_ids,add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = mode
        ))
    end
    id_array = []
    for bond ∈ 1:n_bonds
        
        bond_tuple = config.α_table[α_tier+1,bond]
        
        if config.holstein
            phonon_coupling = HolsteinCoupling(
                model_geometry = model_geometry,
                phonon_mode = phonon_ids[bond_tuple[2]],
                bond = bond_tuple[1],
                α_mean = bond_tuple[4]
            )
            push!(id_array,phonon_ids[bond_tuple[2]])
            phonon_coupling_id = add_holstein_coupling!(
                electron_phonon_model = electron_phonon_model,
                holstein_coupling = phonon_coupling,
                model_geometry = model_geometry
            )
        elseif config.ssh
            phonon_coupling = SSHCoupling(
                model_geometry = model_geometry,
                tight_binding_model = tight_binding_model,
                phonon_modes = (phonon_ids[bond_tuple[2]],phonon_ids[bond_tuple[3]]),
                bond = bond_tuple[1],
                α_mean = bond_tuple[4]
            )
            push!(id_array,(phonon_ids[bond_tuple[2]],phonon_ids[bond_tuple[3]]))
            
            phonon_coupling_id = add_ssh_coupling!(
                electron_phonon_model = electron_phonon_model,
                ssh_coupling = phonon_coupling,
                tight_binding_model = tight_binding_model
            )
        end
    end
    
    return electron_phonon_model, phonon_ids, (id_array...,)
end
