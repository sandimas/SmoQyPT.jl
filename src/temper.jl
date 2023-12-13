function temper_sym!(
        G, B, additional_info, fermion_greens_calculator,

        logdetG, sgndetG, 
        n_tier, shift_val, do_shift,
        fermion_path_integral, fermion_path_integral_tmp,
        electron_phonon_parameters, electron_phonon_parameters_tmp,
        config
    )

    for tier ∈ 0:n_tier-2
        p0("tier ",tier)
        MPI.Barrier(config.mpi_comm)    
        weights_r = zeros(Float64,4)
        if tier == config.mpi_rank_tier
            # sender
            # 0 - initialization
            
            receiver = config.n_walker_per_tier*(tier+1)  + ((config.mpi_rank_pID + shift_val) % config.n_walker_per_tier)
            
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
                additional_info["tempering_acceptance_rate"] += 1.0
                logdetG =logdetG_s
                sgndetG =sgndetG_s
                copyto!(G,G_s)
                copyto!(electron_phonon_parameters.x,electron_phonon_parameters_tmp.x)
                copyto!(fermion_path_integral.V,fermion_path_integral_tmp.V)
                copyto!(fermion_greens_calculator , fermion_greens_calculator_tmp)
                copyto!(B,B_s)
            end

        end


        if tier+1 == config.mpi_rank_tier
            # receiver
            # 0 - initialization
            sender = config.n_walker_per_tier*tier  + ((config.mpi_rank_pID - shift_val + config.n_walker_per_tier) % config.n_walker_per_tier)
            
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
                additional_info["tempering_acceptance_rate"] += 1.0
                logdetG=logdetG_r
                sgndetG=sgndetG_r
                copyto!(G,G_r)
                copyto!(electron_phonon_parameters.x,electron_phonon_parameters_tmp.x)
                copyto!(fermion_path_integral.V,fermion_path_integral_tmp.V)
                fermion_greens_calculator = dqmcf.FermionGreensCalculator( fermion_greens_calculator_tmp)
                copyto!(B,B_r)
            
            end

        end
        MPI.Barrier(config.mpi_comm)
    end
    shift_val = (do_shift) ? (shift_val + 1) % config.n_walker_per_tier : 0




    return (logdetG, sgndetG, shift_val)
end