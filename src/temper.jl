function temper_sym!(
        G, B, additional_info, fermion_greens_calculator,
        fermion_greens_calculator_alt,
        logdetG, sgndetG, 
        n_tier, shift_val, do_shift,
        fermion_path_integral,
        electron_phonon_parameters,
        config
    )
    epp = electron_phonon_parameters
    fpi = fermion_path_integral  
    x′ = similar(epp.x)          
    weights_r = zeros(Float64,4)
    G′ = fermion_greens_calculator_alt.G′

    for tier ∈ 0:n_tier-2
        
        MPI.Barrier(config.mpi_comm)    
# p0("tier ",tier)

        x_old = copy(epp.x)

        if tier == config.mpi_rank_tier
            # sender
            # 0 - initialization
            
            receiver = config.n_walker_per_tier*(tier+1)  + ((config.mpi_rank_pID + shift_val) % config.n_walker_per_tier)
            
            # 1 - swap X fields
            MPI.Send(epp.x,config.mpi_comm,dest=receiver)
            MPI.Recv!(x′,config.mpi_comm)
            # 2- calculate updated state
            Sb = SmoQyDQMC.bosonic_action(epp)
            
            SmoQyDQMC.update!(fpi, epp, epp.x, x′)
            
            Sb′ = SmoQyDQMC.bosonic_action(epp)
            calculate_propagators!(B,fpi, calculate_exp_K = config.ssh, calculate_exp_V = config.holstein)
            logdetG′, sgndetG′ = dqmcf.calculate_equaltime_greens!(G′, fermion_greens_calculator_alt,B)
            # 3 - receive weights
            MPI.Recv!(weights_r,config.mpi_comm)
            # 4 - calculate update 

            
p(0,Sb′, " ",logdetG′, " ",weights_r[4], " ",weights_r[3] )
p(0,Sb, " ",logdetG, " ",weights_r[2], " ",weights_r[1] )
            
            lnP = - Sb′ - 2.0* logdetG′ - weights_r[4] - 2.0 * weights_r[3]
            lnP += Sb + 2.0 * logdetG + weights_r[2] + 2.0 * weights_r[1]
            if isfinite(lnP)
                P = [lnP,log(rand(config.rng,Float64))]
            else
                P = [0.0,1.0]
                additional_info["NaN"] += 1.0
# println("rank ",config.mpi_rank, ",  lnP ",lnP)
            end
            # 5 send and process update
            MPI.Isend(P,config.mpi_comm,dest=receiver)
            if P[1] > P[2]
                additional_info["tempering_acceptance_rate"] += 1.0
                logdetG =logdetG′
                sgndetG =sgndetG′
                copyto!(G,G′)
                copyto!(fermion_greens_calculator , fermion_greens_calculator_alt)
            else
                update!(fpi, epp, epp.x, x_old)
               
                calculate_propagators!(B,fpi, calculate_exp_K = config.ssh, calculate_exp_V = config.holstein)
            end

        end


        if tier+1 == config.mpi_rank_tier
            # receiver
            # 0 - initialization
            sender = config.n_walker_per_tier*tier  + ((config.mpi_rank_pID - shift_val + config.n_walker_per_tier) % config.n_walker_per_tier)
            
            
            # 1 - swap X fields
            MPI.Recv!(x′,config.mpi_comm)
            MPI.Send(epp.x,config.mpi_comm,dest=sender)
            # 2 - calculate updated state
            weights_r[1] = logdetG
            weights_r[2] = SmoQyDQMC.bosonic_action(epp)
            
            update!(fpi, epp, epp.x, x′)
            
            weights_r[4] = SmoQyDQMC.bosonic_action(epp)
            calculate_propagators!(B,fpi, calculate_exp_K = config.ssh, calculate_exp_V = config.holstein)
            logdetG′, sgndetG′ = dqmcf.calculate_equaltime_greens!(G′, fermion_greens_calculator_alt,B)
            weights_r[3] = logdetG′

            # 3 - send weights from child to parent
            MPI.Send(weights_r,config.mpi_comm,dest=sender)
            # 4 - nothing
            # 5 - receive and do update
            P = zeros(Float64,2) 
            MPI.Recv!(P,config.mpi_comm)
            if P[1] > P[2]
                additional_info["tempering_acceptance_rate"] += 1.0
                logdetG =logdetG′
                sgndetG =sgndetG′
                copyto!(G,G′)
                copyto!(fermion_greens_calculator , fermion_greens_calculator_alt)
            else
                
                update!(fpi, epp, epp.x, x_old)
                calculate_propagators!(B,fpi, calculate_exp_K = config.ssh, calculate_exp_V = config.holstein)
            end

        end
        MPI.Barrier(config.mpi_comm)
    end # tier loop
    shift_val = (do_shift) ? (shift_val + 1) % config.n_walker_per_tier : 0


    return (logdetG, sgndetG, shift_val)
end