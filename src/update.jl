function do_updates_sym!(
        G,logdetG, sgndetG,δG,δθ,
        additional_info,
        tight_binding_parameters,
        electron_phonon_parameters,
        fermion_path_integral,
        fermion_greens_calculator,
        fermion_greens_calculator_alt,
        B, rng, id_tuple, hmc_updater,
        δG_max,  chemical_potential_tuner,
        config 
    )
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
    if (!isnothing(config.avg_N))
        logdetG, sgndetG = update_chemical_potential!(
            G, logdetG, sgndetG,
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B
        )
    end

    return (logdetG, sgndetG, δG, δθ)
end

function do_updates_asym!(
        G,logdetG, sgndetG,δG,δθ,
        electron_phonon_parameters,
        fermion_path_integral,
        fermion_greens_calculator,
        fermion_greens_calculator_alt,
        B, rng, id_tuple, hmc_updater,
        δG_max,  chemical_potential_tuner,
        additional_info 
    )


    return logdetGup, sgndetGup, logdetGdn, sgndetGdn
end