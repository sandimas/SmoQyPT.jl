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
    id_pair_array = []
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
            push!(id_pair_array,(phonon_ids[bond_tuple[2]],phonon_ids[bond_tuple[2]]))
            
        elseif config.ssh
            phonon_coupling = SSHCoupling(
                model_geometry = model_geometry,
                tight_binding_model = tight_binding_model,
                phonon_modes = (phonon_ids[bond_tuple[2]],phonon_ids[bond_tuple[3]]),
                bond = bond_tuple[1],
                α_mean = bond_tuple[4]
            )
            push!(id_pair_array,(phonon_ids[bond_tuple[2]],phonon_ids[bond_tuple[3]]))
            
            phonon_coupling_id = add_ssh_coupling!(
                electron_phonon_model = electron_phonon_model,
                ssh_coupling = phonon_coupling,
                tight_binding_model = tight_binding_model
            )
        end
    end

return electron_phonon_model, phonon_ids, (id_array...,), (id_pair_array...,)
end
