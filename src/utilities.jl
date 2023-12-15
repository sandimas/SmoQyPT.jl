
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


function check_model_tempering(config)
    if (config.model == "Holstein") && (config.tempering_param == "U") ||
        (config.model == "Hubbard") && (config.tempering_param == "α") ||
        (config.model == "SSH") && (config.tempering_param == "U") 
        shut_it_down(config.model, " cannot temper on ", config.tempering_param)
    end
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

