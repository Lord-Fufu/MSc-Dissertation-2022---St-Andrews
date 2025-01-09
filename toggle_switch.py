import numpy as np
from numba import jit


@jit(nopython=True)
def generate_langevin_trajectory( duration = 720,
                                    repression_threshold = 10,
                                    hill_coefficient = 5,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 1,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    delta_t = 1.0,
                                    sampling_timestep = 1.0):
    '''Generate one trace of the Hes5 model. This function implements a stochastic version of
    the model model in Monk, Current Biology (2003). It applies the rejection method described
    in Cai et al, J. Chem. Phys. (2007) as Algorithm 2. This method is an exact method to calculate
    the temporal evolution of stochastic reaction systems with delay. At the end of the trajectory,
    transcription events will have been scheduled to occur after the trajectory has terminated.
    This function returns this transcription schedule, as well.

    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold 

    hill_coefficient : float

    degradation_rate : float

    basal_transcription_rate : float

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        trajectory in order to get rid of any overshoots, for example

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column number of A,
        third column is number of B
    '''
    total_time = duration + equilibration_time
    sample_times = np.arange(0, total_time, delta_t)
    full_trace = np.zeros((len(sample_times), 3))
    full_trace[:,0] = sample_times

    repression_threshold = float(repression_threshold)
    # inital_condition
    current_a = float(initial_a)
    current_b = float(initial_b)
    next_a = current_a
    next_b = current_b
    
    full_trace[0,1] = current_a
    full_trace[0,2] = current_b
    # basal_transcription_rate*= system_size
    # repression_threshold*=system_size
    
    for i, time in enumerate(sample_times[1:]):
        power_a = np.power(current_b/repression_threshold,hill_coefficient)
        hill_function_a = 1/(1+power_a)
        next_a += ((basal_transcription_rate*hill_function_a
                      - degradation_rate*current_a)*delta_t
                      + np.sqrt(((basal_transcription_rate*hill_function_a + degradation_rate*current_a)/system_size
                                  + 2*power_a*hill_function_a**3*basal_transcription_rate**2/switching_rate)*delta_t)
                                  *np.random.normal(0,1))
        power_b = np.power(current_a/repression_threshold,hill_coefficient)
        hill_function_b = 1/(1+power_b)
        next_b += ((basal_transcription_rate*hill_function_b
                      - degradation_rate*current_b)*delta_t
                      + np.sqrt(((basal_transcription_rate*hill_function_b + degradation_rate*current_b)/system_size
                                  + 2*power_b*hill_function_b**3*basal_transcription_rate**2/switching_rate)*delta_t)
                                  *np.random.normal(0,1))
        if next_a < 0:
            next_a = np.abs(next_a)
        if next_b < 0:
            next_b = np.abs(next_b)
        
        current_a = next_a
        current_b = next_b

        full_trace[i+1,1] = current_a
        full_trace[i+1,2] = current_b

    # get rid of the equilibration time now
    trace = full_trace[ full_trace[:,0]>=equilibration_time ]
    trace[:,0] -= equilibration_time

    # ensure we only sample every minute in the final trace
    if delta_t>=1.0:
        sampling_timestep_multiple = 1
    else:
        sampling_timestep_multiple = int(round(sampling_timestep/delta_t))

    trace_to_return = trace[::sampling_timestep_multiple]


    return trace_to_return

@jit(nopython = True)
def identify_reaction(random_number, base_propensity, propensities):
    '''Choose a reaction from a set of possiblities using a random number and the corresponding
    reaction propensities. To be used, for example, in a Gillespie SSA.

    This function will find j such that

    sum_0^(j-1) propensities[j] < random_number*sum(propensities) < sum_0^(j) propensities[j]

    Parameters
    ----------

    random_number : float
        needs to be between 0 and 1

    base_propensity : float
        the sum of all propensities in the propensities argument. This is a function argument
        to avoid repeated calculation of this value throughout the algorithm, which improves
        performance

    propensities : ndarray
        one-dimensional array of arbitrary length. Each entry needs to be larger than zero.

    Returns
    -------

    reaction_index : int
        The reaction index
    '''
    scaled_random_number = random_number*base_propensity
    propensity_sum = 0.0
    for reaction_index, propensity in enumerate(propensities):
        if scaled_random_number < propensity_sum + propensity:
            return reaction_index
        else:
            propensity_sum += propensity

    ##Make sure we never exit the for loop:
    raise(RuntimeError("This line should never be reached."))

@jit(nopython=True)
def generate_stochastic_trajectory( duration = 720,
                                    repression_threshold = 10,
                                    hill_coefficient = 5,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 1,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    sampling_timestep = 1.0):

    '''
    Parameters
    ----------

    duration : float
        duration of the trace in minutes

    repression_threshold : float
        repression threshold 

    hill_coefficient : float

    degradation_rate : float

    basal_transcription_rate : float

    equlibration_time : float
        add a neglected simulation period at beginning of the trajectory of length equilibration_time
        trajectory in order to get rid of any overshoots, for example

    Returns
    -------

    trace : ndarray
        2 dimensional array, first column is time, second column number of A,
        third column is number of B
    '''
    total_time = duration + equilibration_time
    sample_times = np.arange(equilibration_time, total_time, sampling_timestep)
    trace = np.zeros((len(sample_times), 3))
    trace[:,0] = sample_times

    repression_threshold = float(repression_threshold)
    # inital_condition
    current_a = float(initial_a)
    sigma_a = 1
    sigma_b = 1
    current_b = float(initial_b)
    
    basal_transcription_rate*= system_size
    repression_threshold*=system_size

    propensities = np.array([ basal_transcription_rate*sigma_a,
                              switching_rate*np.power(current_b/(repression_threshold),
                                                                      hill_coefficient), # a switching
                              initial_a*degradation_rate, 
                              basal_transcription_rate, # transcription
                              switching_rate*np.power(current_a/(repression_threshold),
                                                                      hill_coefficient), # b switching
                              initial_b*degradation_rate ] ) # mRNA degradation

    # set up the gillespie algorithm: We first

    # This following index is to keep track at which index of the trace entries
    # we currently are (see definition of trace above). This is necessary since
    # the SSA will calculate reactions at random times and we need to transform
    # calculated reaction times to the sampling time points
    sampling_index = 0

    time = 0.0
    while time < sample_times[-1]:
        base_propensity = (propensities[0] 
                           + propensities[1] 
                           + propensities[2] 
                           + propensities[3]
                           + propensities[4]
                           + propensities[5])

        # two random numbers for Gillespie algorithm
        first_random_number, second_random_number = np.random.rand(2)
        # time to next reaction
        time_to_next_reaction = -1.0/base_propensity*np.log(first_random_number)
        time += time_to_next_reaction
        # identify which of the four reactions occured
        reaction_index = identify_reaction(second_random_number, base_propensity, propensities)
        # execute reaction
        if reaction_index == 0: # a transcription 
            current_a +=1
            if sigma_b == 1:
                propensities[4] = switching_rate*np.power(current_a/repression_threshold, hill_coefficient)
            propensities[2] = current_a*degradation_rate
        elif reaction_index == 1: # a switching
            if sigma_a == 0:
                sigma_a = 1
                propensities[0] = basal_transcription_rate 
                propensities[1] = switching_rate*np.power(current_b/repression_threshold, hill_coefficient)
            else:
                sigma_a = 0
                propensities[0] = 0 
                propensities[1] = switching_rate
        elif reaction_index == 2: #a degradation
                current_a -=1
                if sigma_b == 1:
                    propensities[4] = switching_rate*np.power(current_a/repression_threshold, hill_coefficient)
                propensities[2] = current_a*degradation_rate
        elif reaction_index == 3: # b transcription
                current_b += 1
                if sigma_a == 1:
                    propensities[1] = switching_rate*np.power(current_b/repression_threshold, hill_coefficient)
                propensities[5] = current_b*degradation_rate
        elif reaction_index == 4: # b switching
            if sigma_b == 0:
                sigma_b = 1
                propensities[3] = basal_transcription_rate 
                propensities[4] = switching_rate*np.power(current_a/repression_threshold, hill_coefficient)
            else:
                sigma_b = 0
                propensities[3] = 0 
                propensities[4] = switching_rate
        elif reaction_index == 5: # b degradation
                current_b -= 1
                if sigma_a == 1:
                    propensities[1] = switching_rate*np.power(current_b/repression_threshold, hill_coefficient)
                propensities[5] = current_b*degradation_rate
        else:
            raise(RuntimeError("Couldn't identify reaction. This should not happen."))

        # update trace for all entries until the current time
        while ( sampling_index < len(sample_times) and
                time > trace[ sampling_index, 0 ] ):
            trace[ sampling_index, 1 ] = current_a
            trace[ sampling_index, 2 ] = current_b
            sampling_index += 1

    trace[:,0] -= equilibration_time
    trace[:,1:] /= system_size

#     if return_transcription_times:
#        return trace, delayed_transcription_times
#     else:
    return trace 

