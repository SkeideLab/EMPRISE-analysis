import numpy as np
import random
import pandas as pd
import sys


class NoOtherItemsError(Exception):
    """Error that shows that list has only 1 item and randomly choosing will not produce something else
    """
    pass


def generate_stimulus_sequence(num_blocks, numerosity, stimulus_types):
    """Generates a sequence of stimulus conditions

    :param num_blocks: number of blocks in sequence
    :type num_blocks: int
    :param numerosity: list of numerosities for stimuli
    :type numerosity: list
    :param stimulus_types: modalities of stimuli
    :type stimulus_types: list
    :return: sequence of stimuli conditions, list of possible numerosity/type combinations
    :rtype: list, list
    """
    # Now that the number of blocks is known we can distribute it among numerosities
    # stimulus types
    # list(it.product(numerosity, stimulus_types))
    combinations = [str(x)+'_'+y for x in numerosity for y in stimulus_types]
    stimuli_run = []

    for k in range(int(num_blocks//len(combinations))):
        # random.shuffle(combinations)
        stimuli_run.extend(combinations)

    if len(stimuli_run) < num_blocks:
        random.shuffle(combinations)
        stimuli_run.extend(combinations[: int(num_blocks - len(stimuli_run))])
    # optional test
    random.shuffle(stimuli_run)
    return stimuli_run


def generate_stimulus_sequence_modalitypure(num_blocks, numerosity, balance):
    """Generates a sequence of stimulus conditions

    :param num_blocks: number of blocks in sequence
    :type num_blocks: int
    :param numerosity: list of numerosities for stimuli
    :type numerosity: list
    :param stimulus_types: modalities of stimuli
    :type stimulus_types: list
    :return: sequence of stimuli conditions, list of possible numerosity/type combinations
    :rtype: list, list
    """
    # Now that the number of blocks is known we can distribute it among numerosities
    # stimulus types
    # list(it.product(numerosity, stimulus_types))
    numerosities = [str(x)+'_' for x in numerosity]
    stimuli_run = []

    for k in range(int(num_blocks//len(numerosities))):
        # random.shuffle(combinations)
        stimuli_run.extend(numerosities)

    if len(stimuli_run) < num_blocks:
        random.shuffle(numerosities)
        stimuli_run.extend(numerosities[: num_blocks - len(stimuli_run)])
    # optional test
    random.shuffle(stimuli_run)

    blocks_pmod = num_blocks/3
    for block_i in range(len(stimuli_run)):
        if block_i < blocks_pmod:
            stimuli_run[block_i] = stimuli_run[block_i]+balance[0]
        elif block_i < blocks_pmod*2:
            stimuli_run[block_i] = stimuli_run[block_i]+balance[1]
        elif block_i < blocks_pmod*3:
            stimuli_run[block_i] = stimuli_run[block_i]+balance[2]
        else:
            print('something wrong with stimuli generation')

    return stimuli_run

# FIXME make this into two functions: one to produce the total sequnce, the other to add onsets


def make_onsets_df(run_duration, block_duration, stimuli_run, num_blocks, generated_pauses):
    """Assign onset times to stimuli and pauses.

    :param run_duration: run duration in s
    :type run_duration: float
    :param block_duration: block duration in s
    :type block_duration: float
    :param stimuli_run: sequence of stimuli blocks to run
    :type stimuli_run: list
    :param num_blocks: number of blocks in sequence
    :type num_blocks: int
    :param generated_pauses: sequence of pause durations
    :type generated_pauses: list
    :return: dataframe with onset, duration, trial_type for each block
    :rtype: pandas dataframe
    """
    # %% Now we must get the timings of the stimulus presentation and fixation/pauses
    elapsed_time = 0
    onsets = []
    i = 0
    j = 0
    # while(elapsed_time <= run_duration):
    while np.absolute((run_duration-5)-elapsed_time) > 0.5:
        if i < num_blocks:
            onsets.append([elapsed_time, block_duration, stimuli_run[i]])
            i += 1
            elapsed_time += block_duration
        if j < len(generated_pauses):
            onsets.append([elapsed_time, generated_pauses[j], f"pause"])
            elapsed_time += generated_pauses[j]
            j += 1
        if elapsed_time == sum(generated_pauses) + num_blocks * block_duration:
            break
        #print('Elapsed time', elapsed_time)

    events = pd.DataFrame(onsets, columns=['onset', 'duration', 'trial_type'])
    events['onset'] = events['onset']+5
    return events


# def make_dm(onsets, tr, num_frames):
    # """Make design matrix out of pandas dataframe .

    #:param onsets: pandas dataframe with columns duration, onset, trial_type
    #:type onsets: pandas dataframe
    #:param tr: repetition time of fMRI
    #:type tr: float
    #:param num_frames: number of TRs in fMRI sequence
    #:type num_frames: int
    #:return: nilearn design matrix
    #:rtype: nilearn design matrix
    # """
    #events = onsets.drop(onsets[onsets['trial_type'].str.contains('pause')].index)
    #frame_times = np.arange(num_frames)*tr + tr*0.5

    # dm = make_first_level_design_matrix(frame_times, events,
    # drift_model=None)
    # drop constant
    #dm.drop(labels='constant', axis=1, inplace=True)
    # return dm


def make_contrasts(design_matrix):
    """Make contrast for design matrix for every trial_type against rest

    :param design_matrix: design matrix of GLM
    :type design_matrix: nilearn design matrix
    :return: contrast for each design matrix column/trial_type
    :rtype: dictionary
    """
    contrast_matrix = np.eye(design_matrix.shape[1])

    # make one vector per regressor
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    return basic_contrasts


def check_n_matches_string(stimuli_run, n=3):
    """We want to avoid 3 subsequent combinations
    containing the same modality or the same numerosity
    example: (1, dot), (2, dot), (3, dot) (4, dot) --> to be avoided
    :param stimuli_run: the sequence of presented stimuli
    :type stimuli_run: list of tuples
    :param n: all matches >= this number are to be avoided
    :type n: int
    :return: True if there are no 'n' subsequent matches in sequence else False
    :rtype: bool
    """
    for i in range(len(stimuli_run) - (n+1)):
        count_numerosity = 0
        count_modality = 0
        for j in range(n):
            idx1, idx2 = i+j, i+j+1
            if stimuli_run[idx1].split('_')[0] == stimuli_run[idx2].split('_')[0]:
                count_numerosity += 1  # means the numerosities are matching
            # if stimuli_run[idx1].split('_')[1] == stimuli_run[idx2].split('_')[1]:
            #     count_modality += 1
        if count_modality == n or count_numerosity == n:
            return False
    return True


def add_ll_vis_conds(onsets, samples_per_block):
    """Add low level visual conditions to onsets dataframe

    :param onsets: dataframe with columns duration, trial_type, onset
    :type onsets: pandas dataframe
    :return: dataframe with new column 'll_cond' with string of ll conditions per block, samples divided by '_'
    :rtype: pandas dataframe
    """
    # low level vis conds: ['standard', 'const_circ', 'linear', 'density']
    # how many 'dot' conditions?
    shapes = ['dot', 'rectangle', 'triangle']

    n_dot_trials = np.sum(onsets['trial_type'].str.contains('dot'))
    llconds = ['standard', 'constcirc', 'linear', 'density']

    ll_shapes = [(x, y) for x in shapes for y in llconds]

    wholes, rest = divmod(n_dot_trials*samples_per_block, len(ll_shapes))

    # make list with as many members as number of dot trial samples needed
    ll_seq = ll_shapes*wholes + [ll_shapes[random.randrange(len(ll_shapes))]
                                 for _ in range(rest)]
    random.shuffle(ll_seq)

    # form into sublists of length samples_per_block
    ll_seq_blocks = [ll_seq[x: x+samples_per_block]
                     for x in range(0, len(ll_seq), samples_per_block)]
    ll_blocks = [[tup[1] for tup in block] for block in ll_seq_blocks]
    shape_blocks = [[tup[0] for tup in block] for block in ll_seq_blocks]

    # add each sublist to a dot condition block
    onsets.loc[onsets['trial_type'].str.contains(
        'dot'), ['ll_cond']] = ['_'.join(block) for block in ll_blocks]
    onsets.loc[onsets['trial_type'].str.contains(
        'dot'), ['shape']] = ['_'.join(block) for block in shape_blocks]
    return onsets


def add_gender_voice(onsets, samples_per_block, block_duration):
    """Add new column to onsets dataframe 'gender_voice' for audio trials.

    :param onsets: dataframe with columns 'duration','onset', 'trial_type'
    :type onsets: pandas dataframe
    :return: dataframe with new column 'gender_voice', values 'male' or 'female' for audio trials and 'na' else
    :rtype: pandas dataframe
    """
    sample_dur = block_duration/samples_per_block

    audio_is = onsets.index[onsets['trial_type'].str.contains('audio')]
    index_audio_is = np.arange(len(audio_is))

    onsets['gender_voice'] = 'na'

    max_time_change = 3
    min_time_change = 2
    gender = ['male', 'female']
    random.shuffle(gender)

    for audioblock_i in audio_is:

        reps_same_gender = [
            limit//sample_dur for limit in [min_time_change, max_time_change]]

        voices = []
        while len(voices) < samples_per_block:
            # FIXME shuffling list is computationally expensive
            # alternatives to shuffling: permutations of the list and sample one of these, using re module
            gender.append(gender.pop(0))
        # Do this until all samples are full
            voices.extend(
                [gender[0]]*np.random.random_integers(reps_same_gender[0], reps_same_gender[1]))

        voices = voices[:samples_per_block]
        onsets.loc[audioblock_i, 'gender_voice'] = '_'.join(voices)
    return onsets


def check_match_distr(freq_matches, stimuli_run):
    """Check distribution of matches over numerosities. Return false if not evenly distributed.

    :param freq_matches: list of block with '1' if numerosity matches with next block, 0 else
    :type freq_matches: list
    :param stimuli_run: sequence of stimuli conditions
    :type stimuli_run: list
    :return: False if not evenly distributed (not a good random distribution), True else
    :rtype: boolean
    """
    matches_per_num = [stim.split('_')[0] for stim, match in zip(
        stimuli_run, freq_matches) if match]
    matches_count_per_num = [(num, matches_per_num.count(num))
                             for num in set(matches_per_num)]
    num_matches = sum(freq_matches)
    match_freq_per_num = num_matches/3
    if ['not evenly distributed' for count in matches_count_per_num if np.absolute(count[1] - match_freq_per_num) > 1]:
        return False
    else:
        return True


def generate_pause_sequence(num_pauses, avg_pause=None, pauses=None):

    generated_pauses = []

    if not pauses:
        factors = [0.5, 1, 1.5]
        pauses = [factor*avg_pause for factor in factors]

    # generate pauses, make sure evenly distribution of lengths and mean is that of average pause
    # draw the pause lengths equally from all 3 possible pause lengths
    each_pause_reps = int(num_pauses // 3)
    # fill up additional pauses with the average pause
    additional_pauses_n = int(num_pauses % 3)

    generated_pauses = pauses * each_pause_reps + \
        [pauses[1]] * additional_pauses_n
    np.random.shuffle(generated_pauses)

    return generated_pauses


def generate_four_in_a_row(conditions, n):
    # generates a list of conditions (prime or no_prime) without four identical conditions in a row
    # n is n_trials/2 because we have 2 conditions and each appears half the time

    found = False
    combos_to_check = [[condition]*4 for condition in conditions]

    while not found:

        print('finding prime sequence new try')
        conditions_pool = conditions*(n//2)
        stack = []
        trial_counter = 0

        while trial_counter < 2000 and not found:

            trial_counter += 1
            condition = np.random.choice(conditions_pool, replace=False)
            stack.append(condition)
            if len(stack) >= 4:
                if stack[-4:] in combos_to_check:  # do this recursively for 4 combinations
                    conditions_pool.append(stack.pop())

            if len(stack) == n:
                found = True

    if four_in_a_row(stack):
        print('Condition Not found')
    else:
        return stack


def four_in_a_row(sequence):
    # checks if there are 4 identical conditions ('prime' or 'nonprime') in a row
    for i in range(len(sequence)-4):
        if len(np.unique(sequence[i:i+4])) == 1:
            return True

    return False


def check_constraints_satisfied(i, trial_list, target, prime):
    """Check that constraints for numerosity sequence are satisfied. Nonprime trials must not have identical prime and target. Numerosity must not appear in more than 3 successive trials.

    Parameters
    ----------
    i : int
        index of trial_list for trial that is being filled 
    trial_list : list
        contains strings with trial descriptions containing numbers
    target : int
        target numerosity
    prime : int
        prime numerosity

    Returns
    -------
    boolean
        False if the constraints are violated, else True 
    """
    # prime and target must not be identical
    if prime == target:
        return False
    # prime and target numerosity must not appear in more than 3 surrounding trials
    for num in [target, prime]:
        for j in range(i-4, i+1):
            counter = 0
            if j in range(0, len(trial_list)) and j+4 in range(0, len(trial_list)):
                for numerosity_string in trial_list[j: j+4]:
                    if str(num) in numerosity_string:
                        counter += 1
                # our current trial is still empty, so all other 3 trials containing numerosity violates constraint
                if counter >= 3:
                    return False
    return True


def create_priming_sequence(modalities, duration=300, tr=2.1):
    """look at constraints in google doc to make sequence

    Parameters
    ----------
    modality : _type_
        _description_
    duration : int, optional
        _description_, by default 300
    tr : float, optional
        _description_, by default 2.1
    returns: pandas dataframe with at least columns duration, onset, 
    trial_type (of the type <mod1>-<primingnumber>_<mod2>-<targetnumber>). Can have more columns if necessary 
    (but I don't think so right now? For another example run create_random_blocks()).
    Will have two rows per trial: 1 row for the numbers with the trialtype mentioned above, 1 row for the response period, trialtype 'pause'.
    This is so it can be easily integrated into our code for pause and trial. I will write a 'present' function that does the trial presentation.
    For others ideas please mention them. The code in this fct until now is a start and can be changed/replaced in any way.
    """
    # total duration = 45 mins = 2700 s
    # trial duration = 2 s, including fixation and pause?? correct
    # each run shall vbe of 5 mins
    mod_prime = modalities[0]
    mod_target = modalities[1]
    dur_trial = 2
    duration = duration - tr
    n_trial = np.floor(duration/dur_trial)

    nums = list(np.arange(1, 6))

    conditions = ['prime', 'nonprime']
    # loop through the sequence and check this?
    # must have even number of trials so equal number of prime/nonprime
    if n_trial % 2 != 0:
        n_trial -= 1
    n_trial = int(n_trial)
    print(n_trial)

    # ------------logic--------------------
    # 1. get sequence of (non-)prime trials
    # 2. fill prime trials first, make sure each numerosity has same number of trials
    # 3. fill nonprime trials, make sure each of the numerosities appears as prime and target for the same number of trials
    # 4. constraints for nonprime: make sure prime and target are not equal for this, numerosity does not appear for more than 3 trials in a row

    # get sequnce of conditions: each trial is a prime or nonprime trial
    conditions_seq = generate_four_in_a_row(conditions, n_trial)
    trial_list = []

    # number of trials each numerosity appears during prime trials
    n_prime_trials_per_num = (n_trial//2)//5

    # get pools for prime and nonprime(target and prime) trials to draw numbers from
    prime_trials_pool = nums * n_prime_trials_per_num
    np.random.shuffle(prime_trials_pool)

    nonprime_trials_prime_pool = nums * n_prime_trials_per_num
    np.random.shuffle(nonprime_trials_prime_pool)

    nonprime_trials_target_pool = nums * n_prime_trials_per_num
    np.random.shuffle(nonprime_trials_target_pool)

    for condition in conditions_seq:
        if condition == 'prime':
            # try to pop numerosity,
            # if list is empty (n_trial/2 not divisible by 5) then choose randomly
            try:
                prime = prime_trials_pool.pop()
            except IndexError:
                prime = np.random.choice(nums)
            trial_list.append(f'{mod_prime}-{prime}_{mod_target}-{prime}')
        else:
            trial_list.append('nonprime')

    for i, trial in enumerate(trial_list):
        if trial == 'nonprime':
            # get target and prime
            # if list is empty, choose random numerosity
            try:
                target = nonprime_trials_target_pool.pop(0)
                prime = nonprime_trials_prime_pool.pop(0)

                # check that contraints satisfied
                counter = 0
                while not check_constraints_satisfied(i, trial_list, target, prime):
                    counter += 1
                    # if not, then replace target and prime and draw new from list
                    nonprime_trials_prime_pool.append(prime)
                    nonprime_trials_target_pool.append(target)
                    target = np.random.choice(
                        nonprime_trials_target_pool, replace=False)
                    prime = np.random.choice(
                        nonprime_trials_prime_pool, replace=False)

                    if counter > 20:
                        # infinite loop where the remaining numerosities do not fit
                        raise NoOtherItemsError()
            except (IndexError, NoOtherItemsError) as err:
                prime = np.random.choice(nums)
                target = np.random.choice(nums)

                while not check_constraints_satisfied(i, trial_list, target, prime):
                    prime = np.random.choice(nums)
                    target = np.random.choice(nums)

            trial_list[i] = f'{mod_prime}-{prime}_{mod_target}-{target}'
            #print(trial_list)
    # add pause durations to each trial
    duration = [1.05, 0.95] * len(trial_list)
    tl_withpauses = []
    for trial in trial_list:
        tl_withpauses.extend([trial, 'pause'])

    df = pd.DataFrame({'duration': duration, 'trial_type': tl_withpauses})

    return df


def create_random_blocks(avg_pause=None, block_duration=6, run_duration=175, samples_per_block=6, pauses=[2, 2.5, 3]
                         ):
    # -----------prepare simulations-------------------
    stimulus_types = ['audio', 'dot', 'digit']
    numerosity = [2, 3, 4]

    if not avg_pause:
        if len(pauses) == 3:
            avg_pause = pauses[1]
        else:
            print(
                'pauses list must have length of 3 with middle entry being the average pause')
            sys.exit()

    num_blocks = run_duration // (avg_pause + block_duration)

    found_correct_trial = False
    while not found_correct_trial:
        # make stimuli sequence
        stimuli_seq = generate_stimulus_sequence(
            num_blocks, numerosity, stimulus_types)
        # True when no problems, this means loop will not run again
        found_correct_trial = check_n_matches_string(stimuli_seq)

    # make pauses sequence
    generated_pauses = generate_pause_sequence(
        num_pauses=num_blocks, avg_pause=avg_pause, pauses=pauses)

    # add onsets for blocks
    onsets = make_onsets_df(run_duration, block_duration,
                            stimuli_seq, num_blocks, generated_pauses)
    # FIXME those functions have side effecs unto onsets. Make a class with these methods, inheriting from pandas df?
    chosen_design_onsets = add_gender_voice(add_ll_vis_conds(
        onsets, samples_per_block=samples_per_block), samples_per_block=samples_per_block, block_duration=block_duration)

    return chosen_design_onsets
