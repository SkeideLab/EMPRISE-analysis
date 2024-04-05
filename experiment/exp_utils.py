import gc
import itertools as it
import os

import numpy as np
import pandas as pd
import serial
from psychopy import prefs  # noqa
prefs.hardware['audioLib'] = ['PTB', 'sounddevice', 'pyo', 'pygame']  # noqa
from psychopy import (__version__, clock, constants, core, data, event, gui,
                      logging, monitors, sound, visual)
from psychopy.constants import FINISHED, NOT_STARTED, STARTED
from psychopy.hardware import keyboard
from psychopy.hardware.emulator import launchScan

import predefine_run_blocks


def run_exp(location, exp_name='EMPRISE', train=False):

    expInfo = initialize_exp(exp_name, location, train)

    # expInfo for info that is stable over run iterations
    # exp_params for info that changes and belongs to that specific run (except window)

    expInfo['fix_cross'] = FixationCross(
        expInfo['win'], switch_on=False, exp_name=exp_name, radius=expInfo['radius'])

    # TODO only use one keyboard???
    for run in range(int(expInfo['starting_run']), expInfo['number_runs']+1):

        exp_params = execute_run(expInfo, exp_name, location, run, train)

    exp_params['win'].flip()

    finished_time = exp_params['globalClock'].getTime()
    while exp_params['globalClock'].getTime() < finished_time + 5:
        keys = event.getKeys(keyList=['escape'])
        # print(keys)
        if keys:
            print('goodbye')
            break

    exp_params['win'].close()
    safe_quit(exp_params)


def execute_run(expInfo, exp_name, location, run, train):
    exp_params = {}
    if exp_name == 'priming':
        exp_params['modality'] = expInfo['runs_modality'][run - 1]

    if train:
        exp_params['train'] = True
        exp_params['continue_key'] = 'space'
    exp_params['date'] = data.getDateStr(format='%Y_%m_%d_%H%M')
    exp_params['win'] = expInfo['win']
    exp_params['audio'] = expInfo['audio']
    exp_params['routine_timer'] = expInfo['routine_timer']
    exp_params['globalClock'] = core.Clock()

    # create a default keyboard
    exp_params['defaultKeyboard'] = keyboard.Keyboard()
    ############################################################################
    vol = launchScan(exp_params['win'], expInfo['MR_settings'],
                     globalClock=exp_params['globalClock'], mode='Scan')
    ############################################################################

    # Have a short delay after the scanner has been started
    exp_params['routine_timer'].reset(0)
    exp_params['routine_timer'].add(expInfo['offset'])  # seconds

    start_time = clock.getTime()

    expInfo['background'].setAutoDraw(True)
    switch_fixcross(expInfo['fix_cross'], True)

    exp_params['win'].flip()

    # ----------logging-------------------------
    if location == 'mockup':
        prepare_logging_mockup(exp_params, expInfo)
    elif 'scanner' in location:
        exp = prepare_logging_scanner(exp_params, expInfo, train, run)

    onsets = create_seq_per_exp(exp_params, exp_name)

    trials = data.TrialHandler(trialList=onsets.to_dict(
        'records'), nReps=1, method='sequential')
    # setting params for this exp
    # exp_params['vf_params'] = calc_VF(monitor_data=Monitor_data)
    exp.addLoop(trials)
    exp_params['stimuli'] = prepare_stim(
        trials.trialList, -1,  exp_params, expInfo)

    exp_params['feedback'] = False

    exp_params['feedback_timer'] = core.CountdownTimer()
    exp_params['stim_clock'] = core.Clock()
    exp_params['stim_keyboard'] = keyboard.Keyboard(
        clock=exp_params['stim_clock'])
    # boolean to indicate if response in block
    exp_params['responded'] = False

    wait_routinetimer_end(exp_params)

    print(f"time on routinetimer: {exp_params['routine_timer'].getTime()}")
    # exp_params['routine_timer'].reset()

    for trial_index, trial in enumerate(trials):
        present(trial_list=trials.trialList,
                trial_index=trial_index, experimenthandler=exp, exp_params=exp_params, expInfo=expInfo)

    log_run_end(start_time, exp)
    show_ask_continue(expInfo['fix_cross'], run, exp_params, expInfo)
    return exp_params


def log_run_end(start_time, exp):
    # after a full run the runtime is saved
    end_time = clock.getTime()
    run_time = end_time - start_time
    print('full run time: '+str(run_time))
    m, s = divmod(run_time, 60)
    mins = int(m)
    secs = int(s)
    logging.exp('Full run completed. Time:' +
                str(mins) + " mins and " + str(secs) + " seconds")

    # exp.saveAsWideText(filename+'.csv')
    exp.close()


def wait_routinetimer_end(exp_params):
    time_remaining = exp_params['routine_timer'].getTime()
    if time_remaining < 0:
        logging.warn('Overshot the preparation period by ' +
                     str(-time_remaining))
    else:
        while exp_params['routine_timer'].getTime() > 1/60:
            exp_params['win'].flip()


def create_seq_per_exp(exp_params, exp_name):

    if exp_name == 'harveyvisual':
        exp_params['block_duration'] = 4.2
        onsets = create_sequence_harvey(
            block_duration=exp_params['block_duration'],
            samples_per_block=exp_params['samples_per_block'])
    elif exp_name == 'harveydigits':
        exp_params['block_duration'] = 4.2
        onsets = create_sequence_harvey(
            block_duration=exp_params['block_duration'],
            samples_per_block=exp_params['samples_per_block'],
            stim_type="digit")
    elif exp_name == 'harveywritten':
        exp_params['block_duration'] = 4.2
        onsets = create_sequence_harvey(
            block_duration=exp_params['block_duration'],
            samples_per_block=exp_params['samples_per_block'],
            stim_type="written")
    elif exp_name == 'harveyspoken':
        exp_params['block_duration'] = 4.2
        onsets = create_sequence_harveyaudio(
            block_duration=exp_params['block_duration'],
            samples_per_block=exp_params['samples_per_block'],
            stim_type="spoken")
    elif exp_name == 'harveyaudio':
        exp_params['block_duration'] = 4.2
        onsets = create_sequence_harveyaudio(
            block_duration=exp_params['block_duration'],
            samples_per_block=exp_params['samples_per_block'])

    elif exp_name == 'priming':
        exp_params['trial_clock'] = core.Clock()
        exp_params['mask1'] = visual.TextStim(
            exp_params['win'], text='XXX', color='black', autoLog=True, name='mask1')
        exp_params['mask2'] = visual.TextStim(
            exp_params['win'], text='XXX', color='black', autoLog=True, name='mask2')
        onsets = create_sequence_priming(exp_params['modality'])
    elif 'harveyspokenvisual' in exp_name:
        exp_params['block_duration'] = 4.2
        # condition is incongruent or congruent, latter part of exp_name
        onsets = create_sequence_harveymixed(
            block_duration=exp_params['block_duration'],
            samples_per_block=exp_params['samples_per_block'],
            stim_type='spoken',
            condition=exp_name.replace('harveyspokenvisual_', ''))
    else:
        exp_params['block_duration'] = 6
        onsets = create_sequence(
            block_duration=exp_params['block_duration'], samples_per_block=exp_params['samples_per_block'])

    return onsets


def prepare_logging_scanner(exp_params, expInfo, train, run):
    exp_params['match_key'] = expInfo['left button (match/odd)']
    exp_params['nonmatch_key'] = expInfo['right button (non-match/even)']
    exp_params['samples_per_block'] = expInfo['samples_per_block']

    if train:
        exp = data.ExperimentHandler(
            name='emprise', savePickle=False)
    else:
        run_id = u'sub-%s_ses-%s_run-%s_%s_%s' % (
            expInfo['participant'], expInfo['session'], run, expInfo['expName'], exp_params['date'])
        print(f"Now running: {run_id}")
        filename = os.path.join(expInfo['out_path'], run_id)
        logFile = logging.LogFile(filename+'.log', level=logging.EXP)
        logging.console.setLevel(logging.WARNING)
        logging.setDefaultClock(exp_params['globalClock'])
        logging.exp(expInfo)
        logging.exp(expInfo['MR_settings'])

        exp = data.ExperimentHandler(
            name='emprise', dataFileName=filename, extraInfo={key: expInfo[key] for key in expInfo.keys() if key != '20dots'}, savePickle=False)

    return exp


def prepare_logging_mockup(exp_params, expInfo):
    # serial response buttons
    exp_params['ser'] = expInfo['ser']
    exp = data.ExperimentHandler(
        name='emprise', savePickle=False)
    exp_params['samples_per_block'] = 6  # samples_per_block


def create_sequence_priming(modality):
    onsets = predefine_run_blocks.create_priming_sequence(modality)
    # make pandas dataframe with columns mod_prime, num_prime,mod_target_mod_target,duration
    onsets[['prime', 'target']] = onsets['trial_type'].str.split(
        '_', expand=True)
    # events.rename(columns={0: 'prime', 1: 'target'}, inplace=True)
    onsets[['mod_prime', 'num_prime']
           ] = onsets['prime'].str.split('-', expand=True)
    onsets[['mod_target', 'num_target']
           ] = onsets['target'].str.split('-', expand=True)

    onsets = onsets.drop(['prime', 'target', 'trial_type'], axis=1)
    return onsets


def make_out_path(expInfo, location):
    # ---------------file organizing-------------------
    # Ensure that relative paths start from the same directory as this script
    expInfo['_thisDir'] = os.path.dirname(os.path.abspath(__file__))
    os.chdir(expInfo['_thisDir'])
    expInfo['out_path'] = os.path.join(expInfo['_thisDir'], 'logfiles')

    if 'scanner' in location:
        for identifier in ['participant', 'session']:
            if not expInfo[identifier]:
                expInfo[identifier] = '_'

        expInfo['out_path'] = os.path.join(
            expInfo['out_path'], expInfo['session'], f"{expInfo['participant']}_d-{data.getDateStr(format='%Y%m%d')}")

    if not os.path.exists(expInfo['out_path']):
        os.makedirs(expInfo['out_path'], exist_ok=True)


def prep_window(expInfo, location):
    # --------------------monitor prep---------------------------
    if location == '7tscanner':
        Monitor_data = [1024, 768, 34, 130]
    else:
        # location is skyra, coded as 'scanner'
        Monitor_data = [1024, 768, 40, 118]

    # this saves the monitor config to a psychopy settings file so we can use it later
    mon = monitors.Monitor(
        'stimulus_screen', width=Monitor_data[2], distance=Monitor_data[3])
    mon.setSizePix((Monitor_data[0], Monitor_data[1]))
    mon.save()

    # -------------------window------------------
    expInfo['win'] = visual.Window(
        size=(Monitor_data[0], Monitor_data[1]), fullscr=True, screen=0,
        winType='pyglet', allowGUI=False,
        monitor='stimulus_screen', color='black', colorSpace='rgb',
        blendMode='avg', useFBO=True,
        units='deg', name='stim_window')


def initialize_exp(exp_name, location, train):

    if 'scanner' in location:
        expInfo = initialize_scanner(train, location, exp_name)
    elif location == 'mockup':
        expInfo = initialize_mockup()

    make_out_path(expInfo, location)

    expInfo['psychopy version'] = __version__
    expInfo['expName'] = exp_name

    prep_window(expInfo, location)

    sync = 't'
    if location == '7tscanner':
        sync = '5'
    # --------------------mri settings-----------------------------
    expInfo['MR_settings'] = {
        'TR': 2.000,     # duration (sec) per whole-brain volume
        # number of whole-brain 3D volumes per scanning run (for simulation mode!)
        'volumes': 5,
        # character to use as the sync timing event; assumed to come at start of a volume
        'sync': sync,
        # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
        'skip': 0,
        'sound': False    # in test mode: play a tone as a reminder of scanner noise
    }

    expInfo['routine_timer'] = core.CountdownTimer()

    # ----------------stimulus and exp prep----------------------------
    if location == '7tscanner':
        expInfo['background'] = visual.Rect(
            expInfo['win'], size=[2, 2], units='norm', fillColor='grey')
        expInfo['offset'] = 2.1
        expInfo['audio'] = []
        expInfo['number_runs'] = 8
        expInfo['radius'] = 0.75
        if 'visual' in exp_name:
            expInfo['20dots'] = load_20dots()
        if exp_name == 'harveyaudio':
            expInfo['audio'] = prepare_audio_stim(['sine_440khz', 'sine_333khz',
                                                   'sine_359khz', 'sine_392khz', 'sine_500khz', 'sine_1000khz'])
        if 'spoken' in exp_name:
            expInfo['audio'] = prepare_audio_stim(['female_1', 'female_2', 'female_3',
                                                   'male_1', 'male_2', 'male_3'])
        if exp_name == 'priming':
            expInfo['number_runs'] = 9
            expInfo['radius'] = 3
            # go from this row to the last row, for each run there is one row
            expInfo['runs_modality'] = load_run_order(expInfo)
            print('Modality of the run with the run order',
                  expInfo['runs_modality'])

    else:
        expInfo['audio'] = prepare_audio_stim(['male', 'female'])
        expInfo['number_runs'] = 6
        expInfo['radius'] = 3
        expInfo['background'] = visual.Circle(expInfo['win'], radius=expInfo['radius'], edges=32, lineWidth=0, lineColor=None,
                                              lineColorSpace='rgb', fillColor='grey', fillColorSpace='rgb', name='feedback_circle')
        expInfo['vf_feedback'] = expInfo['background']
        expInfo['offset'] = 5

    return expInfo


def load_20dots():
    directory = os.path.dirname(os.path.abspath(__file__))
    dots_path = os.path.join(
        directory, 'input_positions', '20dots.csv')
    return pd.read_csv(
        dots_path, header=None)


def load_run_order(expInfo):
    df = pd.read_csv(os.path.join(
        expInfo['_thisDir'], 'priming', 'run_order.csv'))
    # tuple format not so suitable
    try:
        order = int(expInfo['run_order']) - 1
    except ValueError as err:
        order = 0
    mod_order = list(df.loc[:, f'col_{order}'])
    tpl = []
    for element in mod_order:
        a, b = eval(element)
        tpl.append((a, b))
    del mod_order
    return tpl


def initialize_scanner(train, location, exp_name):

    # Store info about the experiment session
    if location == 'scanner':
        expInfo = {'participant': '', 'session': '', 'starting_run': '',
                   'left button (match/odd)': [], 'right button (non-match/even)': ['b'], 'samples_per_block': 6
                   }
    elif location == '7tscanner':
        if exp_name in ['harveyvisual', 'harveyaudio', 'harveydigits', 'harveywritten', 'harveyspoken', 'harveyspokenvisual_congruent', 'harveyspokenvisual_incongruent']:
            expInfo = {'participant': '', 'session': '', 'starting_run': '',
                       'left button (match/odd)': [], 'right button (non-match/even)': ['1', '2', '3', '4'], 'samples_per_block': 1
                       }
        elif exp_name == 'priming':
            expInfo = {'participant': '', 'session': '', 'starting_run': '', 'run_order': '',
                       'left button (match/odd)': ['1', '2'], 'right button (non-match/even)': ['3', '4'], 'samples_per_block': 1
                       }
    else:
        print(f'location unknown! {location}')
    fixed = fixed = [
        'left button (match/odd)', 'right button (non-match/even)', 'samples_per_block']

    if not train:
        dlg = gui.DlgFromDict(dictionary=expInfo,
                              sortKeys=False, title=exp_name, fixed=fixed)
        if dlg.OK == False:
            core.quit()

    if not expInfo['starting_run']:
        expInfo['starting_run'] = 1

    return expInfo


def initialize_mockup():

    # Store info about the experiment session
    expName = 'EMPRISE'
    expInfo = {'port number': '', 'starting_run': 1}

    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()

    expInfo['ser'] = serial.Serial(
        f"/dev/ttyUSB{expInfo['port number']}", timeout=0)

    return expInfo


def show_ask_continue(fix_cross, run, exp_params, expInfo):

    expInfo['background'].autoDraw = False
    switch_fixcross(fix_cross, False)
    if run != expInfo['number_runs']:
        ask = visual.TextStim(
            exp_params['win'], text=f'Bereit für den nächsten Durchlauf? \nEs folgt Durchlauf {run+1}. \n[Leertaste]', name='ask_next_run', color='white')
        ask.autoDraw = True
        exp_params['win'].flip()
        wait = True
        while wait:
            if exp_params['defaultKeyboard'].getKeys(
                    keyList=['escape']):
                safe_quit(exp_params)
            if exp_params['defaultKeyboard'].getKeys(
                    keyList=['space']):
                ask.autoDraw = False
                exp_params['win'].flip()
                wait = False


def switch_fixcross(fix_cross, switch):
    for line in fix_cross:
        line.autoDraw = switch


# button 1 ist match key!
def check_buttons(exp_params, both_keys=False):
    # if we are in the mockup, use serial connection
    if 'ser' in exp_params:
        resp = list(exp_params['ser'].read())
        if len(resp) > 0:
            exp_params['ser'].reset_input_buffer()
            # if resp[0] == 252:
            #     return {'key': 'match_key', 'rt': np.nan}
            if resp[0] == 248:
                return {'key': 'nonmatch_key', 'rt': np.nan}
            else:
                print(f'unrecognizable serial input: {resp[0]}')
    else:
        # # always use list for keys now

        if isinstance(exp_params['nonmatch_key'], str):
            exp_params['nonmatch_key'] = [exp_params['nonmatch_key']]
        if both_keys:
            if isinstance(exp_params['match_key'], str):
                exp_params['match_key'] = [exp_params['match_key']]

        # # we are in the scanner, use keyboard input
        if both_keys:
            resp = exp_params['stim_keyboard'].getKeys(
                keyList=exp_params['match_key'] + exp_params['nonmatch_key'])
        else:
            resp = exp_params['stim_keyboard'].getKeys(
                keyList=exp_params['nonmatch_key'])
        if not resp in ['', None, []]:
            resp = resp[0]
            if both_keys:
                if resp.name in exp_params['match_key']:
                    return {'key': 'match_key', 'rt': resp.rt}
            if resp.name in exp_params['nonmatch_key']:
                return {'key': 'nonmatch_key', 'rt': resp.rt}

    # check if we should send continue key (during training)
    if 'continue_key' in exp_params:
        resp = exp_params['defaultKeyboard'].getKeys(
            keyList=[exp_params['continue_key']])
        if not resp in ['', None, []]:
            return {'key': 'continue_key', 'rt': np.nan}

    # if we were not able to return a match or nonmatch key, return none (no answer)
    return None


def safe_quit(exp_params):
    if 'ser' in exp_params:
        exp_params['ser'].close()
    core.quit()


def init_component(thisComponent):
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED


def present_priming(trial_list, trial_index, experimenthandler, exp_params, expInfo):
    """strategy: present previously prepared fixationcross, mask, prime, mask, target here.
    then wait for response and prepare new stimuli in present()

    Parameters
    ----------
    trial_list : _type_
        _description_
    trial_index : _type_
        _description_
    experimenthandler : _type_
        _description_
    exp_params : _type_
        _description_
    expInfo : _type_
        _description_
    """
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    frameTolerance = 0.001  # how close to onset before 'same' frame
    exp_params['routine_timer'].add(trial_list[trial_index]['duration'])
    # # update component parameters for each repeat
    # key_resp.keys=[]
    # key_resp.rt=[]
    # _key_resp_allKeys=[]
    # keep track of which components have finished
    mask1 = exp_params['mask1']
    mask2 = exp_params['mask2']
    prime = exp_params['stimuli']['prime']
    target = exp_params['stimuli']['target']

    trialComponents = [mask1,
                       prime, mask2, target]
    for thisComponent in trialComponents:
        if isinstance(thisComponent, list):
            for comp in thisComponent:
                init_component(comp)
        else:
            init_component(thisComponent)

    # reset timers
    t = 0
    _timeToFirstFrame = expInfo['win'].getFutureFlipTime(clock="now")
    print(_timeToFirstFrame)
    # t0 is time of first possible flip
    exp_params['trial_clock'].reset(-_timeToFirstFrame)
    frameN = -1
    # switch on fixation cross for first phase
    switch_fixcross(expInfo['fix_cross'], True)
    # -------Run Routine "trial"-------
    while continueRoutine and exp_params['routine_timer'].getTime() > 0:
        # get current time
        t = exp_params['trial_clock'].getTime()
        tThisFlip = expInfo['win'].getFutureFlipTime(
            clock=exp_params['trial_clock'])
        tThisFlipGlobal = expInfo['win'].getFutureFlipTime(clock=None)
        # number of completed frames (so 0 is the first frame)
        frameN = frameN + 1
        # update/draw components on each frame

        # *mask1* updates
        if mask1.status == NOT_STARTED and tThisFlip >= 0.25-frameTolerance:
            switch_fixcross(expInfo['fix_cross'], False)
            # keep track of start time/frame for later
            mask1.frameNStart = frameN  # exact frame index
            mask1.tStart = t  # local t and not account for scr refresh
            mask1.tStartRefresh = tThisFlipGlobal  # on global time
            # time at next scr refresh
            expInfo['win'].timeOnFlip(mask1, 'tStartRefresh')
            mask1.setAutoDraw(True)
        if mask1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mask1.tStartRefresh + 0.25-frameTolerance:
                # keep track of stop time/frame for later
                mask1.tStop = t  # not accounting for scr refresh
                mask1.frameNStop = frameN  # exact frame index
                # time at next scr refresh
                expInfo['win'].timeOnFlip(mask1, 'tStopRefresh')
                mask1.setAutoDraw(False)

        # *num_word* updates
        for prime_i in prime:
            if prime_i.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                prime_i.frameNStart = frameN  # exact frame index
                prime_i.tStart = t  # local t and not account for scr refresh
                prime_i.tStartRefresh = tThisFlipGlobal  # on global time
                # time at next scr refresh
                expInfo['win'].timeOnFlip(prime_i, 'tStartRefresh')
                prime_i.setAutoDraw(True)
            if prime_i.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prime_i.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    prime_i.tStop = t  # not accounting for scr refresh
                    prime_i.frameNStop = frameN  # exact frame index
                    # time at next scr refresh
                    expInfo['win'].timeOnFlip(prime_i, 'tStopRefresh')
                    prime_i.setAutoDraw(False)

        # *mask2* updates
        if mask2.status == NOT_STARTED and tThisFlip >= 0.55-frameTolerance:
            # keep track of start time/frame for later
            mask2.frameNStart = frameN  # exact frame index
            mask2.tStart = t  # local t and not account for scr refresh
            mask2.tStartRefresh = tThisFlipGlobal  # on global time
            # time at next scr refresh
            expInfo['win'].timeOnFlip(mask2, 'tStartRefresh')
            mask2.setAutoDraw(True)
        if mask2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mask2.tStartRefresh + 0.25-frameTolerance:
                # keep track of stop time/frame for later
                mask2.tStop = t  # not accounting for scr refresh
                mask2.frameNStop = frameN  # exact frame index
                # time at next scr refresh
                expInfo['win'].timeOnFlip(mask2, 'tStopRefresh')
                mask2.setAutoDraw(False)

        # *num_dots* updates
        for target_i in target:
            if target_i.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                target_i.frameNStart = frameN  # exact frame index
                target_i.tStart = t  # local t and not account for scr refresh
                target_i.tStartRefresh = tThisFlipGlobal  # on global time
                # time at next scr refresh
                expInfo['win'].timeOnFlip(target_i, 'tStartRefresh')
                target_i.setAutoDraw(True)
            if target_i.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_i.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    target_i.tStop = t  # not accounting for scr refresh
                    target_i.frameNStop = frameN  # exact frame index
                    # time at next scr refresh
                    expInfo['win'].timeOnFlip(target_i, 'tStopRefresh')
                    target_i.setAutoDraw(False)

        # check for quit (typically the Esc key)
        if exp_params['defaultKeyboard'].getKeys(keyList=["escape"]):
            core.quit()

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if isinstance(thisComponent, list):
                for comp in thisComponent:
                    if hasattr(comp, "status") and comp.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
            else:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            expInfo['win'].flip()

    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if isinstance(thisComponent, list):
            for comp in thisComponent:
                if hasattr(comp, "setAutoDraw"):
                    comp.setAutoDraw(False)
        else:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
    for target_i in target:
        expInfo['win'].timeOnFlip(target_i, 'tStopRefresh')
    experimenthandler.addData('mask1.started', mask1.tStartRefresh)
    experimenthandler.addData('mask1.stopped', mask1.tStopRefresh)
    experimenthandler.addData(
        'prime.started', [prime_i.tStartRefresh for prime_i in prime])
    experimenthandler.addData(
        'prime.stopped', [prime_i.tStopRefresh for prime_i in prime])
    experimenthandler.addData('mask2.started', mask2.tStartRefresh)
    experimenthandler.addData('mask2.stopped', mask2.tStopRefresh)
    # need this here so we can save target stop time
    exp_params['win'].flip()
    experimenthandler.addData(
        'target.started', [target_i.tStartRefresh for target_i in target])
    experimenthandler.addData(
        'target.stopped', [target_i.tStopRefresh for target_i in target])

    for item, name in zip([mask1, prime[0], mask2, target[0]], ['mask1', 'prime', 'mask2', 'target']):
        print(f"{name} started: {item.tStartRefresh}")
        print(f"{name} stopped: {item.tStopRefresh}")
        try:
            print(f"{name} duration: {item.tStopRefresh-item.tStartRefresh}")
        except:
            pass
    experimenthandler.nextEntry()


def is_pause(trial):
    return ('num' in trial and trial['num'] == 0) or ('mod_prime' in trial and trial['mod_prime'] == 'pause')


def execute_pause(expInfo, exp_params, trial, trial_list, trial_index, exp):
    if expInfo['expName'] == 'priming':
        check_buttons(exp_params, both_keys=True)
    exp_params['routine_timer'].add(trial['duration'])
    exp_params['stimuli'] = prepare_stim(
        trial_list, trial_index,  exp_params, expInfo)
    print('new stimuli created')
    print(exp_params['globalClock'].getTime())
    logging.flush()
    # Returns the number of
    # objects it has collected
    # and deallocated
    collected = gc.collect()

    # Prints Garbage collector
    # as 0 object
    print("Garbage collector: collected",
          "%d objects." % collected)
    # Wait for ISI to complete
    time_remaining = exp_params['routine_timer'].getTime()
    if time_remaining < 0:
        logging.warn('Overshot the expected period by ' +
                     str(-time_remaining))
    else:
        while exp_params['routine_timer'].getTime() > 0.005:
            check_stop_feedback(exp_params)
            if expInfo['expName'] == 'priming':
                give_feedback(trial_list, trial_index, exp, exp_params)


def present(trial_list, trial_index, experimenthandler, exp_params, expInfo):
    """Present the stimuli.
        If this block is a pause, wait the duration and prepare the stimuli
        for the next block, calling prepare_stim().
        If this is a stimuli block, go through every timepoint in exp_params['stimuli'],
        add 0.4s to the timer, present the stimuli using start_stim(),
        check for feedback using give_feedback(). Turn off stimuli presentation
        after and log data for this trial with log_data()

    Args:
        trial_list (list): list of dicts. one dict per block
        trial_index (int): index of current block
        experimenthandler (psychopy.data.experimenthandler): saves experiment parameters
    """
    exp = experimenthandler
    trial = trial_list[trial_index]

    # if pause: wait and prepare next stimuli
    if is_pause(trial):
        execute_pause(expInfo, exp_params, trial, trial_list, trial_index, exp)
        exp.nextEntry()
        return

    # this block has not been responded to yet
    exp_params['responded'] = False

    if expInfo['expName'] == 'priming':
        present_priming(trial_list, trial_index,
                        experimenthandler, exp_params, expInfo)
    else:
        # go through samples during one Block
        audio_started = 0
        for sample_i, stims in enumerate(exp_params['stimuli']):
            present_sample(stims, exp_params, trial, sample_i,
                           trial_list, trial_index, exp)
        exp_params['win'].flip()


def get_sample_dur(trial, exp_params):
    return trial['duration']/exp_params['samples_per_block']


def present_sample(stims, exp_params, trial, sample_i, trial_list, trial_index, exp):
    print(stims)
    # save display onsets and RT
    data = {}
    exp_params['routine_timer'].add(get_sample_dur(trial, exp_params))
    start_stim(stims, data, trial, sample_i, exp_params)
    # wait for end of this sample, give feedback
    while exp_params['routine_timer'].getTime() > 0:
        # check status of audio stimulus: have we started playing and now the status is finished? If yes take the time as stop time
        if 'audio' in trial['mod']:
            if stims['stims'].status == constants.STARTED:
                audio_started = True
            elif stims['stims'].status == constants.FINISHED and audio_started:
                data['t_stop'] = exp_params['globalClock'].getTime()
                audio_started = False

        # check for quit (typically the Esc key)
        if exp_params['defaultKeyboard'].getKeys(keyList=["escape"]):
            safe_quit(exp_params)
        give_feedback(trial_list, trial_index, exp, exp_params)

        check_stop_feedback(exp_params)

    wait_for_response_training(
        exp_params, sample_i, trial_index, trial_list, exp)

    force_feedback_if_needed(exp_params, sample_i,
                             trial_list, trial_index, exp)

    # STOP STIMULATION
    if 'audio' in trial['mod']:
        if stims['stims'].status != constants.FINISHED:
            stims['stims'].stop()
            exp_params['win'].timeOnFlip(data, 't_stop')
    else:
        for s in stims['stims']:
            try:
                s.setAutoDraw(False)
            except AttributeError as err:
                # 'mixed' condition
                # don't stop because audio may be longer than dot presentation
                pass
        exp_params['win'].timeOnFlip(data, 't_stop')

    exp_params['win'].flip()
    log_data(data, exp, stims)


def wait_for_response_training(exp_params, sample_i, trial_index, trial_list, exp):
    # TRAINING: wait for response
    # don't go on if it is the last sample of the block and we have an earlier number to compare to
    if ('train' in exp_params) and (sample_i + 1 == exp_params['samples_per_block']) and (trial_index > 0):
        repeat = True
        # wait for response
        while repeat:
            give_feedback(trial_list, trial_index, exp, exp_params)
            # if we have a response, we can go on
            repeat = not exp_params['responded']
            # check for quit (typically the Esc key)
            if exp_params['defaultKeyboard'].getKeys(keyList=["escape"]):
                safe_quit(exp_params)
        # reset routine timer so that pause will have normal duration, no skipping forward
        exp_params['routine_timer'].reset()


def force_feedback_if_needed(exp_params, sample_i, trial_list, trial_index, exp):
    # RESPONSE
    # in case of no response being for the whole block, give feedback now
    if not exp_params['responded'] and (sample_i + 1 == exp_params['samples_per_block']):
        # force show feedback
        give_feedback(trial_list, trial_index, exp,
                      exp_params, force_feedback=True)


def check_stop_feedback(exp_params):
    # if screen is currently showing feedback
    if exp_params['feedback'] and exp_params['feedback_timer'].getTime() <= 0:
        exp_params['feedback'] = False
        exp_params['vf_feedback'].fillColor = 'grey'
        exp_params['win'].flip()


def log_data(data, exp, stims):
    """Log data for this trial.
        Timing data for this trial is in data
        and stimuli info in stims.

    Args:
        data (dict): saves timing data for this sample
        exp (psychopy.data.experimenthandler): handles experiment parameters
        stims (dict): has stimuli conf data that needs to be saved
    """
    [exp.addData(key, data[key]) for key in data]
    stims.pop('stims', None)
    [exp.addData(key, stims[key]) for key in stims]
    exp.nextEntry()


def start_stim(stims, data, trial, sample_i, exp_params):
    """Start drawing stimuli and record the time"""
    mod = trial['mod']
    if 'dot' in mod:
        [s.setAutoDraw(True) for s in stims['stims']]
    elif 'digit' in mod or 'written' in mod:
        [setattr(s, 'pos', t) for s, t in zip(stims['stims'], stims['pos'])]
        [setattr(s, 'font', t) for s, t in zip(stims['stims'], stims['fonts'])]
        [setattr(s, 'height', t)
         for s, t in zip(stims['stims'], stims['size'])]
        [s.setAutoDraw(True) for s in stims['stims']]

    elif 'audio' in mod or 'spoken' in mod:
        stims['stims'].play()
    elif 'mixed' in mod:
        # present two different modalities simultaneously
        for i, stim in enumerate(stims['stims']):
            if 'dot' in stims['mod'][i]:
                stim.setAutoDraw(True)
            elif 'digit' in stims['mod'][i] or 'written' in stims['mod'][i]:
                stim.pos = stims['pos'][i]
                stim.font = stims['fonts'][i]
                stim.height = stims['size'][i]
                stim.setAutoDraw(True)
            elif 'audio' in stims['mod'][i] or 'spoken' in stims['mod'][i]:
                stim.play()
            else:
                print('ERROR: problem with starting stimuli')
    exp_params['win'].timeOnFlip(data, 't_start')

    # clear buffer of button presses that happen during pause
    if sample_i == 0:
        # delete keys pressed during pause if first sample
        exp_params['win'].callOnFlip(check_buttons, exp_params=exp_params)
        exp_params['win'].callOnFlip(exp_params['stim_clock'].reset)
    exp_params['win'].flip()


def start_fb(exp_params, correct):
    """change screen color depending on correctness of answer

    Args:
        exp (psychopy.data.experimenthandler): saves experiment parameters
        correct (boolean): was the answer given correct or not
    """
    if 'vf_feedback' in exp_params:
        fb_color = 'red'
        if correct:
            fb_color = 'green'
        exp_params['feedback_timer'].reset()
        exp_params['feedback_timer'].add(0.5)
        exp_params['feedback'] = True
        exp_params['vf_feedback'].fillColor = fb_color
        exp_params['win'].flip()


def give_feedback(trial_list, trial_index, exp, exp_params, force_feedback=False):
    """check for button presses,
        check if match trial,
        determine if answer is correct,
        call start_fb to give fb,
        log.

    Args:
        trial_list (list): list of dicts. Each dict describes one block
        trial_index (int): index of current trial
        exp (psychopy.data.experimenthandler): saves parameter for experiment
        force_feedback(boolean): should feedback be given even if no button press
    """

    if exp.extraInfo['expName'] == 'priming':
        if not exp_params['responded']:
            ans = check_buttons(exp_params, both_keys=True)
            if not ans:
                return
            if ans['key'] == 'match_key':
                exp.addData('button', 'left')
                exp.addData('answer', 'odd')
                exp_params['responded'] = True
                exp.addData('RT', ans['rt'])
            elif ans['key'] == 'nonmatch_key':
                exp.addData('button', 'right')
                exp.addData('answer', 'even')
                exp_params['responded'] = True
                exp.addData('RT', ans['rt'])
        return

    if trial_index == 0:
        return
    match = False

    if trial_list[trial_index]['num'] == trial_list[trial_index-2]['num']:
        match = True
        exp.addData('match', 'true')
    else:
        exp.addData('match', 'false')

    # FEEDBACK WITHOUT RESPONSE
    # no reply was given, implies match
    if force_feedback:
        if match:
            # turn green
            start_fb(exp_params, True)
            exp.addData('answer_truth', 'correct')
            exp.addData('RT', np.nan)
        else:
            # turn red
            start_fb(exp_params, False)
            exp.addData('answer_truth', 'wrong')
            exp.addData('RT', np.nan)
        return

    if not exp_params['responded']:
        ans = check_buttons(exp_params)
        if not ans:
            return

        if ans['key'] == 'match_key':
            exp.addData('button', 'left')
            exp.addData('answer', 'match')
            exp_params['responded'] = True
        elif ans['key'] == 'nonmatch_key':
            exp.addData('button', 'right')
            exp.addData('answer', 'nonmatch')
            exp_params['responded'] = True
        elif ans['key'] == 'continue_key':
            exp_params['responded'] = True

        if ((ans['key'] == 'match_key') and match) \
                or ((ans['key'] == 'nonmatch_key') and not match)\
            or (ans['key'] == 'continue_key' and match):
            # turn green
            start_fb(exp_params, True)
            exp.addData('answer_truth', 'correct')
            exp.addData('RT', ans['rt'])
            # print(globalClock.getTime()-exp.thisEntry['t_start'])
            # print(exp.thisEntry['RT'])
        elif ((ans['key'] == 'match_key') and not match) \
                or ((ans['key'] == 'nonmatch_key') and match)\
            or (ans['key'] == 'continue_key' and not match):
            # turn red
            start_fb(exp_params, False)
            exp.addData('answer_truth', 'wrong')
            exp.addData('RT', ans['rt'])
            # print(globalClock.getTime()-exp.thisEntry['t_start'])
            # print(exp.thisEntry['RT'])


def prepare_stim_priming(trial_list, trial_index, exp_params, expInfo):

    if not 'mask' in exp_params:
        # init mask image
        pass

    trial = trial_list[trial_index+1]
    stims = {}
    for stim in ['prime', 'target']:
        trial2 = {'num': int(trial[f'num_{stim}']),
                  'mod': trial[f'mod_{stim}']}
        if 'dot' in trial2['mod']:

            trial2['ll_cond'] = ['standard']
            trial2['shape'] = ['dot']
            stimuli = arrange_dots_cond(
                trial2, expInfo, expInfo['radius'], exp_name=expInfo['expName'])
            stimuli = make_shapes(exp_params, expInfo, trial2,
                                  timep=stimuli[0])
        elif 'digit' in trial2['mod']:
            stimuli = make_digit(exp_params, trial2)
            size = get_random_size()
            stimuli = make_digit(exp_params,
                                 trial2,
                                 size=size,
                                 pos=get_positions(
                                     expInfo, vf_radius=expInfo['radius'], sizes=[size])[0],
                                 font=get_random_font())
        elif 'word' in trial2['mod']:
            # make word stimulus
            size = 1
            num2word = {1: 'one', 2: 'two', 3: 'three',
                        4: 'four', 5: 'five', 6: 'six', 7: 'seven'}
            number = num2word[trial2['num']]
            text = visual.TextStim(
                win=exp_params['win'], name='text', text=f'{number}', color='black')
            stimuli = [text]
        stims[stim] = stimuli
    return stims


def prepare_stim(trial_list, trial_index, exp_params, expInfo):
    """Create stimuli for presentation.
        If dot condition: Calculate arrangement
        depending on low-level visual features with arrange_dots_cond().
        Initialize visual stimuli with appropriate attributes for this block.
        Return a list of dicts. One dict per sample timepoint.

    Args:
        trial_list (list): list of dicts. Each dict describes one block with keys including 'mod','num'.
        trial_index (int): index of the current block
        exp (experimenthandler)

    Returns:
        list: list of dicts. Each dict describes one sample within the block, keys including 'stims' for initialized psychopy.visual objects.
    """
    stimuli = []  # list of dicts
    # if end of stimuli pres
    if trial_index+1 == len(trial_list):
        return stimuli

    if expInfo['expName'] == 'priming':
        return prepare_stim_priming(trial_list, trial_index, exp_params, expInfo)

    # get next block specs
    trial = trial_list[trial_index+1]

    if 'dot' in trial['mod']:
        # get shapes, positions and sizes for this block
        stimuli = arrange_dots_cond(
            trial, expInfo, expInfo['radius'], exp_name=expInfo['expName'])

        for timep in stimuli:

            stims = make_shapes(exp_params, expInfo, trial,
                                timep)

            timep['stims'] = stims

    elif 'digit' in trial['mod']:
        stim = make_digit(exp_params, trial)
        for i in range(exp_params['samples_per_block']):
            size = get_random_size()
            stimuli.append({'stims': stim,
                            'pos': get_positions(expInfo, vf_radius=expInfo['radius'], sizes=[size]),
                            'fonts': [get_random_font()],
                            'size': [size]}
                           )
    elif 'written' in trial['mod']:
        stim = make_word(exp_params, trial)
        for i in range(exp_params['samples_per_block']):
            size = get_random_size()
            stimuli.append({'stims': stim,
                            # rk: extend?
                            'pos': get_positions(expInfo, vf_radius=expInfo['radius'], sizes=[size]),
                            'fonts': [get_random_font()],
                            'size': [size]}
                           )
    elif 'audio' in trial['mod']:
        if 'gender_voice' in trial:
            audio_cat_name = 'gender_voice'
        else:
            audio_cat_name = 'audio_cat'
        stim = [exp_params['audio'][sample_category][str(trial['num'])]
                for sample_category
                in trial[audio_cat_name]]
        stimuli.extend([{'stims': sample} for sample in stim])
    elif 'mixed' in trial['mod']:
        # prepare single audio stim
        audio_cat_name = 'audio_cat'
        audio_stim = [exp_params['audio'][sample_category][str(trial[audio_cat_name][1])]
                      for sample_category
                      in trial[audio_cat_name][0]]

        # prepare (multiple) dot stims
        # get shapes, positions and sizes for this block
        stimuli_dots = arrange_dots_cond(
            trial, expInfo, expInfo['radius'], exp_name=expInfo['expName'])
        # generally only 1 timepoint
        for timep in stimuli_dots:
            stims = make_shapes(exp_params, expInfo, trial,
                                timep)
            # add modality info for every single stim
            # needed in case of simultaneous presentation of multiple modalities
            stim_mods = ['dot']*len(stims)
            # add audio stim to list of dot stims
            stims.extend(audio_stim)
            stim_mods.extend(['spoken']*len(audio_stim))

            timep['stims'] = stims
            timep['mod'] = stim_mods

        stimuli.extend(stimuli_dots)
    else:
        print('no valid modality')
        print(trial['mod'])

    return stimuli


def make_digit(exp_params, trial, pos=0.0, font='', size=None):
    stim = [visual.TextStim(exp_params['win'],
            text=str(trial['num']),
            # rk: was always assigned as 'black' but we want to have an oddball
            color=trial['colors'],
            autoLog=True,
            name='digit: '+str(trial['num']),
            height=size, pos=pos, font=font)]

    return stim


def digit2word(digit):
    """Create word string from a digit string.
        Returns a string with a word.
    Args:
        digit (string): String, e.g., '2' or '20'
    Returns:
        string: written word, e.g. 'zwei' or 'zwanzig'
    """
    word_dict = {'1': 'eins',
                 '2': 'zwei',
                 '3': 'drei',
                 '4': 'vier',
                 '5': 'fünf',
                 '20': 'zwanzig'}
    return word_dict[digit]


# copied and adjusted from make_digit
def make_word(exp_params, trial, pos=0.0, font='', size=None):
    stim = [visual.TextStim(exp_params['win'],
            text=digit2word(str(trial['num'])),
            # rk: was 'black' but we want to have an oddball
                            color=trial['colors'],
                            autoLog=True,
                            name='written: '+str(trial['num']),
                            height=size, pos=pos, font=font)]

    return stim


def make_shapes(exp_params, expInfo, trial, timep):
    shape = timep['shapes']
    pos = timep['pos']
    size = timep['size']
    if shape == 'dot':
        # different treatment of harvey and regular emprise because we need the exact circle size as defined by radius
        if 'harvey' in expInfo['expName']:
            stims = [visual.Circle(exp_params['win'],
                                   radius=size[num],
                                   edges=32,
                                   lineWidth=0,
                                   autoLog=True,
                                   lineColor=None,
                                   lineColorSpace='rgb',
                                   fillColor=timep['color'],
                                   pos=pos[num],
                                   # size=(size[num], size[num]),
                                   fillColorSpace='rgb',
                                   name=str(num+1) + ' of '+str(trial['num'])+' dots this sample')
                     for num in range(trial['num'])]
        else:
            stims = [visual.Circle(exp_params['win'],
                                   edges=32,
                                   lineWidth=0,
                                   autoLog=True,
                                   lineColor=None,
                                   lineColorSpace='rgb',
                                   fillColor=timep['color'],
                                   pos=pos[num],
                                   size=(size[num], size[num]),
                                   fillColorSpace='rgb',
                                   name=str(num+1) + ' of '+str(trial['num'])+' dots this sample')
                     for num in range(trial['num'])]
    elif shape == 'triangle':
        stims = [visual.Polygon(exp_params['win'],
                                edges=3,
                                radius=0.5,
                                pos=pos[num],
                                size=(size[num], size[num]),
                                name=str(
            num+1) + ' of '+str(trial['num'])+' triangles this sample',
            fillColor='black',
            autoLog=True)
            for num in range(trial['num'])]
    elif shape == 'rectangle':
        stims = [visual.Rect(exp_params['win'],
                             pos=pos[num],
                             size=(size[num], size[num]),
                             fillColor='black',
                             name=str(
            num+1) + ' of '+str(trial['num']) +
            ' rectangles this sample',
            autoLog=True)
            for num in range(trial['num'])]
    else:
        print('unrecognized shape specified')
    return stims


def prepare_audio_stim(categories):
    sounds = {}

    directory = os.path.dirname(os.path.abspath(__file__))
    for category in categories:
        sounds[category] = {}
        audio_dir = os.path.join(directory, 'audio', category)
        for filename in os.listdir(audio_dir):
            filepath = os.path.join(audio_dir, filename)
            if filename.startswith('eins') or filename.startswith('1_'):
                sounds[category]['1'] = sound.Sound(filepath)
            elif filename.startswith('zwei') or filename.startswith('2_'):
                sounds[category]['2'] = sound.Sound(filepath)
            elif filename.startswith('drei') or filename.startswith('3_'):
                sounds[category]['3'] = sound.Sound(filepath)
            elif filename.startswith('vier') or filename.startswith('4_'):
                sounds[category]['4'] = sound.Sound(filepath)
            elif filename.startswith('fuenf') or filename.startswith('5_'):
                sounds[category]['5'] = sound.Sound(filepath)
            elif filename.startswith('zwanzig') or filename.startswith('20_'):
                sounds[category]['20'] = sound.Sound(filepath)

    return sounds


def arrange_dots_cond(trial, expInfo, vf_radius=3, exp_name='EMPRISE'):
    """Calculate shapes, sizes and positions for dot condition.
        Call exp_utils.get_positions() to calculate positions.

    Args:
        trial (dict): Dictionary with info about this block. Keys
                        include 'num'=numerosity, 'mod'=modality.
        exp(experimenthandler):

    Returns:
       list: list of dicts. Each dict describes the stimulus conf for one sample
                        with keys 'shapes', 'size' and 'pos'.
    """

    # shapes = ['dots', 'triangles', 'rectangles']

    # conditions will now be set in the var trial
    # conditions = ['standard', 'const_circ', 'linear', 'density']
    stimuli = []
    sizes = []
    pos = []
    for i in range(len(trial['ll_cond'])):
        shape = trial['shape'][i]  # shapes[random.randrange(0, len(shapes))]
        # conditions[random.randrange(0, len(conditions))]
        condition = trial['ll_cond'][i]
        # to define: position,size (diameter 0.5 to 0.9deg of visual angle)
        # standard: random size, random pos
        # density: constant mean difference between dots
        # const_circum: summed circumference constant
        # linear: 1 size, pos on a line

        if condition == 'standard':

            if 'priming' in exp_name:
                size = get_random_size()
                sizes = [size]*trial['num']
            elif 'harvey' in exp_name:
                sizes = [0.15]*trial['num']
            else:
                sizes = [get_random_size() for i in range(trial['num'])]

            pos = get_positions(expInfo, vf_radius, sizes, shape)

        elif condition == 'linear':
            sizes = [get_random_size()]*trial['num']
            pos = get_positions(expInfo, 3, sizes, shape, cond='linear')

        elif condition == 'constarea':

            if shape == 'dot':

                # total area that will be maintained over all conditions
                # area = math.pi*radius**2
                total_area = np.pi*0.2**2
                radius_per_dot = np.sqrt((total_area / trial['num']) / np.pi)

                sizes = [radius_per_dot]*trial['num']
                # `get_positions` expects the size as r*2
                pos = get_positions(expInfo, vf_radius,
                                    [s*2 for s in sizes], shape, cond=condition)

            else:
                print('invalid shape for constarea condition! only dots possible')

        elif condition == 'constcirc':
            # calculate a constant circumference
            const_cir = 4 * 0.55  # 0.7 is avg  b/w 0.5 - 0.9 -- was too big
            const_per = 0.55 * 4  # 0.7 is avg b/w 0.5 - 0.9 -- was too big
            if shape == 'dot' or shape == 'triangle':
                # the pi factor would get cancel anyway
                diameter = const_cir/(trial['num'])
                sizes = [diameter for n in range(trial['num'])]
                pos = get_positions(expInfo,
                                    3, sizes, shape, cond='constcirc')
                assert pos != None
                print(shape, condition, diameter, pos)
            elif shape == 'rectangle':
                # the 4 factor would get cancelled anyway
                edge = const_per / (trial['num'])
                sizes = [edge for n in range(trial['num'])]
                pos = get_positions(expInfo, 3, sizes, shape,
                                    cond='constcirc')
                assert pos != None
                print(shape, condition, edge, pos)
        elif condition == 'density':
            # constant mean distance between dots
            sizes = [get_random_size() for i in range(trial['num'])]
            pos = get_positions(expInfo, 3, sizes, shape, cond='density')

        # set color for whole block of stimuli
        if 'colors' in trial:
            color = trial['colors']
        else:
            color = 'black'
        stimuli.append({'shapes': shape, 'size': sizes,
                        'pos': pos, 'cond': condition, 'color': color})
    return stimuli


def get_random_font():
    """Returns one of 'Arial','Times New Roman','Souvenir BT', 'Litograph Light' randomly.

    Returns:
        string: font name
    """
    fonts = ['Arial', 'Times New Roman', 'Souvenir BT', 'Litograph Light']
    return fonts[np.random.randint(0, 3)]


def get_random_font_size():
    """Returns font size from 26-42 randomly.
    Font size has no use for psychopy.

    Returns:
        int: font size
    """
    return np.random.randint(26, 42+1)


def get_random_size():
    """get random object size in degrees of visual angle between 0.5-0.9.

    Returns:
        float: object size between 0.5-0.9
    """
    return np.random.randint(5, 10)/10


def FixationCross(win, switch_on=True, exp_name=None, radius=None):
    """Draws a small white fixation cross on the window win. This will autodrawn forever even if win.flip is called.

    Args:
        win (psychopy.visual.window): window on screen
    """

    # rk: large fix cross for all
    if exp_name in ['harveyvisual', 'harveydigits', 'harveywritten', 'harveyaudio', 'harveyspoken', 'harveyspokenvisual_congruent', 'harveyspokenvisual_incongruent']:
        Line_1 = visual.Line(
            win=win, name='Line_1',
            start=(-1, -1), end=(1, 1),
            units='norm', ori=0, pos=(0, 0),
            lineWidth=1.5, lineColor=[255, 0, 0], colorSpace='rgb',
            fillColor=[1, 1, 1],
            opacity=1, depth=0.0, interpolate=True, autoDraw=False)
        Line_2 = visual.Line(
            win=win, name='Line_2',
            start=(-1, 1), end=(1, -1),
            units='norm', ori=0, pos=(0, 0),
            lineWidth=1.5, lineColor=[255, 0, 0], colorSpace='rgb',
            fillColor=[1, 1, 1],
            opacity=1, depth=0,  # -1.0,
            interpolate=True, autoDraw=False)
    elif exp_name == 'priming':
        Line_1 = visual.Line(
            win=win, name='Line_1',
            start=(-0.2, 0), end=(0.2, 0),
            units='deg', ori=0, pos=(0, 0),
            lineWidth=1.5, lineColor='black', colorSpace='rgb',
            fillColor=[1, 1, 1],
            opacity=1, depth=0.0, interpolate=True, autoDraw=False)
        Line_2 = visual.Line(
            win=win, name='Line_2',
            start=(0, -0.2), end=(0, 0.2),
            units='deg', ori=0, pos=(0, 0),
            lineWidth=1.5, lineColor='black', colorSpace='rgb',
            fillColor=[1, 1, 1],
            opacity=1, depth=0,  # -1.0,
            interpolate=True, autoDraw=False)

    else:
        Line_1 = visual.Line(
            win=win, name='Line_1',
            start=(-0.2, -0.2), end=(0.2, .2),
            units='deg',
            lineWidth=1.5, lineColor='white',
            depth=0,
            autoDraw=False)
        Line_2 = visual.Line(
            win=win, name='Line_2',
            start=(-.2, .2), end=(.2, -.2),
            units='deg',
            lineWidth=1.5, lineColor='white', colorSpace='rgb',
            depth=0,  # -1.0,
            autoDraw=False)

    if switch_on:
        Line_1.autoDraw = True
        Line_2.autoDraw = True

        win.flip()
    return [Line_1, Line_2]


def create_sequence_harveymixed(block_duration, samples_per_block, condition='unimodal', stim_type=None):
    # mix of spoken and visual

    # dot sequence with durations, colors etc
    visual_numbers = [2, 3, 4, 5]
    sequence = create_sequence_harvey(block_duration, samples_per_block,
                                      numbers=visual_numbers, percent_attention=.05)
    if condition != 'unimodal':

        if condition == 'congruent':
            audio_numbers = visual_numbers
        elif condition == 'incongruent':
            audio_numbers = [4, 5, 2, 3]
        # audio sequence
        additional_seq = create_sequence_harveyaudio(
            block_duration, samples_per_block, stim_type=stim_type, numbers=audio_numbers, percent_attention=.05)
        # modality of the sequence is 'mixed'
        sequence.loc[sequence['mod'] != 'pause', 'mod'] = 'mixed'
        # we only need audio category info for each trial from audio sequence
        sequence['audio_cat'] = additional_seq[[
            'audio_cat', 'num']].values.tolist()

    return sequence


def create_sequence_harveyaudio(block_duration, samples_per_block, stim_type=None, numbers=[1, 2, 3, 4, 5], percent_attention=0.1):
    # should have columns num,mod,duration,shape,ll_cond
    # one pattern consists of ascending and descending numerosity samples
    # pattern is repeated 4 times in the run

    # rk: timing of a block (and sub-block) are adjusted to be identical in length with the other modalities
    if stim_type == 'spoken':
        show_time = 0.6  # 0.75
        grey_time = 0.1  # 0.09
        show_time_20 = 0.6  # 0.75
        grey_time_20 = 0.1  # 0.09
        npattern = 4
        samples_per_num_smallnums = 6  # 5
        samples_per_num_20 = 6*4  # 5 * 4
    else:  # harveyaudio
        show_time = 0.5  # 0.3
        grey_time = 0.2  # 0.4
        show_time_20 = 1.93  # 0.8
        grey_time_20 = 0.17  # 0.25
        npattern = 4
        samples_per_num_smallnums = 6
        samples_per_num_20 = 8  # 16

    num_raw = numbers + [20] + numbers[::-1] + [20]
    num = [[n_stim, 0]*samples_per_num_smallnums
           if n_stim < 20
           else [n_stim, 0]*samples_per_num_20
           for n_stim
           in num_raw]
    num = np.array([item for sublist in num for item in sublist])

    duration = np.zeros(shape=len(num))
    mask_smallnums = np.logical_and(num > 0, num < 20)
    mask_20 = (num == 20)
    duration[mask_smallnums] = show_time
    duration[np.roll(mask_smallnums, 1)] = grey_time
    duration[mask_20] = show_time_20
    duration[np.roll(mask_20, 1)] = grey_time_20

    n_pres = int(len(num)/2)

    if stim_type == 'spoken':
        mod = ['audio', 'pause'] * n_pres
        # only the frequent categories (i.e., female)
        audio_categories = ['female_1', 'female_2', 'female_3']
    else:
        mod = ['audio', 'pause'] * n_pres  # original for harveyaudio
        audio_categories = ['sine_440khz', 'sine_333khz',
                            'sine_359khz', 'sine_392khz']  # ], 'sine_500khz']

    audio_cats_seq = np.random.choice(audio_categories, size=n_pres*npattern)
    audio_cat = [[sample]
                 for block
                 in zip(audio_cats_seq, [None]*n_pres*npattern)
                 for sample
                 in block]
    n_audio_events = int(len(audio_cat)/2)
    n_attention_events = int(n_audio_events*percent_attention)
    attention_events_index = np.random.choice(
        [i for i in range(n_audio_events)], size=n_attention_events, replace=False)*2
    audio_cat = np.array(audio_cat)

    if stim_type == 'spoken':  # sample from the oddball male voices
        odd_voices = np.array(["male_1", "male_2", "male_3"])
        n_odds = len(attention_events_index)
        random_odd_voices = np.random.choice(
            odd_voices, size=n_odds, replace=True).tolist()
        for ind, voice in zip(attention_events_index, random_odd_voices):
            audio_cat[ind] = [voice]

    else:  # original: harveyaudio
        audio_cat[attention_events_index] = ['sine_1000khz']

    events = pd.DataFrame(
        data={
            'num': num.tolist() * npattern,
            'mod': mod * npattern,
            'duration': duration.tolist() * npattern,
            'audio_cat': audio_cat.tolist()})

    return events


def create_sequence_harvey(block_duration, samples_per_block, stim_type=None, numbers=[1, 2, 3, 4, 5], percent_attention=.1):
    # should have columns num,mod,duration,shape,ll_cond
    # one pattern consists of ascending and descending numerosity samples
    # pattern is repeated 4 times in the run
    show_time = 0.3
    grey_time = 0.4
    npattern = 4
    adapt_multiplier = 4
    samples_per_num = 6
    # n_num_repeat_stim=int(time_one_stim/(show_time+grey_time))
    # n_num_repeat_adapt=int(time_one_adapt/(show_time+grey_time))

    num_raw = numbers + [20]*adapt_multiplier + \
        numbers[::-1] + [20]*adapt_multiplier
    num = []
    for n_stim in num_raw:
        for _ in range(samples_per_num):
            num.append(n_stim)
            num.append(0)
    n_pres = int(len(num)/2)
    # alternative: [y for x in range(20) for y in ('show_time','grey_time') ]
    duration = [show_time, grey_time] * n_pres
    shape = [['dot'], None]*n_pres
    ll_cond = [['constarea'], None]*n_pres
    if stim_type == 'digit':
        mod = ['digit', 'pause']*n_pres
    elif stim_type == 'written':
        mod = ['written', 'pause']*n_pres
    else:
        mod = ['dot', 'pause']*n_pres
    # [block_duration]*5+[16.8]+[block_duration]*5+[16.8]
    # shape = make_pattern(['dot']*samples_per_block, ['dot'])
    # [['standard']*samples_per_block]*blocks_per_pattern
    # ll_cond = make_pattern(['standard']*samples_per_block, ['standard'])
    # mod = ['dot']*blocks_per_pattern

    events = pd.DataFrame(
        data={'num': num*npattern,
              'mod': mod*npattern,
              'duration': duration*npattern,
              'shape': shape*npattern,
              'll_cond': ll_cond*npattern})

    # 10 * npattern*samples_per_block
    n_colored_pattern = int(len(events['num'])/2)
    n_white_pattern = round(n_colored_pattern*percent_attention)
    n_black_pattern = n_colored_pattern-n_white_pattern
    colors_to_distribute = ['white']*n_white_pattern+['black']*n_black_pattern
    np.random.shuffle(colors_to_distribute)
    colors = [color_row for color in colors_to_distribute for color_row in [
        color, None]]
    # colors = []
    # for num_raw in events['num']:
    #     if num_raw < 10:
    #         colors.append([colors_to_distribute.pop()
    #                       for _ in range(samples_per_block)])
    #     else:
    #         colors.append(['black'])
    events['colors'] = colors
    return events


def make_pattern(content_blocks, content_breaks):
    return [content_blocks]*5+[content_breaks]+[content_blocks]*5+[content_breaks]


def create_sequence(block_duration, samples_per_block):
    """Produce stimuli sequence with durations etc randomly

    Returns:
        pandas.dataframe: dataframe describing the exp order. one row is one
        block. columns include num,mod,duration
    """
    onsets = predefine_run_blocks.create_random_blocks(
        block_duration=block_duration, samples_per_block=samples_per_block)
    events = onsets['trial_type'].str.split('_', expand=True)

    events.rename(columns={0: 'num', 1: 'mod'}, inplace=True)
    # onsets['num'][onsets['num']=='Pause']=0
    events.loc[events['num'] == 'pause', ['mod']] = 'pause'
    events.loc[events['num'] == 'pause', ['num']] = 0
    events['num'] = events['num'].apply(int)
    events[['duration', 'gender_voice']] = onsets[['duration', 'gender_voice']]
    events['duration'] = events['duration'].apply(float)

    events['ll_cond'] = onsets['ll_cond'].str.split('_')
    events['shape'] = onsets['shape'].str.split('_')
    events['gender_voice'] = onsets['gender_voice'].str.split('_')

    return events


def get_positions(expInfo, vf_radius=3, sizes=None, shape='dots', cond='standard', max_search=100):
    """Calculate non-overlapping stimuli positions for 1 sample.
        Make random position in visual field with make_random_pos().
        call check_conflict() to check for overlap.
    Args:
        vf_radius (int, optional): Radius of visual field to be used. Defaults to 3.
        sizes (list, optional): list of floats describing the sizes of stimuli for this sample in degrees of visual angle. Defaults to None.
        shape (str, optional): shape of stimuli for this sample. Defaults to 'dots'.
        cond (str, optional): stimuli configuration condition. 'standard', 'linear', 'density' or 'constcirc'.

    Returns:
        list: list of positions for stimuli in this sample
    """
    num = len(sizes)
    if num == 20 and expInfo != 'prep':
        positions = get_positions_20(expInfo)
        return positions
    if cond == 'linear':
        positions = get_pos_line(vf_radius, sizes, shape, num)
    elif cond == 'density':
        positions = get_pos_density(vf_radius, sizes, shape, num)
    else:
        positions = []
        for i in range(num):
            counter = 0
            search = True
            while search:
                counter += 1
                # print('search')
                pos = make_random_pos(vf_radius, sizes[i])
                search = check_conflict(positions, sizes, pos, shape)
                # if search took too long, use any position
                if counter > max_search:
                    logging.warning(
                        'Search has been exhausted. Positions may overlap')

                    search = False
            positions.append(pos)
    return positions


def get_positions_20(expInfo):
    pos = expInfo['20dots'].sample(axis=0, n=1, replace=True)
    pos = pos.squeeze(axis=0).tolist()
    positions = [[float(x.strip(' []'))
                  for x in s.split(',')] for s in pos]

    return positions


def make_random_pos(vf_radius, size):
    """Create a viable position inside visual field circle so that shape does not go beyond circle

    Args:
        vf_radius (int or float): degrees of visual angle , radius of circle to find positions in
        size (float): size of object so object does not go beyond circle

    Returns:
        list: x,y position in degrees
    """
    dist_extra = 0.1  # distance of edges to aperture
    # since size controls for the diameter
    circle_r = vf_radius-size/2 - dist_extra

    x = np.random.uniform(-1*circle_r, circle_r)
    # circle equation: x^2+y^2=r^2
    y_max = np.sqrt(circle_r**2-x**2)
    y = np.random.uniform(-1*y_max, y_max)

    return [x, y]


def get_pos_density(vf_radius, sizes, shape, num):
    """Calculates shape positions for the high density condition. Each shape will
    be around 1.1 degrees of the center point.

    :param vf_radius: radius of visual field in which stimuli can be placed
    :type vf_radius: int
    :param sizes: list of sizes of the shapes
    :type sizes: list
    :param shape: shape of the stimuli ('dots','triangles' or 'digits')
    :type shape: string
    :param num: number of stimuli needed
    :type num: int
    :return: list of positions. each position a list of x,y in degrees
    :rtype: list of lists
    """
    # this is for the high density condition, not the constant density one
    i = 0
    counter = 0
    total_counter = 0
    # mean distance between points
    mean_dist = 1.1
    #  circle equation x^2+y^2=r^2
    # assume dist = dist between centers
    dist_sq = mean_dist**2

    positions = []
    # do until we have all positions
    while i < num:
        search = True
        if i > 0:

            while search:
                counter += 1
                total_counter += 1
                # make random vector with length mean_dist
                # add this vector to first position to get 2nd position
                vec_y = np.random.uniform(-np.sqrt(dist_sq),
                                          np.sqrt(dist_sq))
                vec_x = np.sqrt(dist_sq-vec_y**2)*np.random.choice([1, -1])
                pos = [positions[0][0]+vec_x, positions[0][1]+vec_y]

                print('search around first position')
                # check for overlap
                search = check_conflict(
                    positions, sizes, pos, shape, vf_radius)
                if total_counter > 1000:
                    logging.warning(
                        'Search has been exhausted. Positions may overlap')
                    search = False
                if counter > 100:
                    # if search unsuccessfull for number of times
                    # start new
                    i = 0
                    counter = 0
                    pos = None
                    positions = []
                    search = False

        while search:
            total_counter += 1
            print('search for center point of density')
            pos = make_random_pos(vf_radius, sizes[i])
            search = check_conflict(positions, sizes, pos, shape, vf_radius)
            if total_counter > 1000:
                logging.warning(
                    'Search has been exhausted. Positions may overlap')
                search = False
        if pos is not None:
            positions.append(pos)
            i += 1
    return positions


def get_pos_line(vf_radius, sizes, shape, num):
    """Calculates shape positions for the linear condition. Each shape will 
    be on a straight line connecting the stimuli in this sample.

    :param vf_radius: radius of visual field in which stimuli can be placed
    :type vf_radius: int
    :param sizes: list of sizes of the shapes
    :type sizes: list
    :param shape: shape of the stimuli ('dots','triangles' or 'digits')
    :type shape: string
    :param num: number of stimuli needed
    :type num: int
    :return: list of positions. each position a list of x,y in degrees
    :rtype: list of lists 
    """
    i = 0
    counter = 0
    total_counter = 0
    positions = []
    # do until we have all positions
    while i < num:
        search = True
        # if 2 points have been generated
        if i > 1:
            # line defined by two end points
            p1 = positions[0]
            p2 = positions[1]
            # vector defined by these endpoints
            vec = [b-a for a, b in zip(p1, p2)]
            # search for next point between p1 and p2
            while search:
                counter += 1
                total_counter += 1
                # get random dist
                fac = np.random.random()
                print('search on line')
                # point on line between p1 and p2
                pos = [p+v*fac for p, v in zip(p1, vec)]
                # check for overlap
                search = check_conflict(
                    positions, sizes, pos, shape, vf_radius)
                if total_counter > 1000:
                    logging.warning(
                        'Search has been exhausted. Using default positions')
                    return default_pos_line(sizes)

                if counter > 100:
                    # if search unsuccessfull for number of times
                    # keep p1, start new
                    i = 0
                    counter = 0
                    pos = positions[0]
                    positions = []
                    search = False

        while search:
            total_counter += 1
            print('search for line endpoints')
            pos = make_random_pos(vf_radius, sizes[i])
            search = check_conflict(positions, sizes, pos, shape, vf_radius)
            if total_counter > 1000:
                logging.warning(
                    'Search has been exhausted. Using default positions')
                return default_pos_line(sizes)
        positions.append(pos)
        i += 1
    return positions


def default_pos_line(sizes):
    positions = [[-2.5, 0], [-1, 0], [1, 0], [2.5, 0]]
    return positions[:len(sizes)]


def check_conflict(positions, sizes, pos, shape, vf_radius=None):
    """Check for conflict (overlap) between new position and list of positions

    Args:
        positions (list): list of fixed positions of stimuli
        sizes (list): list of sizes of stimuli
        pos (list): x,y in visual degrees of new position that needs to be checked
        shape (string): shape of stimuli for this sample
        vf_radius (int): radius of visual field in degrees

    Returns:
        boolean: true if the position conflicts with the fixed ones
    """
    conflict = False
    size = sizes[len(positions)]

    for i, pos_x in enumerate(positions):
        size_x = sizes[i]
        if check_overlap(size, pos, size_x, pos_x, shape, vf_radius):
            return True

    return conflict


def check_overlap(size, pos, size_x, pos_x, shape, vf_radius=3):
    """Check if two objects, defined by their positions, sizes and shape, overlap.

    Args:
        size (float): size of new stimuli
        pos (list): x,y of new position
        size_x (float): size of old stimulus
        pos_x (list): x,y of old position
        shape (string): shape of stimuli
        vf_radius(int): radius of visual field in degrees

    Returns:
        boolean: true if stimuli on positions overlap
    """

    dist_extra = 0.1
    overlap = True
    if 'dot' in shape or 'triangle' in shape:
        r = size/2
        r_x = size_x/2
        dist_r = r+r_x+dist_extra
        dist_x = pos[0]-pos_x[0]
        dist_y = pos[1]-pos_x[1]

        if dist_x**2 + dist_y**2 > dist_r**2:
            overlap = False

    elif 'rectangle' in shape:

        overlap = False
        # add dist_extra on each side of rectangle
        size = size+2*dist_extra
        x_left_new = pos[0]-size*0.5
        x_right_new = pos[0]+size*0.5
        x_left_old = pos_x[0]-size_x*0.5
        x_right_old = pos_x[0]+size_x*0.5

        y_down_new = pos[1]-size*0.5
        y_up_new = pos[1]+size*0.5
        y_down_old = pos_x[1]-size_x*0.5
        y_up_old = pos_x[1]+size_x*0.5

        if (x_left_new >= x_left_old and x_left_new <= x_right_old) or (x_right_new <= x_right_old and x_right_new >= x_left_old):
            if (y_down_new >= y_down_old and y_down_new <= y_up_old) or (y_up_new <= y_up_old and y_up_new >= y_down_old):
                overlap = True

        if (x_left_old >= x_left_new and x_left_old <= x_right_new) or (x_right_old <= x_right_new and x_right_old >= x_left_new):
            if (y_down_old >= y_down_new and y_down_old <= y_up_new) or (y_up_old <= y_up_new and y_up_old >= y_down_new):
                overlap = True

    if vf_radius and not overlap:
        # check for overlap with aperture
        # since size controls for the diameter
        circle_r = vf_radius-size/2 - dist_extra
        # pythagoras
        if circle_r**2 - pos[0]**2 - pos[1]**2 < 0:
            overlap = True

    return overlap


def generate_stimulus_onsets(num_blocks, numerosity, stimulus_types):
    """ Randomly shuffle the stimulus presentation and split
    their proportions evenly among the blocks
    :param num_blocks: number of blocks
    :type num_blocks: int
    :param numerosity: the numerosities we are in interested in
    :type numerosity: list of ints
    :param stimulus_types: modalities of the stimuli being presented
    :type stimulus_types: list of strings
    :return: order of stimulus presentation
    :rtype: list of tuples
    """

    # Now that the number of blocks is known we can distribute it among numerosities
    # stimulus types
    combinations = list(it.product(numerosity, stimulus_types))
    stimuli_run = []
    for k in range(int(num_blocks//len(combinations))):
        # random.shuffle(combinations)
        stimuli_run.extend(combinations)

    np.random.shuffle(combinations)
    # we don't want the same combinations to get added all the time so we shuffle here
    if len(stimuli_run) < num_blocks:
        stimuli_run.extend(combinations[: int(num_blocks - len(stimuli_run))])

    # first we create a sequence of let's say 100 blocks and then randomly shuffle them
    np.random.shuffle(stimuli_run)

    return stimuli_run, combinations


def check_n_matches(stimuli_run, n=3):
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
            if stimuli_run[idx1][0] == stimuli_run[idx2][0]:
                count_numerosity += 1  # means the numerosities are matching
            if stimuli_run[idx1][1] == stimuli_run[idx2][1]:
                count_modality += 1
        if count_modality == n or count_numerosity == n:
            return False
    return True


def write_onsets(run_duration, block_duration, stimuli_run, num_blocks, generated_pauses):
    """ Generate the onsets of stimuli in a run
    :param run_duration: total duration of one run
    :type run_duration: float
    :param block_duration: duration of one block
    :type block_duration: float
    :param stimuli_run: the sequence of the presented stimuli
    :type stimuli_run: list of tuples
    :param num_blocks: number of blocks
    :type num_blocks: int
    :param generated_pauses: the sequences of the pauses generated
    :type generated_pauses: list
    :return: timings of stimulus presentation
    :rtype: list of lists
    """
    # %% Now we must get the timings of the stimulus presentation and fixation/pauses
    elapsed_time = 0
    onsets = []
    i = 0
    j = 0
    while (elapsed_time <= run_duration):
        if i < num_blocks:
            onsets.append([elapsed_time, block_duration, stimuli_run[i]])
            i += 1
            elapsed_time += block_duration
        if j < len(generated_pauses):
            onsets.append([elapsed_time, generated_pauses[j], 0])
            elapsed_time += generated_pauses[j]
            j += 1
        if np.absolute(sum(generated_pauses) + num_blocks * block_duration - elapsed_time) < 0.1:
            break

    return onsets


def read_trials():
    """read in order of trials form tsv file

    Returns:
        pandas.dataframe: dataframe describing the exp order. one row is one
        block. columns include num,mod,duration
    """
    onsets = pd.read_csv('emprise_onsets.tsv', sep='\t')
    onsets = pd.concat([onsets['duration'], onsets['trial_type'].str.replace(
        pat='\(|\)', repl='').str.split(',', expand=True)], axis=1)
    onsets.rename(columns={0: 'num', 1: 'mod'}, inplace=True)
    # onsets['num'][onsets['num']=='Pause']=0
    onsets.loc[onsets['num'] == 'Pause', ['mod']] = 'pause'
    onsets.loc[onsets['num'] == 'Pause', ['num']] = 0
    onsets['num'] = onsets['num'].apply(int)
    onsets['duration'] = onsets['duration'].apply(float)
    # change to return onsets for full-length
    onsets2 = onsets.iloc[:20]

    return onsets2
