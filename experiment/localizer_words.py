# Import Packages, Libraries and Files
from psychopy import visual, core, event, sound, __version__, data, gui, logging, monitors
from psychopy.hardware.emulator import launchScan

import os

def prepare_audio_stim(exp_duration, stim_duration):
    sounds = []
    voice_g = ['male', 'female']
    dir = os.path.dirname(os.path.abspath(__file__))
    for gender in voice_g:

        audio_dir = os.path.join(dir, 'audio', gender)
        for filename in os.listdir(audio_dir):
            if not 'eins' in filename:
                filepath = os.path.join(audio_dir, filename)
                sounds.append({'stim': sound.Sound(
                    filepath, name='/'.join(filepath.split(os.path.sep)[-2:]))})

    num_unique_stim = len(sounds)
    num_stim = int(exp_duration / stim_duration)
    n_reps, _ = divmod(num_stim, num_unique_stim)
    # if the duration of the rest time is >1
    if exp_duration - n_reps*num_unique_stim * stim_duration > 1:
        n_reps += 1

    trials = data.TrialHandler(
        trialList=sounds, nReps=n_reps, method='random', name='trials')

    return trials


def FixationCross(win):
    """Draws a small white fixation cross on the window win. This will autodrawn forever even if win.flip is called.

    Args:
        win (psychopy.visual.window): window on screen
    """
    Line_1 = visual.Line(
        win=win, name='Line_1',
        start=(-0.05, -0.05), end=(0.05, .05),
        units='height',
        lineWidth=1.5, lineColor='white',
        depth=0,
        autoDraw=True)
    Line_2 = visual.Line(
        win=win, name='Line_2',
        start=(-.05, .05), end=(.05, -.05),
        units='height',
        lineWidth=1.5, lineColor='white', colorSpace='rgb',
        depth=0,  # -1.0,
        autoDraw=True)

    win.flip()


def show_face(win):
    face = visual.ImageStim(
        win=win, image='localizer_stims/speaking_face.png', name='speaking_face', size=4)
    face.autoDraw = True
    win.flip()
    return face


if __name__ == '__main__':
    # Ensure that relative paths start from the same directory as this script
    _thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_thisDir)
# Store info about the experiment session
    expName = 'localizer-words'
    expInfo = {'participant': '', 'session': ''
               }

    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()

     #---MRI-Scanner-Interface---#
    # settings for launchScan:
    MR_settings = {
        'TR': 2.000,     # duration (sec) per whole-brain volume
        # number of whole-brain 3D volumes per scanning run (for simulation mode!)
        'volumes': 5,
        # character to use as the sync timing event; assumed to come at start of a volume
        'sync': 't',
        # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
        'skip': 0,
        'sound': False    # in test mode: play a tone as a reminder of scanner noise
    }

    expInfo['date'] = data.getDateStr(format='%Y_%m_%d_%H%M')
    expInfo['psychopy version'] = __version__

    # Clocks
    global_clock = core.Clock()
    trial_clock = core.CountdownTimer()

    Monitor_data = [1024, 768, 40, 118]

    # this saves the monitor config to a psychopy settings file so we can use it later
    mon = monitors.Monitor(
        'stimulus_screen', width=Monitor_data[2], distance=Monitor_data[3])
    mon.setSizePix((Monitor_data[0], Monitor_data[1]))
    mon.save()

    # LOGGING -------------------------------------------------------------
    out_path = os.path.join(_thisDir, 'logfiles')
    # if not os.path.exists(out_path):
    #     os.mkdir(out_path)
    # filename = os.path.join(out_path, u'sub-%s_ses-%s_%s_%s' %
    #                         (expInfo['participant'], expInfo['session'], expName, expInfo['date']))
    for identifier in ['participant', 'session']:
            if not expInfo[identifier]:
                expInfo[identifier] = '_'
    out_path = os.path.join(
            out_path, expInfo['session'], f"{expInfo['participant']}_d-{data.getDateStr(format='%Y%m%d')}")

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    run_id = u'sub-%s_ses-%s_%s_%s' % (
                    expInfo['participant'], expInfo['session'], expName, data.getDateStr(format='%Y_%m_%d_%H%M'))
    print(f"Now running: {run_id}")
    filename = os.path.join(out_path, run_id)
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)
    logging.setDefaultClock(global_clock)
    logging.exp(expInfo)
    logging.exp(MR_settings)

    win = visual.Window(
        size=(Monitor_data[0], Monitor_data[1]), fullscr=True, screen=0,
        winType='pyglet', allowGUI=False,
        monitor='stimulus_screen', color='black', colorSpace='rgb',
        blendMode='avg', useFBO=True,
        units='deg', name='localizer_window')

    # timing in seconds
    exp_duration = 60
    stim_duration = 0.6

    trials = prepare_audio_stim(exp_duration, stim_duration)
    #sounds = prepare_audio_stim()
    ############################################################################
    vol = launchScan(win, MR_settings, globalClock=global_clock,mode='Scan')
    ############################################################################
    vf_circ=visual.Circle(win, radius=3, edges=32, lineWidth=0, lineColor=None,
                  lineColorSpace='rgb', fillColor='grey', fillColorSpace='rgb', autoDraw=True, name='feedback_circle')

    face=show_face(win)

    start_time = global_clock.getTime()

    # short delay after sync pulse so voice is not so startling
    while global_clock.getTime() < start_time+0.4:
        keys = event.getKeys(keyList=['escape'])
            # print(keys)
        if keys:
            print('goodbye')
            break

    trial_clock.reset()
    for i in range(trials.nTotal):
        trial_clock.add(stim_duration)
        trials.next()['stim'].play()
        trials.addData('onset', global_clock.getTime())
        trials.addData('duration', 0.4)
        trials.addData(
            'trial_type', trials.trialList[trials.thisIndex]['stim'].name)
        while trial_clock.getTime() > 0.05:
            keys = event.getKeys(keyList=['escape'])
            # print(keys)
            if keys:
                print('goodbye')
                break
        if keys:
            print('goodbye')
            break
        if global_clock.getTime() > (exp_duration-stim_duration):
            break
            # audio.play()
            # core.wait(0.6)
            # if event.getKeys():
            #     core.quit()

    end_time = global_clock.getTime()
    run_time = end_time - start_time
    print('full run time: '+str(run_time))
    print(end_time)
    logging.exp('full run time: '+str(run_time))
    trials.saveAsWideText(fileName=filename+'_events.tsv')

    face.autoDraw=False
    vf_circ.autoDraw=False
    win.flip()
    while global_clock.getTime() < exp_duration+5:
        keys = event.getKeys(keyList=['escape'])
        # print(keys)
        if keys:
            print('goodbye')
            break
    core.quit()
