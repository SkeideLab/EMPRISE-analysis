#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on Juli 27, 2022, at 13:40
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'priming'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\Anne-Sophie\\Work\\EMPRISE\\experiment\\priming\\priming_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=(1024, 768), fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "trial"
trialClock = core.Clock()
fixation_cross = visual.ShapeStim(
    win=win, name='fixation_cross', vertices='cross',
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
mask1 = visual.TextStim(win=win, name='mask1',
    text='XX',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
num_word = visual.TextStim(win=win, name='num_word',
    text='num_word',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
mask2 = visual.TextStim(win=win, name='mask2',
    text='XX',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
num_dots = visual.ShapeStim(
    win=win, name='num_dots',
    size=(0.5, 0.5), vertices='circle',
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-4.0, interpolate=True)
key_resp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=5.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    trialComponents = [fixation_cross, mask1, num_word, mask2, num_dots, key_resp]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_cross* updates
        if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_cross.frameNStart = frameN  # exact frame index
            fixation_cross.tStart = t  # local t and not account for scr refresh
            fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
            fixation_cross.setAutoDraw(True)
        if fixation_cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_cross.tStartRefresh + 0.25-frameTolerance:
                # keep track of stop time/frame for later
                fixation_cross.tStop = t  # not accounting for scr refresh
                fixation_cross.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fixation_cross, 'tStopRefresh')  # time at next scr refresh
                fixation_cross.setAutoDraw(False)
        
        # *mask1* updates
        if mask1.status == NOT_STARTED and tThisFlip >= 0.25-frameTolerance:
            # keep track of start time/frame for later
            mask1.frameNStart = frameN  # exact frame index
            mask1.tStart = t  # local t and not account for scr refresh
            mask1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mask1, 'tStartRefresh')  # time at next scr refresh
            mask1.setAutoDraw(True)
        if mask1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mask1.tStartRefresh + 0.25-frameTolerance:
                # keep track of stop time/frame for later
                mask1.tStop = t  # not accounting for scr refresh
                mask1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(mask1, 'tStopRefresh')  # time at next scr refresh
                mask1.setAutoDraw(False)
        
        # *num_word* updates
        if num_word.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            num_word.frameNStart = frameN  # exact frame index
            num_word.tStart = t  # local t and not account for scr refresh
            num_word.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(num_word, 'tStartRefresh')  # time at next scr refresh
            num_word.setAutoDraw(True)
        if num_word.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > num_word.tStartRefresh + 0.05-frameTolerance:
                # keep track of stop time/frame for later
                num_word.tStop = t  # not accounting for scr refresh
                num_word.frameNStop = frameN  # exact frame index
                win.timeOnFlip(num_word, 'tStopRefresh')  # time at next scr refresh
                num_word.setAutoDraw(False)
        
        # *mask2* updates
        if mask2.status == NOT_STARTED and tThisFlip >= 0.55-frameTolerance:
            # keep track of start time/frame for later
            mask2.frameNStart = frameN  # exact frame index
            mask2.tStart = t  # local t and not account for scr refresh
            mask2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mask2, 'tStartRefresh')  # time at next scr refresh
            mask2.setAutoDraw(True)
        if mask2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mask2.tStartRefresh + 0.25-frameTolerance:
                # keep track of stop time/frame for later
                mask2.tStop = t  # not accounting for scr refresh
                mask2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(mask2, 'tStopRefresh')  # time at next scr refresh
                mask2.setAutoDraw(False)
        
        # *num_dots* updates
        if num_dots.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            num_dots.frameNStart = frameN  # exact frame index
            num_dots.tStart = t  # local t and not account for scr refresh
            num_dots.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(num_dots, 'tStartRefresh')  # time at next scr refresh
            num_dots.setAutoDraw(True)
        if num_dots.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > num_dots.tStartRefresh + 0.25-frameTolerance:
                # keep track of stop time/frame for later
                num_dots.tStop = t  # not accounting for scr refresh
                num_dots.frameNStop = frameN  # exact frame index
                win.timeOnFlip(num_dots, 'tStopRefresh')  # time at next scr refresh
                num_dots.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        if key_resp.status == NOT_STARTED and tThisFlip >= 1.05-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + 0.950-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.frameNStop = frameN  # exact frame index
                win.timeOnFlip(key_resp, 'tStopRefresh')  # time at next scr refresh
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['y', 'n', 'left', 'right', 'space'], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('fixation_cross.started', fixation_cross.tStartRefresh)
    trials.addData('fixation_cross.stopped', fixation_cross.tStopRefresh)
    trials.addData('mask1.started', mask1.tStartRefresh)
    trials.addData('mask1.stopped', mask1.tStopRefresh)
    trials.addData('num_word.started', num_word.tStartRefresh)
    trials.addData('num_word.stopped', num_word.tStopRefresh)
    trials.addData('mask2.started', mask2.tStartRefresh)
    trials.addData('mask2.stopped', mask2.tStopRefresh)
    trials.addData('num_dots.started', num_dots.tStartRefresh)
    trials.addData('num_dots.stopped', num_dots.tStopRefresh)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    trials.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        trials.addData('key_resp.rt', key_resp.rt)
    trials.addData('key_resp.started', key_resp.tStartRefresh)
    trials.addData('key_resp.stopped', key_resp.tStopRefresh)
    thisExp.nextEntry()
    
# completed 5.0 repeats of 'trials'


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
