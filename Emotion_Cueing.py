#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on Mon Nov 23 11:50:10 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'Emotion_Cueing'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1024,768]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/ishapatel/Desktop/PSY310_Lab/Solo Project/Emotion_Cueing.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    ins = visual.TextStim(win=win, name='ins',
        text='Instructions\n\nKeep your eyes focused on the plus sign (+) in the centre of the screen at all times.\n\nFollowed by a brief cue flash. It will point to where the target is likely to appear most of the time.\nThe moment the target (● - a dot) appears, press the corresponding arrow key on your keyboard:\n\t• Press the LEFT (←) arrow key if the target is on the left.\n\t• Press the RIGHT (→) arrow key if the target is on the right.\n\nPress SPACE to begin. ',
        font='Arial',
        pos=[0,0], draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_ins = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Trial" ---
    fixation1 = visual.ShapeStim(
        win=win, name='fixation1', vertices='cross',units='pix', 
        size=(10,10),
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from code
    import random
    
    neutral_images = [
        'DistAOne copy/1333.jpg',
        'DistAOne copy/1450.jpg',
        'DistAOne copy/1670.jpg',
        'DistAOne copy/1812.jpg',
        'DistAOne copy/2070.jpg',
        'DistAOne copy/2102.jpg',
        'DistAOne copy/2104.jpg',
        'DistAOne copy/2153.jpg',
        'DistAOne copy/2190.jpg',
        'DistAOne copy/2200.jpg',
        'DistAOne copy/2205.jpg',
        'DistAOne copy/2215.jpg',
        'DistAOne copy/2221.jpg',
        'DistAOne copy/2235.jpg',
        'DistAOne copy/2270.jpg',
        'DistAOne copy/2273.jpg',
        'DistAOne copy/2312.jpg',
        'DistAOne copy/2320.jpg',
        'DistAOne copy/2377.jpg',
        'DistAOne copy/2383.jpg',
        'DistAOne copy/2396.jpg',
        'DistAOne copy/2397.jpg',
        'DistAOne copy/2411.jpg',
        'DistAOne copy/2440.jpg',
        'DistAOne copy/2489.jpg',
        'DistAOne copy/2491.jpg',
        'DistAOne copy/2493.jpg',
        'DistAOne copy/2495.jpg',
        'DistAOne copy/2499.jpg',
        'DistAOne copy/2506.jpg',
        'DistAOne copy/2512.jpg',
        'DistAOne copy/2513.jpg',
        'DistAOne copy/2516.jpg',
        'DistAOne copy/2518.jpg',
        'DistAOne copy/2530.jpg',
        'DistAOne copy/2560.jpg',
        'DistAOne copy/2570.jpg',
        'DistAOne copy/2580.jpg',
        'DistAOne copy/2593.jpg',
        'DistAOne copy/2620.jpg',
        'DistAOne copy/2695.jpg',
        'DistAOne copy/2720.jpg',
        'DistAOne copy/2745.1.jpg',
        'DistAOne copy/2840.jpg',
        'DistAOne copy/2850.jpg',
        'DistAOne copy/2870.jpg',
        'DistAOne copy/2890.jpg',
        'DistAOne copy/2900.2.jpg',
        'DistAOne copy/2980.jpg',
        'DistAOne copy/4561.jpg',
        'DistAOne copy/4622.jpg',
        'DistAOne copy/5875.jpg',
        'DistAOne copy/7004.jpg',
        'DistAOne copy/7006.jpg',
        'DistAOne copy/7010.jpg',
        'DistAOne copy/7014.jpg',
        'DistAOne copy/7016.jpg',
        'DistAOne copy/7019.jpg',
        'DistAOne copy/7020.jpg',
        'DistAOne copy/7025.jpg',
        'DistAOne copy/7030.jpg',
        'DistAOne copy/7031.jpg',
        'DistAOne copy/7032.jpg',
        'DistAOne copy/7034.jpg',
        'DistAOne copy/7036.jpg',
        'DistAOne copy/7040.jpg',
        'DistAOne copy/7041.jpg',
        'DistAOne copy/7050.jpg',
        'DistAOne copy/7052.jpg',
        'DistAOne copy/7057.jpg',
        'DistAOne copy/7060.jpg',
        'DistAOne copy/7100.jpg',
        'DistAOne copy/7130.jpg',
        'DistAOne copy/7140.jpg',
        'DistAOne copy/7150.jpg',
        'DistAOne copy/7185.jpg',
        'DistAOne copy/7187.jpg',
        'DistAOne copy/7217.jpg',
        'DistAOne copy/7234.jpg',
        'DistAOne copy/7235.jpg',
        'DistAOne copy/7300.jpg',
        'DistAOne copy/7365.jpg',
        'DistAOne copy/7509.jpg',
        'DistAOne copy/7513.jpg',
        'DistAOne copy/7705.jpg',
        'DistAOne copy/7710.jpg',
        'DistAOne copy/7950.jpg',
        'DistAOne copy/9210.jpg',
        'DistAOne copy/9260.jpg',
        'DistAOne copy/9700.jpg',
    ]
    
    high_images = [
        'DistATwo copy/1022.jpg',
        'DistATwo copy/1026.jpg',
        'DistATwo copy/1080.jpg',
        'DistATwo copy/1101.jpg',
        'DistATwo copy/1114.jpg',
        'DistATwo copy/1271.jpg',
        'DistATwo copy/1302.jpg',
        'DistATwo copy/1303.jpg',
        'DistATwo copy/1310.jpg',
        'DistATwo copy/1321.jpg',
        'DistATwo copy/1726.jpg',
        'DistATwo copy/1820.jpg',
        'DistATwo copy/1931.jpg',
        'DistATwo copy/2045.jpg',
        'DistATwo copy/2057.jpg',
        'DistATwo copy/2058.jpg',
        'DistATwo copy/2071.jpg',
        'DistATwo copy/2120.jpg',
        'DistATwo copy/2352.1.jpg',
        'DistATwo copy/2375.1.jpg',
        'DistATwo copy/2458.jpg',
        'DistATwo copy/2800.jpg',
        'DistATwo copy/3010.jpg',
        'DistATwo copy/3017.jpg',
        'DistATwo copy/3061.jpg',
        'DistATwo copy/3170.jpg',
        'DistATwo copy/3211.jpg',
        'DistATwo copy/3301.jpg',
        'DistATwo copy/3302.jpg',
        'DistATwo copy/3360.jpg',
        'DistATwo copy/4007.jpg',
        'DistATwo copy/4008.jpg',
        'DistATwo copy/4085.jpg',
        'DistATwo copy/4142.jpg',
        'DistATwo copy/4180.jpg',
        'DistATwo copy/4210.jpg',
        'DistATwo copy/4220.jpg',
        'DistATwo copy/4232.jpg',
        'DistATwo copy/4300.jpg',
        'DistATwo copy/4302.jpg',
        'DistATwo copy/4490.jpg',
        'DistATwo copy/4598.jpg',
        'DistATwo copy/4601.jpg',
        'DistATwo copy/4604.jpg',
        'DistATwo copy/4631.jpg',
        'DistATwo copy/4647.jpg',
        'DistATwo copy/4649.jpg',
        'DistATwo copy/4651.jpg',
        'DistATwo copy/4664.1.jpg',
        'DistATwo copy/4666.jpg',
        'DistATwo copy/4668.jpg',
        'DistATwo copy/4669.jpg',
        'DistATwo copy/4672.jpg',
        'DistATwo copy/4683.jpg',
        'DistATwo copy/4692.jpg',
        'DistATwo copy/4693.jpg',
        'DistATwo copy/4697.jpg',
        'DistATwo copy/4698.jpg',
        'DistATwo copy/4770.jpg',
        'DistATwo copy/4800.jpg',
        'DistATwo copy/5920.jpg',
        'DistATwo copy/5950.jpg',
        'DistATwo copy/6230.jpg',
        'DistATwo copy/6350.jpg',
        'DistATwo copy/6550.jpg',
        'DistATwo copy/6900.jpg',
        'DistATwo copy/6910.jpg',
        'DistATwo copy/7360.jpg',
        'DistATwo copy/7361.jpg',
        'DistATwo copy/7450.jpg',
        'DistATwo copy/7499.jpg',
        'DistATwo copy/7640.jpg',
        'DistATwo copy/8001.jpg',
        'DistATwo copy/8065.jpg',
        'DistATwo copy/8160.jpg',
        'DistATwo copy/8178.jpg',
        'DistATwo copy/8179.jpg',
        'DistATwo copy/8191.jpg',
        'DistATwo copy/8192.jpg',
        'DistATwo copy/8206.jpg',
        'DistATwo copy/8251.jpg',
        'DistATwo copy/8260.jpg',
        'DistATwo copy/8341.jpg',
        'DistATwo copy/8475.jpg',
        'DistATwo copy/9156.jpg',
        'DistATwo copy/9295.jpg',
        'DistATwo copy/9332.jpg',
        'DistATwo copy/9582.jpg',
        'DistATwo copy/9596.jpg',
        'DistATwo copy/9926.jpg',
    ]
    
    
    
    image = visual.ImageStim(
        win=win,
        name='image', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(450,450),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    fixation2 = visual.ShapeStim(
        win=win, name='fixation2', vertices='cross',units='pix', 
        size=(10,10),
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    arrow1 = visual.TextStim(win=win, name='arrow1',
        text='',
        font='Arial',
        units='pix', pos=[0,0], draggable=False, height=60.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    fixation3 = visual.ShapeStim(
        win=win, name='fixation3', vertices='cross',units='pix', 
        size=(10,10),
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    target = visual.ShapeStim(
        win=win, name='target',units='pix', 
        size=(40,40), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank" ---
    blank1 = visual.TextStim(win=win, name='blank1',
        text=None,
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "end" ---
    End = visual.TextStim(win=win, name='End',
        text='Thank you for your participation!',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[ins, key_resp_ins],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_ins
    key_resp_ins.keys = []
    key_resp_ins.rt = []
    _key_resp_ins_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    thisExp.currentRoutine = instructions
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ins* updates
        
        # if ins is starting this frame...
        if ins.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            ins.frameNStart = frameN  # exact frame index
            ins.tStart = t  # local t and not account for scr refresh
            ins.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ins, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ins.started')
            # update status
            ins.status = STARTED
            ins.setAutoDraw(True)
        
        # if ins is active this frame...
        if ins.status == STARTED:
            # update params
            pass
        
        # *key_resp_ins* updates
        waitOnFlip = False
        
        # if key_resp_ins is starting this frame...
        if key_resp_ins.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_ins.frameNStart = frameN  # exact frame index
            key_resp_ins.tStart = t  # local t and not account for scr refresh
            key_resp_ins.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_ins, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_ins.started')
            # update status
            key_resp_ins.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_ins.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_ins.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_ins.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_ins.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_ins_allKeys.extend(theseKeys)
            if len(_key_resp_ins_allKeys):
                key_resp_ins.keys = _key_resp_ins_allKeys[-1].name  # just the last key pressed
                key_resp_ins.rt = _key_resp_ins_allKeys[-1].rt
                key_resp_ins.duration = _key_resp_ins_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if key_resp_ins.keys in ['', [], None]:  # No response was made
        key_resp_ins.keys = None
    thisExp.addData('key_resp_ins.keys',key_resp_ins.keys)
    if key_resp_ins.keys != None:  # we had a response
        thisExp.addData('key_resp_ins.rt', key_resp_ins.rt)
        thisExp.addData('key_resp_ins.duration', key_resp_ins.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=50, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('spatial_conditions_4.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Trial" ---
        # create an object to store info about Routine Trial
        Trial = data.Routine(
            name='Trial',
            components=[fixation1, image, fixation2, arrow1, fixation3, target, key_resp],
        )
        Trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        fixation1.setPos([0,0])
        # Run 'Begin Routine' code from code
        # Randomly choose target side
        target_side = random.choice(["left", "right"])
        
        if target_side == "left":
            target_x = -350
            correct_key = "left"
        else:
            target_x = 350
            correct_key = "right"
        
        # Choose arrow direction
        if validity == "valid":
            if target_side == "left":
                cue = "←"
            else:
                cue = "→"
        else:
            if target_side == "left":
                cue = "→"
            else:
                cue = "←"
        
        if emotion == "neutral":
            image_file = random.choice(neutral_images)
        else:
            image_file = random.choice(high_images)
        image.setImage(image_file)
        fixation2.setPos([0,0])
        arrow1.setText(cue)
        fixation3.setPos([0,0])
        target.setPos((target_x, 0))
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for Trial
        Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Trial.tStart = globalClock.getTime(format='float')
        Trial.status = STARTED
        thisExp.addData('Trial.started', Trial.tStart)
        Trial.maxDuration = None
        # keep track of which components have finished
        TrialComponents = Trial.components
        for thisComponent in Trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Trial" ---
        thisExp.currentRoutine = Trial
        Trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.7:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation1* updates
            
            # if fixation1 is starting this frame...
            if fixation1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                fixation1.frameNStart = frameN  # exact frame index
                fixation1.tStart = t  # local t and not account for scr refresh
                fixation1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation1.started')
                # update status
                fixation1.status = STARTED
                fixation1.setAutoDraw(True)
            
            # if fixation1 is active this frame...
            if fixation1.status == STARTED:
                # update params
                pass
            
            # if fixation1 is stopping this frame...
            if fixation1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation1.tStop = t  # not accounting for scr refresh
                    fixation1.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation1.stopped')
                    # update status
                    fixation1.status = FINISHED
                    fixation1.setAutoDraw(False)
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # *fixation2* updates
            
            # if fixation2 is starting this frame...
            if fixation2.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                # keep track of start time/frame for later
                fixation2.frameNStart = frameN  # exact frame index
                fixation2.tStart = t  # local t and not account for scr refresh
                fixation2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation2.started')
                # update status
                fixation2.status = STARTED
                fixation2.setAutoDraw(True)
            
            # if fixation2 is active this frame...
            if fixation2.status == STARTED:
                # update params
                pass
            
            # if fixation2 is stopping this frame...
            if fixation2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation2.tStartRefresh + 0.4-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation2.tStop = t  # not accounting for scr refresh
                    fixation2.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation2.stopped')
                    # update status
                    fixation2.status = FINISHED
                    fixation2.setAutoDraw(False)
            
            # *arrow1* updates
            
            # if arrow1 is starting this frame...
            if arrow1.status == NOT_STARTED and tThisFlip >= 1.1-frameTolerance:
                # keep track of start time/frame for later
                arrow1.frameNStart = frameN  # exact frame index
                arrow1.tStart = t  # local t and not account for scr refresh
                arrow1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrow1.started')
                # update status
                arrow1.status = STARTED
                arrow1.setAutoDraw(True)
            
            # if arrow1 is active this frame...
            if arrow1.status == STARTED:
                # update params
                pass
            
            # if arrow1 is stopping this frame...
            if arrow1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrow1.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    arrow1.tStop = t  # not accounting for scr refresh
                    arrow1.tStopRefresh = tThisFlipGlobal  # on global time
                    arrow1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrow1.stopped')
                    # update status
                    arrow1.status = FINISHED
                    arrow1.setAutoDraw(False)
            
            # *fixation3* updates
            
            # if fixation3 is starting this frame...
            if fixation3.status == NOT_STARTED and tThisFlip >= 1.3-frameTolerance:
                # keep track of start time/frame for later
                fixation3.frameNStart = frameN  # exact frame index
                fixation3.tStart = t  # local t and not account for scr refresh
                fixation3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation3.started')
                # update status
                fixation3.status = STARTED
                fixation3.setAutoDraw(True)
            
            # if fixation3 is active this frame...
            if fixation3.status == STARTED:
                # update params
                pass
            
            # if fixation3 is stopping this frame...
            if fixation3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation3.tStartRefresh + 0.4-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation3.tStop = t  # not accounting for scr refresh
                    fixation3.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation3.stopped')
                    # update status
                    fixation3.status = FINISHED
                    fixation3.setAutoDraw(False)
            
            # *target* updates
            
            # if target is starting this frame...
            if target.status == NOT_STARTED and tThisFlip >= 1.7-frameTolerance:
                # keep track of start time/frame for later
                target.frameNStart = frameN  # exact frame index
                target.tStart = t  # local t and not account for scr refresh
                target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target.started')
                # update status
                target.status = STARTED
                target.setAutoDraw(True)
            
            # if target is active this frame...
            if target.status == STARTED:
                # update params
                pass
            
            # if target is stopping this frame...
            if target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    target.tStop = t  # not accounting for scr refresh
                    target.tStopRefresh = tThisFlipGlobal  # on global time
                    target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target.stopped')
                    # update status
                    target.status = FINISHED
                    target.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 1.7-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct_key)) or (key_resp.keys == correct_key):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Trial" ---
        for thisComponent in Trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Trial
        Trial.tStop = globalClock.getTime(format='float')
        Trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Trial.stopped', Trial.tStop)
        # Run 'End Routine' code from code
        thisExp.addData("image_file", image_file)
        thisExp.addData("cue_direction", cue)
        thisExp.addData("target_side", target_side)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp.keys',key_resp.keys)
        trials.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Trial.maxDurationReached:
            routineTimer.addTime(-Trial.maxDuration)
        elif Trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.700000)
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[blank1],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        blank1.setText('')
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        thisExp.currentRoutine = blank
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blank1* updates
            
            # if blank1 is starting this frame...
            if blank1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                blank1.frameNStart = frameN  # exact frame index
                blank1.tStart = t  # local t and not account for scr refresh
                blank1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank1.started')
                # update status
                blank1.status = STARTED
                blank1.setAutoDraw(True)
            
            # if blank1 is active this frame...
            if blank1.status == STARTED:
                # update params
                pass
            
            # if blank1 is stopping this frame...
            if blank1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    blank1.tStop = t  # not accounting for scr refresh
                    blank1.tStopRefresh = tThisFlipGlobal  # on global time
                    blank1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank1.stopped')
                    # update status
                    blank1.status = FINISHED
                    blank1.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 50 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[End],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    thisExp.currentRoutine = end
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *End* updates
        
        # if End is starting this frame...
        if End.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            End.frameNStart = frameN  # exact frame index
            End.tStart = t  # local t and not account for scr refresh
            End.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(End, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'End.started')
            # update status
            End.status = STARTED
            End.setAutoDraw(True)
        
        # if End is active this frame...
        if End.status == STARTED:
            # update params
            pass
        
        # if End is stopping this frame...
        if End.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > End.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                End.tStop = t  # not accounting for scr refresh
                End.tStopRefresh = tThisFlipGlobal  # on global time
                End.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'End.stopped')
                # update status
                End.status = FINISHED
                End.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            end.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if end.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
