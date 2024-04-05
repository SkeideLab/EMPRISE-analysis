import serial
from psychopy import core


def identify_targets():
    # make list with one dict per port. Dict contains port and boolean that controls if it will be used
    sers = [{'ser': serial.Serial(
        f"/dev/ttyUSB{ser_num}", timeout=0), 'use': False} for ser_num in [0, 1]]
    for target in ['Buttons', 'Sensor']:
        ser_name = identify_serialport(target=target, sers=sers)
        ser_name = ser_name.replace('/dev/ttyUSB', '')
        print(f"Der Port für den/die {target} ist: {ser_name}")
        core.wait(2)
    print('Bitte Port-Nummer (1 oder 0) notieren und später eingeben!')
    print('Das Skript "show_movie_motion.py" benötigt den Port des Sensors, \ndie anderen benötigen den Port der Buttons!')
    for ser in sers:
        ser['ser'].close()


def identify_serialport(target, sers):

    if target == 'Buttons':
        # -------------------serial port------------------------------------------
        print('**********************************************************')
        print('ACHTUNG: Bitte Motion Sensor nicht bewegen!')
        print('ACHTUNG: Bitte einen der Knöpfe drücken!')
        print('***********************************************************')
    elif target == 'Sensor':
        # -------------------serial port------------------------------------------
        print('**********************************************************')
        print('ACHTUNG: Bitte Knöpfe nicht drücken!')
        print('ACHTUNG: Bitte Motion Sensor einmal kurz bewegen!')
        print('***********************************************************')
    else:
        print('error')
    [ser['ser'].reset_input_buffer() for ser in sers]
    # while we have no response from either port
    while not any([ser['use'] for ser in sers]):
        for ser in sers:
            resp = list(ser['ser'].read())
            if len(resp) > 0:
                ser['use'] = True

    # assign port that will be used to ser
    ser_name = [ser['ser'] for ser in sers if ser['use'] == True][0].port
    for ser in sers:
        ser['use'] = False
    return ser_name


def main():
    identify_targets()


if __name__ == "__main__":
    main()
