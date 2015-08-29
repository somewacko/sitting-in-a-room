'''
    sitting-in-a-room.py | zo7

    Runs Alvin Lucier's "I Am Sitting In A Room" for any sound or room.

    Takes an input signal (x) and an impulse response (h), computing the
    following system for a non-negative number n:

        y(0) = x + x*h
        y(n) = y(n-1) + y(n-1)*h   (where * is the convolution operator)

    The main difference between this and Alvin's piece is that this is
    allowing the sound to play all the way through, while Alvin has his tape
    machines looping continually.

    Usage:
        python sitting-in-a-room.py {input filename} {ir filename} {n}

        Files must be 1-channel, 16-bit .wav files, ideally with the same
        sampling rate.
'''

import getopt, os, sys, time
import numpy as np
import scipy.io.wavfile as wav


# ---- Utility functions

def scale_wav(signal):
    ''' Scales a 16-bit wav signal to float between -1.0 and 1.0 '''

    return signal * scale_wav.factor

scale_wav.factor = 1 / float(2**15)


def normalize(signal):
    ''' Normalizes a signal so that max(|sig|) = 1.0. '''

    return signal / np.max(np.abs(signal))


def next_pow_2(n):
    ''' Finds next greatest power of 2 from 'n'. '''

    i = 2
    while i < n:
        i *= 2
    return i


def is_audible(sample):
    ''' Checks if a given float sample between -1 and 1 would be audible in a
        16bit .wav file. '''

    return sample <= -is_audible.smallest or is_audible.smallest <= sample

is_audible.smallest = 1 / float(2**15)


def find_end(signal):
    ''' Finds the end of the signal where it stops being audible. '''

    i = 1
    while not is_audible(signal[-i]) and i <= len(signal):
        i += 1
    return len(signal) - i


def write_to_wav(data, sampling_rate, output_filename):
    ''' Writes the data to a .wav file. '''

    # Normalize and scale it to fit float to int16
    scaled = np.int16(data/np.max(np.abs(data)) * np.power(2, 15))

    wav.write(output_filename, sampling_rate, scaled)


# ---- Sitting in a room...

def fft_convolve(a, b):
    ''' Convolves two signals 'a' and 'b' using FFT. '''

    a = a.copy()
    b = b.copy()

    # Determine the length of the new signal - Take the sum of the lengths of
    # the signals and go to the next power of 2 (to make FFT run optimally).
    length = next_pow_2( len(a) + len(b) )

    # Resize and zero-pad the signals for our new length.
    a.resize(length)
    b.resize(length)

    # Compute the convolution.
    result = np.real(np.fft.ifft( np.fft.fft(a) * np.fft.fft(b) ))

    # Return only the audible portion
    end = find_end(result)
    return result[0:end]


def sitting_in_a_room(input_signal, ir_signal, n, loud=False):
    ''' Takes an input signal and an impulse response and convolves the signal
        by the impulse convolved 'n' times by itself. '''

    signal = input_signal.copy()

    for i in range(1, n+1):

        # Display progress message
        msg = '\rComputing iteration {0:4d}/{1}'.format(i, n)
        sys.stdout.write(msg)
        sys.stdout.flush()

        # Compute the convolution
        convolved = fft_convolve(signal, ir_signal)

        # Mix the last computed signal back in.
        signal.resize(len(convolved))
        signal = signal + convolved

        # Normalize the new signal so it doesn't explode.
        signal = normalize(signal)
    print('')

    return signal


def main(input_filename, ir_filename, n, output_filename):
    ''' Main method - Runs 'sitting_in_a_room' for a given input and impulse
        response and writes it to a file. '''

    # Read in files and scale them
    print('\nReading {0}...'.format(input_filename))
    (r1, input_signal) = wav.read(input_filename)
    print('Reading {0}...'.format(ir_filename))
    (r2, ir_signal)    = wav.read(ir_filename)

    input_signal = scale_wav(input_signal)
    ir_signal    = scale_wav(ir_signal)

    # Enforce 1-channel signals (for now...)
    if len(np.shape(input_signal)) > 1:
        print('Error: {0} is not mono. Aborting.'.format(input_filename))
        return
    if len(np.shape(ir_signal)) > 1:
        print('Error: {0} is not mono. Aborting.'.format(ir_filename))
        return

    if r1 != r2:
        print('Warning: sampling rate of {0} and {1} differ.'.format(
            input_filename, ir_filename))
        print('         Using sampling rate of {0} ({1})'.format(
            input_filename, r1))
    sampling_rate = r1 # Use input signal's sampling rate

    print('Running "sitting in a room" with {0} passes...'.format(n))

    runtime = time.time()
    data = sitting_in_a_room(input_signal, ir_signal, n)
    runtime = time.time()-runtime

    write_to_wav(data, sampling_rate, output_filename)
    print('Result written to {0}'.format(output_filename))

    print('\nFinished in {:0.2f} seconds\n'.format(runtime))


def print_usage():
    print('\n'
          'Usage:\n'
          '    python sitting-in-a-room.py {input file} {ir file} {n}\n')


if __name__ == '__main__':

    try:
        input_filename = sys.argv[1]
        ir_filename    = sys.argv[2]
        n              = int(sys.argv[3])
    except:
        print_usage()
        sys.exit(2)

    output_filename = 'output/out-{0}-{1}-n{2:03d}.wav'.format(
            input_filename[:-4], ir_filename[:-4], n)

    if not os.path.exists('output'):
        os.makedirs('output')

    main('sound/'+input_filename, 'sound/'+ir_filename, n, output_filename)


