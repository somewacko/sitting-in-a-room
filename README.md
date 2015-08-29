# I Am Sitting In A Room...
----------------
Compute's a system inspired by [Alvin Lucier's "I Am Sitting In A Room"](https://youtu.be/2jU9mJbJsQ8) for any sound or room, for any number of iterations.

This program is different from Alvin's piece in that it allows each sound to fully play out before starting a new iteration.

### Usage:

	python sitting-in-a-room.py {input file} {IR file} {num iterations}

The two sounds must be in the `sound/` directory, and should be 1-channel, 16-bit .wav files, ideally with matching sample rates.
	
Outputs a file in `output/` with the name `out-{input}-{ir}-{n}.wav`

### Todo:

- Allow for stereo inputs and beyond.
- Have an option to produce output files at each iteration.
- Have an option to allow the sounds to overlap, more like Alvin's piece.

### Credits:

The example sound is from [freesound.org](http://freesound.org/) by user [digitopia](http://freesound.org/people/digitopia/sounds/76497/).