# I Am Sitting In A Room...

Runs a process inspired by [Alvin Lucier's "I Am Sitting In A Room"](https://www.youtube.com/watch?v=fAxHlLK3Oyk) for a sound in a room defined by its impulse response.

Essentially, it just applies a convolution reverb over and over again, reinforcing the room's resonant frequencies and smoothing out irregularities present in the original sound.

[My voice after being processed for 20 iterations.](https://drive.google.com/file/d/0B_jWp8d11aB5ckthQ0ZRNWR6NDA/view?usp=sharing)

[After 100 iterations...](https://drive.google.com/file/d/0B_jWp8d11aB5Y0lZYWthUUtZYnc/view?usp=sharing)

### Usage:

	python sitting-in-a-room.py {input filename} {IR filename} {options}
	        -n, -num_passes {int} : The number of iterations to compute. (Default 10)
	        -l, -level {float} : The amount of "room" you want for each pass. (Default 0.5, or -6dB)
	        -full : Output all of the iterations side-by-side, much like the real thing.

The two sounds must be in the `sound/` directory, and should be 1-channel, 16-bit .wav files, ideally with matching sample rates.

Requires [NumPy](http://www.numpy.org/) and [SciPy](https://www.scipy.org/).

### Try it out for yourself!

- Get a microphone and some recording software.
- Make a recording of yourself speaking, or making any noise. This is your input sound.
- Place your microphone on the other side of the room and make a recoding of you clapping or hitting something very hard. This is your impulse response.
- Place your two files in the `sound/` folder.
- Run the script with your sounds!

Or just experiment with any two sounds you think would be interesting, of course.

### Credits:

The example sound provided is from [freesound.org](http://freesound.org/) by user [digitopia](http://freesound.org/people/digitopia/sounds/76497/).
