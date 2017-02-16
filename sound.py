import subprocess

sample_time = 1

while True:

    command = "/usr/bin/arecord -D plughw:0,0 -d " + str(sample_time) + " -f S16_LE | /usr/bin/sox -t .wav - -n stat"

    p = subprocess.Popen(command, bufsize=1, shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in p.stdout:
        if "Maximum amplitude" in line:
            print "Max:", line.split()[-1]

