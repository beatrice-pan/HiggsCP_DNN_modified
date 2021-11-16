import numpy as np


def find_first_line(lines, phrase):
    for i, line in enumerate(lines):
        if phrase in line:
            return i


def read_raw_root(name, num_particles):
    with open(name) as f:
        lines = f.readlines()
    # Filter out unnecessary lines.
    # The interesting lines start with the first "TUPLE" and end at "Analysed in total".
    lines = lines[find_first_line(lines, "TUPLE"):find_first_line(lines, "Analysed in total:")]
    # Ignore debug lines.
    lines = [line for line in lines if not line.startswith("Analysed:")]
    # Find ids of lines that start descriptions of examples.
    ids = [int(idx) for idx, line in enumerate(lines) if line.startswith("TUPLE")]

    # Ensure that there are `num_particles` particles for each example.
    assert ids == range(0, num_particles * len(ids), num_particles)
    print(len(lines), len(ids)*num_particles)
    # If the numbers are not equal, check the last lines of pythia file 
    assert len(lines) == num_particles * len(ids)
    lines = [line.strip() for line in lines]

    num_examples = len(ids)

    weights = [float(lines[num_particles * i].strip().split()[1]) for i in range(num_examples)]
    weights = np.array(weights)

    values = [map(float, " ".join(lines[num_particles * i + 1: num_particles * (i + 1)]).split())
              for i in range(num_examples)]
    values = np.array(values)

    return values, weights