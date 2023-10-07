# Alamot's OpenCL Fractals

![pylint score](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/pylint.svg)


## Description

This is a python application that computes and renders Fractals (for the time being only the Mandelbrot set) using OpenCL in order to parallelize and distribute the work across different platforms / devices.

OpenCL (Open Computing Language) is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and others. 

The application divides each fractal frame into blocks that are processed separately by each OpenCL device. The results from each are combined to produce the final image. It makes use of coarse-grained SVM (Shared Virtual Memory) buffers to avoid copying large amount of data, back and forth.


## Requirements

- At least one Platform / device (and its OpenCL driver) that support OpenCL >= 2.0 and coarse-grained SVM 
- Kivy
- NumPy
- PyOpenCL (with OpenCL >= 2.0 support)


## Known limitations

- No infinite zoom: The application make uses of hardware double-precision decimal numbers that have a limit in how far one can zoom in. Software infinite-precision numbers and tricks like the pertubation method have not been implemented (yet).


## Todo

- [ ] Infinite zoom


## Install 

```
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```


## Run

Because Kivy hijacks the command-line arguments, please prefer to use:

```
$ ./run.sh
```

Otherwise, set KIVY_NO_ARGS=1 before running alamot_opencl_fractals.py:
``` 
$ KIVY_NO_ARGS=1 ./alamot_opencl_fractals.py 
```

Command-line arguments:
```
usage: ./run.sh [-h] [-f {yes,no}] [-p (0-27)] [-w WIDTH] [-v HEIGHT] [-x X] [-y Y] [-z ZOOM]

options:
  -h, --help            show this help message and exit
  -f {yes,no}, --fullscreen {yes,no}
                        Fullscreen mode
  -p (0-27), --palette (0-27)
                        Palette index
  -w WIDTH, --width WIDTH
                        The width of the fractal frame
  -v HEIGHT, --height HEIGHT
                        The height of the fractal frame
  -x X                  The x position in fractal
  -y Y                  The y position in fractal
  -z ZOOM, --zoom ZOOM  The zoom factor
```


## Mouse and keyboard usage

### Mouse 
- **Left click**: Select area for zoom in
- **Middle click**: Use next color palette
- **Right click**: Go back to the previous zoom (i.e. zoom out)

### Keyboard
- **WASD or arrows keys**: Move fractal
- **Plus (+) / Minus (-) keys**: Zoom in / out
- **PageUp or c**: Use next color palette
- **PageDn**: Use previous color palette
- **Backspace**: Go back to the previous zoom (i.e. zoom out) 
- *F5**: Increase maximum iterations by 1000 (may improve current frame rendering)
- **Esc or q**: Exit


## Output samples

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/03Oct2023_161526.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/04Oct2023_234628.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_005743.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_012333.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_151040.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_013526.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_151506.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/04Oct2023_100907.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/04Oct2023_235654.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_011818.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_012428.jpg)

![Fractal Image](https://raw.githubusercontent.com/Alamot/alamot_opencl_fractals/master/images/05Oct2023_191423.jpg)
