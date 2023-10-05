#!/usr/bin/env python
""" Alamot's OpenCL Fractals """
import sys
import argparse
import numpy as np
# pyopencl imports
import pyopencl as cl
from pyopencl import characterize
# Kivy imports
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics import Color, Line
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.logger import Logger
from kivy.app import App


# Maximum number of decimal digits
MAX_DECIMAL_DIGITS = 20

# Palettes (RGB color factors/weights)
PALETTES = [{"RF": np.ubyte(15), "GF": np.ubyte(5),  "BF": np.ubyte(1)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(5),  "BF": np.ubyte(15)},
            {"RF": np.ubyte(5),  "GF": np.ubyte(1),  "BF": np.ubyte(15)},
            {"RF": np.ubyte(15), "GF": np.ubyte(1),  "BF": np.ubyte(5)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(15), "BF": np.ubyte(5)},
            {"RF": np.ubyte(5),  "GF": np.ubyte(15), "BF": np.ubyte(1)},
            {"RF": np.ubyte(5),  "GF": np.ubyte(5),  "BF": np.ubyte(10)},
            {"RF": np.ubyte(5),  "GF": np.ubyte(10), "BF": np.ubyte(5)},
            {"RF": np.ubyte(10), "GF": np.ubyte(5),  "BF": np.ubyte(5)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(2),  "BF": np.ubyte(3)},
            {"RF": np.ubyte(3),  "GF": np.ubyte(2),  "BF": np.ubyte(1)},
            {"RF": np.ubyte(2),  "GF": np.ubyte(3),  "BF": np.ubyte(1)},
            {"RF": np.ubyte(2),  "GF": np.ubyte(1),  "BF": np.ubyte(3)},
            {"RF": np.ubyte(3),  "GF": np.ubyte(1),  "BF": np.ubyte(2)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(3),  "BF": np.ubyte(2)},
            {"RF": np.ubyte(50), "GF": np.ubyte(1),  "BF": np.ubyte(1)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(50), "BF": np.ubyte(1)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(1),  "BF": np.ubyte(50)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(50), "BF": np.ubyte(50)},
            {"RF": np.ubyte(50), "GF": np.ubyte(0),  "BF": np.ubyte(50)},
            {"RF": np.ubyte(50), "GF": np.ubyte(50), "BF": np.ubyte(1)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(1),  "BF": np.ubyte(1)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(0),  "BF": np.ubyte(0)},
            {"RF": np.ubyte(0),  "GF": np.ubyte(1),  "BF": np.ubyte(0)},
            {"RF": np.ubyte(0),  "GF": np.ubyte(0),  "BF": np.ubyte(1)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(0),  "BF": np.ubyte(1)},
            {"RF": np.ubyte(1),  "GF": np.ubyte(1),  "BF": np.ubyte(0)},
            {"RF": np.ubyte(0),  "GF": np.ubyte(1),  "BF": np.ubyte(1)}]


# OpenCL code for the Mandelbrot set
MANDELBROT_OPENCL_CODE = """
// Enable double precision floats
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot (const double x0, const double y0,
                          const double step,
                          const unsigned int width,
                          const unsigned int max_iters,
                          const unsigned char red_factor,
                          const unsigned char green_factor,
                          const unsigned char blue_factor,
                          __global unsigned int *restrict framebuf)
{
    double x = 0.0;
    double y = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    unsigned int iters = 0;
    const size_t idx = get_global_id(0);
    const size_t xpos = idx % width;
    const size_t ypos = (idx - xpos) / width;
    const double xstep = x0 + (xpos * step);
    const double ystep = y0 - (ypos * step);

    #pragma unroll
    while ((x2 + y2 < 4) && (iters < max_iters))
    {
      y = 2 * x * y + ystep;
      x = x2 - y2 + xstep;
      x2 = x * x;
      y2 = y * y;
      iters++;
    }

    if (iters >= max_iters)
    {
      framebuf[width * ypos + xpos] = 0xFF000000;
    }
    else
    {
      framebuf[width * ypos + xpos] = (0xFF000000 |
                                      (iters * blue_factor) << 16 |
                                      (iters * green_factor) << 8 |
                                      (iters * red_factor));
    }
}
"""


def get_decimal_scale(decimal_number):
    ''' Return the position of the first non-zero digit in mantissa '''
    if decimal_number == 0:
        return 0
    idx = 0
    mantissa = f'{decimal_number:.20f}'.split(".")[-1]
    for digit in mantissa:
        idx += 1
        if digit != "0":
            return idx
    return idx


class MyBoxLayout(BoxLayout):
    """ Class representing a vertical box layout that contains
        the image blocks that compose a fractal frame. """

    def __init__(self, n_blocks, **kwargs):
        super().__init__(**kwargs)
        self.rectx = 0
        self.recty = 0
        self.images = []
        # We distribute the calculation of fractal image in distinct
        # image blocks (number of blocks = number of OpenCL platforms)
        for _ in range(n_blocks):
            self.images.append(Image())
            self.add_widget(self.images[-1])
        # Area selection rectangle (for zoom, initially hidden)
        with self.canvas.after:
            Color(1, 1, 1)
            self.rect = Line(width=1, rectangle=(0, 0, 0, 0))


class FractalsApp(App):
    """ Kivy App Class """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Alamot's OpenCL Fractals"
        self.workers = []
        self.zoom_history = []
        self.palette_index = args.palette
        self.step_size = np.double(1 / args.zoom)
        self.max_iters = self.calculate_max_iterations(self.step_size)
        # Find devices that support coarsed-grain buffer SVM and create workers
        for queue in self.get_csvm_queues():
            self.workers.append({"queue": queue})
        self.width = np.uint32(args.width)
        # Adjust the height in case it isn't evenly divisible by num of workers
        remainder = args.height % len(self.workers)
        self.height = np.uint32(args.height - remainder)
        self.x_0 = np.double(args.x - self.step_size * self.width / 2)
        self.block_height = int(self.height / len(self.workers))
        self.screen = MyBoxLayout(len(self.workers), orientation='vertical',
                                  padding=(0, remainder, 0, 0))

    def get_csvm_queues(self):
        ''' Find devices in platforms that support Coarse-grained SVM
            (Shared Virtual Memory) and get Command Queues for them. '''
        csvm_queues = []
        for platform in cl.get_platforms():
            Logger.info("APP: Platform: %s (%s)",
                        platform.name, platform.version.strip())
            context = cl.Context(dev_type=cl.device_type.ALL,
                                 properties=[(cl.context_properties.PLATFORM,
                                              platform)])
            queue = cl.CommandQueue(context)
            context = queue.context
            device = queue.device
            Logger.info("APP:  -> Device: %s", device.name)
            # Is coarse-grained buffer SVM (Shared Virtual Memory) supported?
            csvm = characterize.has_coarse_grain_buffer_svm(device)
            Logger.info("APP:      * Coarse-grained buffer SVM: %s", csvm)
            if csvm:  # If yes, we save the queue
                csvm_queues.append(queue)
        num_queues = len(csvm_queues)
        Logger.info("APP: %d workers will be created.", num_queues)
        if num_queues == 0:
            Logger.error("APP: No device found to support coarse-grained SVM.")
            sys.exit(-1)
        return csvm_queues

    def build(self):
        # Bind keyboard and mouse events
        Window.bind(on_keyboard=self.hook_keyboard)
        Window.bind(on_touch_down=self.on_touch_down)
        Window.bind(on_touch_move=self.on_touch_move)
        Window.bind(on_touch_up=self.on_touch_up)

        # Initialization of workers
        for idx, worker in enumerate(self.workers):
            # Build the OpenCL program for each worker
            program = cl.Program(worker["queue"].context,
                                 MANDELBROT_OPENCL_CODE).build()
            # Allocate coarse-grained SVM buffer for each worker
            buffer_size = 4 * self.width * self.block_height
            # Size is multiplied by 4 because we have 4 bytes per pixel (RGBA)
            svm_buffer = cl.SVM(cl.csvm_empty(worker["queue"].context,
                                              buffer_size, np.ubyte))
            assert isinstance(svm_buffer.mem, np.ndarray)
            # The buffer shape is divided by 4 because we'll pass it as uint32
            svm_buffer_shape = tuple(e // 4 for e in svm_buffer.mem.shape)
            # Create a texture for each worker's image block
            texture = Texture.create(size=(self.width, self.block_height),
                                     colorfmt="rgba")
            self.screen.images[idx].texture = texture
            # Calculate y0 of fractal
            block_heights = (len(self.workers) - idx - 1) * self.block_height
            y_dim = (self.height / 2) - block_heights
            y_0 = np.double(args.y + self.step_size * y_dim)
            # Save worker parameters
            worker["program"] = program
            worker["svm_buffer"] = svm_buffer
            worker["svm_buffer_shape"] = svm_buffer_shape
            worker["texture"] = self.screen.images[idx].texture
            worker["y_0"] = y_0

        # Draw the fractal
        self.update_fractal()
        return self.screen

    def update_fractal(self):
        ''' Draw a fractal frame. '''
        for worker in self.workers:
            # Execute the OpenCL kernel for each worker
            worker["program"].mandelbrot(worker["queue"],
                                         worker["svm_buffer_shape"], None,
                                         self.x_0, worker["y_0"],
                                         self.step_size, self.width,
                                         self.max_iters,
                                         PALETTES[self.palette_index]["RF"],
                                         PALETTES[self.palette_index]["GF"],
                                         PALETTES[self.palette_index]["BF"],
                                         worker["svm_buffer"])
        for worker in self.workers:
            # Collect the results
            worker["queue"].finish()
            # We map the Coarse-grained buffer SVM (Shared Virtual Memory)
            with worker["svm_buffer"].map_rw(worker["queue"]) as svm_buffer:
                # We blit each worker result to the texture of its image block
                worker["texture"].blit_buffer(svm_buffer,
                                              colorfmt="rgba",
                                              bufferfmt='ubyte')
        self.screen.canvas.ask_update()

    def calculate_max_iterations(self, step_size):
        ''' Calculate the maximum iterations given the step size. '''
        decimal_scale = get_decimal_scale(step_size)
        if not decimal_scale:
            decimal_scale = MAX_DECIMAL_DIGITS / 2
        max_iters = np.uint32((2 ** decimal_scale) *
                              ((MAX_DECIMAL_DIGITS - decimal_scale) ** 2))
        Logger.info("APP: Calculated max iterations: %s", max_iters)
        return max_iters

    def zoom(self, direction):
        ''' Zoom in (direction = +1) or out (direction = -1) 1/10 of scale. '''
        center_x = self.x_0 + self.step_size * self.width / 2
        center_y = self.workers[-1]["y_0"] - self.step_size * self.height / 2
        self.calculate_max_iterations(self.step_size)
        decimal_scale = get_decimal_scale(self.step_size)
        self.step_size -= direction * (10 ** -(decimal_scale + 1))
        self.x_0 = np.double(center_x - self.step_size * self.width / 2)
        for idx, worker in enumerate(self.workers):
            block_heights = (len(self.workers) - idx - 1) * self.block_height
            y_dim = (self.height / 2) - block_heights
            worker["y_0"] = np.double(center_y + self.step_size * y_dim)

    def zoom_into_area(self, x_1, y_1, x_2, y_2):
        ''' Zoom into the designated area '''
        # Calculate the parameters for the new zoom area
        center_x = self.x_0 + self.step_size * (x_1 + x_2) / 2
        center_y = self.workers[-1]["y_0"] - self.step_size * (y_1 + y_2) / 2
        self.step_size = (abs(x_2 - x_1) * self.step_size) / self.width
        self.max_iters = self.calculate_max_iterations(self.step_size)
        self.x_0 = center_x - self.step_size * self.width / 2
        for idx, worker in enumerate(self.workers):
            block_heights = (len(self.workers) - idx - 1) * self.block_height
            y_dim = (self.height / 2) - block_heights
            worker["y_0"] = np.double(center_y + self.step_size * y_dim)

    def goto_previous_zoom(self):
        ''' Go back in zoom history one step. '''
        if self.zoom_history:
            previous_zoom = self.zoom_history.pop()
            self.step_size = previous_zoom["step_size"]
            self.max_iters = previous_zoom["max_iters"]
            self.x_0 = previous_zoom["x_0"]
            for idx, worker in enumerate(self.workers):
                worker["y_0"] = previous_zoom["y_0"][idx]

    def next_palette(self):
        ''' Use the next color palette. '''
        self.palette_index += 1
        if self.palette_index >= len(PALETTES):
            self.palette_index = 0

    def previous_palette(self):
        ''' Use the previous color palette. '''
        self.palette_index -= 1
        if self.palette_index < 0:
            self.palette_index = len(PALETTES) - 1

    def on_touch_down(self, _window, touch):
        ''' Mouse button press handler '''
        if touch.button != "left":  # Is left mouse button pressed?
            return True  # We processed the event (do not propagate it).
        self.screen.rectx = touch.x
        self.screen.recty = touch.y
        self.screen.rect.rectangle = (touch.x, touch.y, 1, 1)
        return True      # We processed the event. Do not propagate it.

    def on_touch_move(self, _window, touch):
        ''' Mouse pointer move handler '''
        if touch.button != "left":  # Is left mouse button pressed?
            return True  # We processed the event (do not propagate it).
        # Update selection rectangle position
        self.screen.rect.rectangle = (self.screen.rectx,
                                      self.screen.recty,
                                      touch.x - self.screen.rectx,
                                      touch.y - self.screen.recty)
        return True      # We processed the event (do not propagate it).

    def on_touch_up(self, _window, touch):
        ''' Mouse button release handler '''
        if touch.button == "right":     # Right click (Go to previous zoom)
            self.goto_previous_zoom()
        elif touch.button == "middle":  # Middle click (Next palette)
            self.next_palette()
        elif touch.button == 'left':    # Left click (Zoom into selected area)
            # Hide selection rectangle
            self.screen.rect.rectangle = (0, 0, 0, 0)
            # Save previous zoom parameters
            self.zoom_history.append({"step_size": self.step_size,
                                      "max_iters": self.max_iters,
                                      "x_0": self.x_0,
                                      "y_0": [w["y_0"] for w in self.workers]
                                      })
            # Zoom into the selected area
            self.zoom_into_area(self.screen.rectx, self.screen.recty,
                                touch.x, touch.y)
        # Draw the new frame
        self.update_fractal()
        return True      # We processed the event (do not propagate it).

    def hook_keyboard(self, _window, key, *_):
        ''' Keyboard handler '''
        if key in (113, 27):         # q or esc (Exit)
            sys.exit(0)
        elif key == 286:             # F5 (Increase max iters by 1000)
            self.max_iters += np.uint32(1000)
            Logger.info("APP: Max iterations increased to: %s", self.max_iters)
        elif key == 8:               # Backspace (Go to previous zoom)
            self.goto_previous_zoom()
        elif key in (99, 265, 280):  # c or PageUp (Next palette)
            self.next_palette()
        elif key in (259, 281):      # PageDown (Previous palette)
            self.previous_palette()
        elif key in (97, 276):       # a or left_arrow (Move left)
            decimal_scale = get_decimal_scale(self.step_size)
            self.x_0 += 10 ** -(decimal_scale - 1)
        elif key in (100, 275):      # d or right_arrow (Move right)
            decimal_scale = get_decimal_scale(self.step_size)
            self.x_0 -= 10 ** -(decimal_scale - 1)
        elif key in (119, 273):      # w or up_arrow (Move up)
            decimal_scale = get_decimal_scale(self.step_size)
            for worker in self.workers:
                worker["y_0"] += 10 ** -(decimal_scale - 1)
        elif key in (115, 274):      # s or down_arrow (Move down)
            decimal_scale = get_decimal_scale(self.step_size)
            for worker in self.workers:
                worker["y_0"] -= 10 ** -(decimal_scale - 1)
        elif key in (45, 269):       # - (Zoom out)
            self.zoom(-1)
        elif key in (61, 270):       # + (Zoom in)
            self.zoom(+1)
        self.update_fractal()
        return True      # We processed the event (do not propagate it).


if __name__ == "__main__":
    assert cl.get_cl_header_version()[0] >= 2, "OpenCL >= 2.0 is required."
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--fullscreen", default='yes',
                           help="Fullscreen mode",
                           choices=('yes', 'no'))
    argParser.add_argument("-p", "--palette", type=int, default=0,
                           help="Palette index",
                           choices=range(len(PALETTES)), metavar="(0-27)")
    argParser.add_argument("-w", "--width", type=int, default=2194,
                           help="The width of the fractal frame")
    argParser.add_argument("-v", "--height", type=int, default=1234,
                           help="The height of the fractal frame")
    argParser.add_argument("-x", type=float, default=0.0,
                           help="The x position in fractal")
    argParser.add_argument("-y", type=float, default=0.0,
                           help="The y position in fractal")
    argParser.add_argument("-z", "--zoom", type=int, default=100,
                           help="The zoom factor")
    args = argParser.parse_args()

    # Kivy setings
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    Window.size = (args.width, args.height)
    if args.fullscreen == 'yes':
        Window.fullscreen = 'auto'

    FractalsApp().run()
