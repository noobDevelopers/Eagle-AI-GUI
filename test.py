import joystick as jk
import numpy as np
import time

class test(jk.Joystick):
    # initialize the infinite loop decorator
    _infinite_loop = jk.deco_infinite_loop()

    def _init(self, *args, **kwargs):
        """
        Function called at initialization, see the doc
        """
        self._t0 = time.time()  # initialize time
        self.xdata = np.array([self._t0])  # time x-axis
        self.ydata = np.array([0.0])  # fake data y-axis
        # create a graph frame
        self.mygraph = self.add_frame(jk.Graph(name="test", size=(500, 500), pos=(50, 50), fmt="go-", xnpts=10000, xnptsmax=10000, xylim=(None, None, 0, 1)))

    @_infinite_loop(wait_time=0.2)
    def _generate_data(self):  # function looped every 0.2 second to read or produce data
        """
        Loop starting with the simulation start, getting data and
    pushing it to the graph every 0.2 seconds
        """
        # concatenate data on the time x-axis
        self.xdata = jk.core.add_datapoint(self.xdata, time.time(), xnptsmax=self.mygraph.xnptsmax)
        # concatenate data on the fake data y-axis
        self.ydata = jk.core.add_datapoint(self.ydata, np.random.random(), xnptsmax=self.mygraph.xnptsmax)
        self.mygraph.set_xydata(t, self.ydata)

t = test()
t.start()
t.stop()