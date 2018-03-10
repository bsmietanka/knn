
import time

measurements = {}

def performance(method):

    global measurements

    def measure(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        if (measurements.get(method.__name__) == None):
            measurements[method.__name__] = [0., 0]
        measurements[method.__name__][0] += end - start
        measurements[method.__name__][1] += 1
        return result

    return measure

def dump():
    global measurements
    for name, results in measurements.items():
        print("{0:20} : avg {1:.5f};\ttotal {2:.5f};\thits {3:d}".format(name, float(results[0] / results[1]), float(results[0]), int(results[1])))
    measurements = {}
