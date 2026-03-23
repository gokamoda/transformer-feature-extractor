from multiprocessing import Manager, Pool
from time import sleep


class PoolWithTqdm:
    def __init__(self, num_processes, args):
        self.num_processes = num_processes
        self._manager = Manager()
        self.pool = Pool(processes=num_processes)
        self.args_with_queue = args

    def __enter__(self):
        self.position_manager = self._manager.Queue()
        for pos in range(1, self.num_processes + 1):
            self.position_manager.put(pos)
        self.args_with_queue = [
            arg + (self.position_manager,) for arg in self.args_with_queue
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()
        self.pool.join()
        self._manager.shutdown()


class PoolWithTqdmSingle:
    def __init__(self, args):
        self._queue = args[-1]
        self.args = args[:-1]
        self.position = None

    def __enter__(self):
        self.position = self._queue.get()
        sleep(0.01 * self.position)  # Simulate some initial setup delay
        return self.args, self.position

    def __exit__(self, exc_type, exc_value, traceback):
        self._queue.put(self.position)
