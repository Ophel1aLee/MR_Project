import time

class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.history = []

    def start(self):
        self.start_time = time.time() * 1000

    def stop(self):
        self.end_time = time.time() * 1000

    def record_time(self):
        duration = self.end_time - self.start_time
        self.history.append(duration)
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None