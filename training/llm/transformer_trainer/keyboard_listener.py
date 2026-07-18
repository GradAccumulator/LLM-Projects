import msvcrt
from threading import Event, Thread

class KeyboardListener:
    def __init__(self, key_map:dict, trainer):
        self._key_map = key_map
        self._trainer = trainer

        self._event = Event()
        self._thread = Thread(
            target=self._run,
            daemon=True,
        )
    
    def _on_press(self, key):
        command_type = self._key_map.get(key)
        if command_type is not None:
            self._trainer.submit_command(command_type())
    
    def _run(self):
        while not self._event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                
                if key in ("\x00", "\xe0"):
                    msvcrt.getwch()
                    continue

                self._on_press(key)
            self._event.wait(1/60)
        
    def start(self):
        return self._thread.start()
    
    def stop(self):
        self._event.set()
        return self._thread.join()
    