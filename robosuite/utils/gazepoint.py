import socket
import re
import threading

class GazepointClient:
    def __init__(self, host='127.0.0.1', port=4242):  # obtained from gazepoint documentation
        self.host = host  
        self.port = port  
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.last_valid_x = 0
        self.last_valid_y = 0
        self._running = False
        self._lock = threading.Lock()  # Add thread lock for safe data access

    def connect(self):
        self.socket.connect((self.host, self.port))  
        self.send_commands()
        self._running = True
        # Start data reading thread
        threading.Thread(target=self._read_data, daemon=True).start()
        
    def send_commands(self):  
        commands = [
            '<SET ID="ENABLE_SEND_CURSOR" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'
        ]
        for cmd in commands:
            self.socket.send(cmd.encode())

    def _read_data(self):
        """Background thread for reading data"""
        while self._running:
            try:
                data = self.socket.recv(1024).decode()
                
                # Find first complete record
                start = data.find("<REC")
                if start == -1:
                    continue
                    
                end = data.find("/>", start)
                if end == -1:
                    continue
                    
                # Extract just the first record
                record = data[start:end+2]
                
                # Extract FPOGX and FPOGY using regex
                x_match = re.search(r'FPOGX="([-\d\.]+)"', record)
                y_match = re.search(r'FPOGY="([-\d\.]+)"', record)
                
                if x_match and y_match:
                    with self._lock:  # Thread-safe update
                        self.last_valid_x = float(x_match.group(1))
                        self.last_valid_y = float(y_match.group(1))
            except:
                if not self._running:
                    break
            
    def get_latest_gaze(self):
        """Thread-safe get of latest gaze position"""
        with self._lock:
            return self.last_valid_x, self.last_valid_y
        
    def close(self):
        self._running = False
        self.socket.close()