"""
Dataset Generator Web Monitoring Server
Similar to VSR++ training GUI
"""

import threading
import json
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import os


def detect_local_ip():
    """Detect local IP address for access"""
    try:
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        local_address = temp_socket.getsockname()[0]
        temp_socket.close()
        return local_address
    except:
        return "localhost"


class DatasetDataStore:
    """Thread-safe storage for dataset generation progress"""
    
    def __init__(self):
        self._data_lock = threading.Lock()
        self._full_state = self._create_empty_state()
        
    def _create_empty_state(self):
        """Create empty state with all fields"""
        return {
            # Current video info
            'current_video_name': '',
            'current_video_index': 0,
            'total_videos': 0,
            'current_video_duration': 0,
            
            # Current video progress (per category)
            'current_video_progress': {
                'master': {'created': 0, 'target': 0, 'percent': 0.0},
                'space': {'created': 0, 'target': 0, 'percent': 0.0},
                'toon': {'created': 0, 'target': 0, 'percent': 0.0},
                'universal': {'created': 0, 'target': 0, 'percent': 0.0},
            },
            
            # Overall progress (per category)
            'overall_progress': {
                'master': {'created': 0, 'target': 150000, 'percent': 0.0},
                'space': {'created': 0, 'target': 60000, 'percent': 0.0},
                'toon': {'created': 0, 'target': 50000, 'percent': 0.0},
                'universal': {'created': 0, 'target': 50000, 'percent': 0.0},
            },
            
            # Patch distribution by size (final values)
            'patch_distribution': {
                'master': {
                    '540': {'count': 0, 'target': 0},
                    '1080': {'count': 0, 'target': 0},
                    '2160': {'count': 0, 'target': 0},
                },
                'space': {
                    '540': {'count': 0, 'target': 0},
                    '1080': {'count': 0, 'target': 0},
                    '2160': {'count': 0, 'target': 0},
                },
                'toon': {
                    '540': {'count': 0, 'target': 0},
                    '1080': {'count': 0, 'target': 0},
                    '2160': {'count': 0, 'target': 0},
                },
                'universal': {
                    '540': {'count': 0, 'target': 0},
                    '1080': {'count': 0, 'target': 0},
                    '2160': {'count': 0, 'target': 0},
                },
            },
            
            # ETA estimates
            'eta': {
                'master': 'N/A',
                'space': 'N/A',
                'toon': 'N/A',
                'universal': 'N/A',
                'total': 'N/A',
            },
            
            # Performance metrics
            'scenes_processed': 0,
            'patches_created_total': 0,
            'avg_time_per_scene': 0.0,
            'start_time': time.time(),
        }
    
    def update(self, updates):
        """Update state with new values"""
        with self._data_lock:
            self._recursive_update(self._full_state, updates)
    
    def _recursive_update(self, target, source):
        """Recursively update nested dicts"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._recursive_update(target[key], value)
            else:
                target[key] = value
    
    def get_complete_snapshot(self):
        """Get complete state snapshot"""
        with self._data_lock:
            return json.loads(json.dumps(self._full_state))


class MonitoringRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for monitoring"""
    
    data_repository = None
    refresh_interval_sec = 2
    
    def log_message(self, format, *args):
        """Suppress standard logging"""
        pass
    
    def do_GET(self):
        """GET request handler"""
        if self.path == '/monitoring/data':
            self._deliver_json_snapshot()
        elif self.path.startswith('/monitoring'):
            self._deliver_main_page()
        else:
            self.send_error(404)
    
    def _deliver_json_snapshot(self):
        """Deliver complete data snapshot as JSON"""
        full_data = self.data_repository.get_complete_snapshot()
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        json_output = json.dumps(full_data, indent=2)
        self.wfile.write(json_output.encode('utf-8'))
    
    def _deliver_main_page(self):
        """Deliver main monitoring HTML page"""
        template_path = os.path.join(
            os.path.dirname(__file__),
            'templates', 'monitor.html'
        )
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        except FileNotFoundError:
            self.send_error(404, "Template not found")


class DatasetMonitoringServer:
    """Web server for dataset generation monitoring"""
    
    def __init__(self, data_store, port=8765):
        self.data_store = data_store
        self.port = port
        self.server = None
        self.server_thread = None
        self.local_ip = detect_local_ip()
        
    def start(self):
        """Start the monitoring server in a background thread"""
        MonitoringRequestHandler.data_repository = self.data_store
        
        self.server = HTTPServer(('0.0.0.0', self.port), MonitoringRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        print(f"\n{'='*70}")
        print(f"📊 Dataset Generator Monitoring UI gestartet!")
        print(f"   Lokaler Zugriff:  http://localhost:{self.port}/monitoring")
        print(f"   Netzwerk-Zugriff: http://{self.local_ip}:{self.port}/monitoring")
        print(f"{'='*70}\n")
    
    def stop(self):
        """Stop the monitoring server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
