"""
Web-based Training Monitor - Custom lightweight HTTP server for VSR++ monitoring
Uses threading and simple HTTP handlers for minimal overhead
"""

import threading
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from queue import Queue
import socket


class TrainingStateHolder:
    """Thread-safe container for current training state"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._current_data = {
            'iteration': 0,
            'total_loss': 0.0,
            'learn_rate': 0.0,
            'time_remaining': 'N/A',
            'iter_speed': 0.0,
            'gpu_memory': 0.0,
            'best_score': 0.0,
            'is_validating': False,
            'is_training': True,
            'timestamp': time.time()
        }
        
    def refresh_metrics(self, **kwargs):
        """Update stored metrics atomically"""
        with self._lock:
            self._current_data.update(kwargs)
            self._current_data['timestamp'] = time.time()
    
    def fetch_snapshot(self):
        """Get current state snapshot"""
        with self._lock:
            return self._current_data.copy()


class MonitorRequestHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler for monitoring endpoints"""
    
    # Class variables shared across instances
    state_holder = None
    command_queue = None
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/status':
            self._serve_status_json()
        elif self.path == '/' or self.path == '/index.html':
            self._serve_dashboard_html()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/action':
            self._process_action_command()
        else:
            self.send_error(404)
    
    def _serve_status_json(self):
        """Return current metrics as JSON"""
        snapshot = self.state_holder.fetch_snapshot()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        self.wfile.write(json.dumps(snapshot).encode())
    
    def _process_action_command(self):
        """Process incoming command requests"""
        content_len = int(self.headers.get('Content-Length', 0))
        body_data = self.rfile.read(content_len)
        
        try:
            cmd_data = json.loads(body_data.decode())
            action_type = cmd_data.get('command', '')
            
            if action_type == 'validate':
                self.command_queue.put('validate')
                response = {'status': 'queued', 'command': 'validate'}
            else:
                response = {'status': 'unknown', 'command': action_type}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(400, str(e))
    
    def _serve_dashboard_html(self):
        """Serve monitoring dashboard page"""
        html_content = self._generate_dashboard_markup()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def _generate_dashboard_markup(self):
        """Generate HTML dashboard with embedded styles and scripts"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>VSR++ Training Monitor</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            color: #00d4ff;
            margin-bottom: 30px;
            text-align: center;
            font-size: 2.5em;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        .status-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 15px;
            font-size: 0.5em;
            vertical-align: middle;
        }
        .badge-training { background: #00c853; color: white; }
        .badge-validating { background: #ff9800; color: white; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .metric-label {
            color: #888;
            font-size: 0.9em;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #00d4ff;
        }
        .metric-unit {
            font-size: 0.5em;
            color: #888;
            margin-left: 5px;
        }
        .controls {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .btn {
            background: linear-gradient(135deg, #00c853 0%, #00a040 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 200, 83, 0.6);
        }
        .btn:active {
            transform: translateY(0);
        }
        .timestamp {
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            VSR++ Training Monitor
            <span id="statusBadge" class="status-badge badge-training">Training</span>
        </h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Iteration</div>
                <div class="metric-value" id="iteration">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Loss</div>
                <div class="metric-value" id="totalLoss">0.0000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Learning Rate</div>
                <div class="metric-value" id="learnRate">0.0<span class="metric-unit">×10⁻⁴</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ETA</div>
                <div class="metric-value" id="timeRemaining">--:--:--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Speed</div>
                <div class="metric-value" id="iterSpeed">0.0<span class="metric-unit">it/s</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Memory</div>
                <div class="metric-value" id="gpuMemory">0.0<span class="metric-unit">GB</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Best Quality</div>
                <div class="metric-value" id="bestScore">0.0<span class="metric-unit">%</span></div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="triggerValidation()">🔍 Trigger Validation</button>
        </div>
        
        <div class="timestamp">
            Last updated: <span id="lastUpdate">--</span>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('iteration').textContent = 
                        data.iteration.toLocaleString();
                    document.getElementById('totalLoss').textContent = 
                        data.total_loss.toFixed(4);
                    document.getElementById('learnRate').innerHTML = 
                        (data.learn_rate * 10000).toFixed(2) + '<span class="metric-unit">×10⁻⁴</span>';
                    document.getElementById('timeRemaining').textContent = 
                        data.time_remaining;
                    document.getElementById('iterSpeed').innerHTML = 
                        (1 / data.iter_speed).toFixed(2) + '<span class="metric-unit">it/s</span>';
                    document.getElementById('gpuMemory').innerHTML = 
                        data.gpu_memory.toFixed(2) + '<span class="metric-unit">GB</span>';
                    document.getElementById('bestScore').innerHTML = 
                        (data.best_score * 100).toFixed(1) + '<span class="metric-unit">%</span>';
                    
                    const badge = document.getElementById('statusBadge');
                    if (data.is_validating) {
                        badge.textContent = 'Validating';
                        badge.className = 'status-badge badge-validating';
                    } else if (data.is_training) {
                        badge.textContent = 'Training';
                        badge.className = 'status-badge badge-training';
                    }
                    
                    const updateTime = new Date(data.timestamp * 1000);
                    document.getElementById('lastUpdate').textContent = 
                        updateTime.toLocaleTimeString();
                })
                .catch(err => console.error('Update failed:', err));
        }
        
        function triggerValidation() {
            fetch('/api/action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: 'validate'})
            })
            .then(response => response.json())
            .then(data => {
                alert('Validation queued successfully!');
            })
            .catch(err => {
                alert('Failed to trigger validation: ' + err);
            });
        }
        
        // Auto-refresh every 1 second
        setInterval(updateDashboard, 1000);
        updateDashboard(); // Initial load
    </script>
</body>
</html>'''


class WebInterface:
    """Main web interface controller for training monitoring"""
    
    def __init__(self, port_number=5050):
        self.port_num = port_number
        self.metrics_holder = TrainingStateHolder()
        self.cmd_queue = Queue()
        self.http_server = None
        self.server_thread = None
        
        # Configure handler with shared state
        MonitorRequestHandler.state_holder = self.metrics_holder
        MonitorRequestHandler.command_queue = self.cmd_queue
        
        self._initialize_server()
    
    def _initialize_server(self):
        """Start HTTP server in daemon thread"""
        try:
            self.http_server = HTTPServer(('0.0.0.0', self.port_num), MonitorRequestHandler)
            
            self.server_thread = threading.Thread(
                target=self.http_server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            print(f"🌐 Web monitor active: http://localhost:{self.port_num}")
            
        except OSError as e:
            if e.errno == 48 or e.errno == 98:  # Address already in use
                print(f"⚠️  Port {self.port_num} busy, web monitor disabled")
            else:
                raise
    
    def update(self, iteration, total_loss, learn_rate, time_remaining, 
               iter_speed, gpu_memory, best_score=0.0, is_validating=False):
        """Update training metrics for web display"""
        self.metrics_holder.refresh_metrics(
            iteration=iteration,
            total_loss=total_loss,
            learn_rate=learn_rate,
            time_remaining=time_remaining,
            iter_speed=iter_speed,
            gpu_memory=gpu_memory,
            best_score=best_score,
            is_validating=is_validating,
            is_training=not is_validating
        )
    
    def check_commands(self):
        """Check for pending commands from web UI"""
        if not self.cmd_queue.empty():
            return self.cmd_queue.get()
        return None
    
    def shutdown(self):
        """Stop the web server"""
        if self.http_server:
            self.http_server.shutdown()
