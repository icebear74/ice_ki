"""
VSR++ Vollständiges Web-Monitoring-System
Zeigt ALLE Terminal-GUI-Daten im Browser mit einzigartiger Architektur
"""

import threading
import json
import time
import errno
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue


def detect_local_ip():
    """Ermittelt die lokale IP-Adresse für TensorBoard-Links"""
    try:
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        local_address = temp_socket.getsockname()[0]
        temp_socket.close()
        return local_address
    except:
        return "localhost"


class CompleteTrainingDataStore:
    """Speichert ALLE Trainingsdaten thread-sicher"""
    
    def __init__(self):
        self._data_lock = threading.Lock()
        self._full_state = self._create_empty_state()
        
    def _create_empty_state(self):
        """Erstellt leeren State mit ALLEN Feldern"""
        return {
            # Grundlegende Metriken
            'step_current': 0,
            'epoch_num': 1,
            'step_max': 100000,
            'epoch_step_current': 0,
            'epoch_step_total': 1000,
            
            # Verluste (alle Komponenten)
            'total_loss_value': 0.0,
            'l1_loss_value': 0.0,
            'ms_loss_value': 0.0,
            'gradient_loss_value': 0.0,
            'perceptual_loss_value': 0.0,
            
            # Adaptive Gewichte
            'l1_weight_current': 1.0,
            'ms_weight_current': 1.0,
            'gradient_weight_current': 1.0,
            'perceptual_weight_current': 0.0,
            'gradient_clip_val': 1.0,
            
            # Adaptive Status (NEW)
            'adaptive_mode': 'Stable',
            'adaptive_is_cooldown': False,
            'adaptive_cooldown_remaining': 0,
            
            # Lernrate
            'learning_rate_value': 0.0,
            'lr_phase_name': 'warmup',
            
            # Performance
            'iteration_duration': 0.0,
            'vram_usage_gb': 0.0,
            'adam_momentum_avg': 0.0,
            
            # Zeitschätzungen
            'eta_total_formatted': 'N/A',
            'eta_epoch_formatted': 'N/A',
            
            # Quality-Metriken (ALLE)
            'quality_lr_value': 0.0,
            'quality_ki_value': 0.0,
            'quality_improvement_value': 0.0,
            'quality_ki_to_gt_value': 0.0,
            'quality_lr_to_gt_value': 0.0,
            'validation_loss_value': 0.0,
            'best_quality_ever': 0.0,
            
            # Layer-Aktivitäten (dict: layer_name -> percentage)
            'layer_activity_map': {},
            
            # Statusflags
            'training_active': True,
            'validation_running': False,
            'training_paused': False,
            
            # Netzwerk
            'local_ip_address': detect_local_ip(),
            'tensorboard_port': 6006,
            
            # Zeitstempel
            'last_update_time': time.time()
        }
    
    def update_all_metrics(self, **updates):
        """Aktualisiert beliebige Metriken atomar"""
        with self._data_lock:
            self._full_state.update(updates)
            self._full_state['last_update_time'] = time.time()
    
    def get_complete_snapshot(self):
        """Liefert vollständige Kopie aller Daten"""
        with self._data_lock:
            return self._full_state.copy()


class WebMonitorRequestProcessor(BaseHTTPRequestHandler):
    """Verarbeitet HTTP-Anfragen für Monitoring"""
    
    data_repository = None
    action_queue = None
    refresh_interval_sec = 5
    
    def log_message(self, format, *args):
        """Unterdrückt Standard-Logging"""
        pass
    
    def do_GET(self):
        """GET-Request-Handler"""
        if self.path == '/monitoring/data':
            self._deliver_json_snapshot()
        elif self.path == '/monitoring/config':
            self._deliver_config_json()
        elif self.path.startswith('/monitoring'):
            self._deliver_main_page()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """POST-Request-Handler"""
        if self.path == '/monitoring/command':
            self._process_user_command()
        else:
            self.send_error(404)
    
    def _deliver_json_snapshot(self):
        """Liefert kompletten Datensnapshot als JSON"""
        full_data = self.data_repository.get_complete_snapshot()
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        json_output = json.dumps(full_data, indent=2)
        self.wfile.write(json_output.encode('utf-8'))
    
    def _deliver_config_json(self):
        """Liefert Konfiguration (z.B. Aktualisierungsintervall)"""
        config = {
            'refresh_interval_seconds': self.refresh_interval_sec,
            'auto_refresh_enabled': True
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        self.wfile.write(json.dumps(config).encode('utf-8'))
    
    def _process_user_command(self):
        """Verarbeitet Befehle vom Benutzer"""
        content_length = int(self.headers.get('Content-Length', 0))
        request_body = self.rfile.read(content_length)
        
        try:
            command_data = json.loads(request_body.decode('utf-8'))
            action_type = command_data.get('action', '')
            
            if action_type == 'trigger_validation':
                self.action_queue.put('validate')
                response = {'success': True, 'message': 'Validation queued'}
            elif action_type == 'change_refresh':
                new_interval = command_data.get('interval', 5)
                self.__class__.refresh_interval_sec = max(1, min(60, new_interval))
                response = {'success': True, 'interval': self.__class__.refresh_interval_sec}
            else:
                response = {'success': False, 'message': f'Unknown action: {action_type}'}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error(400, str(e))
    
    def _deliver_main_page(self):
        """Liefert Haupt-HTML-Seite mit eingebettetem JavaScript"""
        html_page = self._build_complete_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_page.encode('utf-8'))
    
    def _build_complete_dashboard_html(self):
        """Baut vollständige Dashboard-HTML mit ALLEN Daten"""
        return '''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VSR++ Training Monitor - Vollansicht</title>
    <style>
        :root {
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-orange: #d29922;
            --accent-purple: #bc8cff;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            padding: 20px;
            line-height: 1.6;
        }
        
        .main-container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header-section {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        h1 {
            font-size: 2.5em;
            color: var(--accent-blue);
            margin-bottom: 10px;
        }
        
        .status-indicator {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 5px;
        }
        
        .status-training { background: var(--accent-green); color: #000; }
        .status-validating { background: var(--accent-orange); color: #000; }
        .status-paused { background: var(--accent-red); color: #fff; }
        
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .info-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
        }
        
        .card-title {
            font-size: 0.85em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }
        
        .card-value {
            font-size: 2em;
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        .card-subtitle {
            font-size: 0.9em;
            color: var(--text-secondary);
            margin-top: 5px;
        }
        
        .section-header {
            font-size: 1.4em;
            color: var(--accent-purple);
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }
        
        .layer-activity-container {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .layer-row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            gap: 15px;
        }
        
        .layer-name {
            min-width: 150px;
            font-size: 0.9em;
            color: var(--text-primary);
        }
        
        .layer-bar-container {
            flex: 1;
            height: 24px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        
        .layer-bar-fill {
            height: 100%;
            transition: width 0.3s ease;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        }
        
        .layer-bar-fill.fusion {
            background: linear-gradient(90deg, var(--accent-orange), var(--accent-red));
        }
        
        .layer-bar-fill.final-fusion {
            background: linear-gradient(90deg, var(--accent-green), #00ff88);
        }
        
        .layer-value {
            min-width: 60px;
            text-align: right;
            font-size: 0.9em;
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        .controls-section {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: var(--accent-blue);
            color: #000;
        }
        
        .btn-primary:hover {
            background: #79c0ff;
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: var(--accent-green);
            color: #000;
        }
        
        .btn-success:hover {
            background: #56d364;
            transform: translateY(-2px);
        }
        
        .link-box {
            display: inline-block;
            padding: 12px 20px;
            background: rgba(88, 166, 255, 0.1);
            border: 1px solid var(--accent-blue);
            border-radius: 6px;
            color: var(--accent-blue);
            text-decoration: none;
            margin: 10px 10px 10px 0;
            transition: all 0.2s;
        }
        
        .link-box:hover {
            background: rgba(88, 166, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .footer-info {
            text-align: center;
            color: var(--text-secondary);
            margin-top: 30px;
            font-size: 0.9em;
        }
        
        .refresh-control {
            display: inline-block;
            margin: 10px;
        }
        
        .refresh-control label {
            margin-right: 10px;
            color: var(--text-secondary);
        }
        
        .refresh-control input {
            width: 80px;
            padding: 6px;
            background: var(--bg-dark);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
        }
        
        .progress-bar-wrapper {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            transition: width 0.5s ease;
        }
        
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: 600;
            color: var(--text-primary);
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header-section">
            <h1>🚀 VSR++ Training Monitor</h1>
            <div>
                <span id="statusBadge" class="status-indicator status-training">Training</span>
            </div>
        </div>
        
        <div class="progress-bar-wrapper">
            <div class="card-title">Gesamt-Fortschritt</div>
            <div class="progress-bar">
                <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                <div id="progressText" class="progress-text">0 / 100,000</div>
            </div>
            <div class="card-subtitle" style="margin-top: 10px;">
                Epoche: <span id="epochInfo">1</span>
            </div>
        </div>
        
        <div class="progress-bar-wrapper">
            <div class="card-title">Epochen-Fortschritt</div>
            <div class="progress-bar">
                <div id="epochProgressFill" class="progress-fill" style="width: 0%; background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));"></div>
                <div id="epochProgressText" class="progress-text">0 / 1000</div>
            </div>
        </div>
        
        <div class="section-header">📉 Loss-Werte & Gewichte</div>
        
        <div class="grid-container">
            <div class="info-card">
                <div class="card-title">L1 Loss</div>
                <div class="card-value" id="l1Loss">0.0000</div>
                <div class="card-subtitle">w: <span id="l1Weight">0.60</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">MS Loss</div>
                <div class="card-value" id="msLoss">0.0000</div>
                <div class="card-subtitle">w: <span id="msWeight">0.20</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Gradient Loss</div>
                <div class="card-value" id="gradLoss">0.0000</div>
                <div class="card-subtitle">w: <span id="gradWeight">0.20</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Perceptual Loss</div>
                <div class="card-value" id="percLoss">0.0000</div>
                <div class="card-subtitle">w: <span id="percWeight">0.00</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Total Loss</div>
                <div class="card-value" id="totalLoss">0.0000</div>
                <div class="card-subtitle">Summe aller Komponenten</div>
            </div>
        </div>
        
        <div class="section-header">🎚️ Adaptive System Status</div>
        
        <div class="grid-container">
            <div class="info-card">
                <div class="card-title">Modus</div>
                <div class="card-value" id="adaptiveMode" style="font-size: 1.5em;">Stable</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Cooldown</div>
                <div class="card-value" id="cooldownStatus">Inaktiv</div>
                <div class="card-subtitle" id="cooldownRemaining"></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Gradient Clip</div>
                <div class="card-value" id="gradClip">1.00</div>
            </div>
        </div>
        
        <div class="section-header">📊 Basis-Metriken</div>
        
        <div class="grid-container">
            <div class="info-card">
                <div class="card-title">Iteration</div>
                <div class="card-value" id="stepValue">0</div>
                <div class="card-subtitle">von <span id="maxSteps">100,000</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Learning Rate</div>
                <div class="card-value" id="learnRate">0.0000</div>
                <div class="card-subtitle">Phase: <span id="lrPhase">warmup</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">ETA (Total)</div>
                <div class="card-value" id="etaTotal">--:--:--</div>
                <div class="card-subtitle">Epoch: <span id="etaEpoch">--:--:--</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Speed</div>
                <div class="card-value" id="iterSpeed">0.00</div>
                <div class="card-subtitle">it/s</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">VRAM</div>
                <div class="card-value" id="vramUsage">0.0</div>
                <div class="card-subtitle">GB</div>
            </div>
        </div>
        
        <div class="section-header">🎯 Quality Metriken</div>
        
        <div class="grid-container">
            <div class="info-card">
                <div class="card-title">LR Quality</div>
                <div class="card-value" id="lrQuality">0.0%</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">KI Quality</div>
                <div class="card-value" id="kiQuality">0.0%</div>
                <div class="card-subtitle">Best: <span id="bestQuality">0.0%</span></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Improvement (KI vs LR)</div>
                <div class="card-value" id="improvement">0.0%</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">KI to GT (PSNR/SSIM)</div>
                <div class="card-value" id="kiToGt">0.0%</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Validation Loss</div>
                <div class="card-value" id="valLoss">0.0000</div>
            </div>
        </div>
        
        <div class="section-header">📊 Layer-Aktivitäten</div>
        
        <div id="layerActivitiesBackward" class="layer-activity-container">
            <h3 style="color: var(--accent-blue); margin-bottom: 15px; font-size: 1.1em;">⬅️ Backward Stream</h3>
            <div id="backwardLayers" style="color: var(--text-secondary); text-align: center;">
                Warte auf Daten...
            </div>
        </div>
        
        <div id="layerActivitiesForward" class="layer-activity-container">
            <h3 style="color: var(--accent-green); margin-bottom: 15px; font-size: 1.1em;">➡️ Forward Stream</h3>
            <div id="forwardLayers" style="color: var(--text-secondary); text-align: center;">
                Warte auf Daten...
            </div>
        </div>
        
        <div id="layerActivitiesFusion" class="layer-activity-container">
            <h3 style="color: var(--accent-orange); margin-bottom: 15px; font-size: 1.1em;">🔗 Fusion</h3>
            <div id="fusionLayers" style="color: var(--text-secondary); text-align: center;">
                Warte auf Daten...
            </div>
        </div>
        
        <div class="section-header">🎮 Steuerung</div>
        
        <div class="controls-section">
            <button class="btn btn-primary" onclick="triggerValidation()">
                🔍 Validation starten
            </button>
            
            <a id="tensorboardLink" href="#" class="link-box" target="_blank">
                📈 TensorBoard öffnen
            </a>
            
            <div class="refresh-control">
                <label for="refreshInterval">Auto-Refresh:</label>
                <input type="number" id="refreshInterval" value="5" min="1" max="60" step="1">
                <span style="color: var(--text-secondary); margin-left: 5px;">Sekunden</span>
                <button class="btn btn-success" onclick="updateRefreshRate()" style="margin-left: 10px;">
                    Speichern
                </button>
            </div>
        </div>
        
        <div class="footer-info">
            Letzte Aktualisierung: <span id="lastUpdate">--</span>
        </div>
    </div>
    
    <script>
        let currentRefreshInterval = 5000;
        let refreshTimer = null;
        
        function fetchAndUpdate() {
            fetch('/monitoring/data')
                .then(response => response.json())
                .then(data => {
                    updateAllFields(data);
                })
                .catch(error => console.error('Fehler beim Laden:', error));
        }
        
        function updateAllFields(data) {
            // Basic metrics
            document.getElementById('stepValue').textContent = data.step_current.toLocaleString('de-DE');
            document.getElementById('maxSteps').textContent = data.step_max.toLocaleString('de-DE');
            document.getElementById('totalLoss').textContent = data.total_loss_value.toFixed(4);
            document.getElementById('learnRate').textContent = data.learning_rate_value.toFixed(6);
            document.getElementById('lrPhase').textContent = data.lr_phase_name;
            document.getElementById('etaTotal').textContent = data.eta_total_formatted;
            document.getElementById('etaEpoch').textContent = data.eta_epoch_formatted;
            
            const iterSpeed = data.iteration_duration > 0 ? (1.0 / data.iteration_duration) : 0;
            document.getElementById('iterSpeed').textContent = iterSpeed.toFixed(2);
            document.getElementById('vramUsage').textContent = data.vram_usage_gb.toFixed(1);
            
            // Loss components with weights
            document.getElementById('l1Loss').textContent = data.l1_loss_value.toFixed(4);
            document.getElementById('l1Weight').textContent = data.l1_weight_current.toFixed(2);
            document.getElementById('msLoss').textContent = data.ms_loss_value.toFixed(4);
            document.getElementById('msWeight').textContent = data.ms_weight_current.toFixed(2);
            document.getElementById('gradLoss').textContent = data.gradient_loss_value.toFixed(4);
            document.getElementById('gradWeight').textContent = data.gradient_weight_current.toFixed(2);
            document.getElementById('percLoss').textContent = data.perceptual_loss_value.toFixed(4);
            document.getElementById('percWeight').textContent = data.perceptual_weight_current.toFixed(2);
            
            // Adaptive system status
            document.getElementById('adaptiveMode').textContent = data.adaptive_mode || 'Stable';
            const cooldownStatus = document.getElementById('cooldownStatus');
            const cooldownRemaining = document.getElementById('cooldownRemaining');
            if (data.adaptive_is_cooldown) {
                cooldownStatus.textContent = 'Aktiv';
                cooldownStatus.style.color = 'var(--accent-orange)';
                cooldownRemaining.textContent = data.adaptive_cooldown_remaining + ' Steps verblieben';
            } else {
                cooldownStatus.textContent = 'Inaktiv';
                cooldownStatus.style.color = 'var(--accent-green)';
                cooldownRemaining.textContent = '';
            }
            document.getElementById('gradClip').textContent = data.gradient_clip_val.toFixed(2);
            
            // Quality metrics with fixed labels
            document.getElementById('lrQuality').textContent = (data.quality_lr_value * 100).toFixed(1) + '%';
            document.getElementById('kiQuality').textContent = (data.quality_ki_value * 100).toFixed(1) + '%';
            document.getElementById('bestQuality').textContent = (data.best_quality_ever * 100).toFixed(1) + '%';
            document.getElementById('improvement').textContent = (data.quality_improvement_value * 100).toFixed(1) + '%';
            document.getElementById('kiToGt').textContent = (data.quality_ki_to_gt_value * 100).toFixed(1) + '%';
            document.getElementById('valLoss').textContent = data.validation_loss_value.toFixed(4);
            
            // Progress - Overall
            document.getElementById('epochInfo').textContent = data.epoch_num;
            const progress = (data.step_current / data.step_max) * 100;
            document.getElementById('progressFill').style.width = progress.toFixed(1) + '%';
            document.getElementById('progressText').textContent = 
                data.step_current.toLocaleString('de-DE') + ' / ' + data.step_max.toLocaleString('de-DE');
            
            // Progress - Epoch
            const epochProgress = data.epoch_step_total > 0 ? 
                (data.epoch_step_current / data.epoch_step_total) * 100 : 0;
            document.getElementById('epochProgressFill').style.width = epochProgress.toFixed(1) + '%';
            document.getElementById('epochProgressText').textContent = 
                data.epoch_step_current + ' / ' + data.epoch_step_total;
            
            // Status badge
            const badge = document.getElementById('statusBadge');
            if (data.validation_running) {
                badge.textContent = 'Validierung';
                badge.className = 'status-indicator status-validating';
            } else if (data.training_paused) {
                badge.textContent = 'Pausiert';
                badge.className = 'status-indicator status-paused';
            } else if (data.training_active) {
                badge.textContent = 'Training';
                badge.className = 'status-indicator status-training';
            }
            
            // Layer activities with grouping
            updateLayerActivities(data.layer_activity_map);
            
            // TensorBoard link
            const tbLink = document.getElementById('tensorboardLink');
            tbLink.href = `http://${data.local_ip_address}:${data.tensorboard_port}`;
            
            // Last update
            const updateTime = new Date(data.last_update_time * 1000);
            document.getElementById('lastUpdate').textContent = updateTime.toLocaleTimeString('de-DE');
        }
        
        function updateLayerActivities(activityMap) {
            if (Object.keys(activityMap).length === 0) {
                document.getElementById('backwardLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Daten</div>';
                document.getElementById('forwardLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Daten</div>';
                document.getElementById('fusionLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Daten</div>';
                return;
            }
            
            // Group layers into categories
            const backwardLayers = [];
            const forwardLayers = [];
            const fusionLayers = [];
            
            // Find max value for normalization (if values are > 1.0)
            let maxValue = 0;
            for (const [layerName, activityValue] of Object.entries(activityMap)) {
                maxValue = Math.max(maxValue, activityValue);
            }
            
            // If max > 1.0, we need to normalize
            const needsNormalization = maxValue > 1.0;
            
            for (const [layerName, activityValue] of Object.entries(activityMap)) {
                // Normalize if needed: convert to 0-100 range
                let displayValue, barWidth;
                if (needsNormalization) {
                    // Show as percentage of max
                    displayValue = ((activityValue / maxValue) * 100).toFixed(1);
                    barWidth = displayValue;
                } else {
                    // Already in 0-1 range, convert to percentage
                    displayValue = (activityValue * 100).toFixed(1);
                    barWidth = displayValue;
                }
                
                // Ensure bar width is valid and doesn't exceed 100%
                barWidth = parseFloat(barWidth);
                if (isNaN(barWidth) || barWidth < 0) {
                    barWidth = 0;
                }
                barWidth = Math.min(100, barWidth);
                
                let barClass = 'layer-bar-fill';
                
                // Categorize layer
                if (layerName.toLowerCase().includes('backward')) {
                    if (layerName.includes('Final Fusion')) {
                        barClass += ' final-fusion';
                    } else if (layerName.includes('Fus')) {
                        barClass += ' fusion';
                    }
                    backwardLayers.push({name: layerName, value: displayValue, width: barWidth, barClass});
                } else if (layerName.toLowerCase().includes('forward')) {
                    if (layerName.includes('Final Fusion')) {
                        barClass += ' final-fusion';
                    } else if (layerName.includes('Fus')) {
                        barClass += ' fusion';
                    }
                    forwardLayers.push({name: layerName, value: displayValue, width: barWidth, barClass});
                } else if (layerName.toLowerCase().includes('fus')) {
                    barClass += ' fusion';
                    fusionLayers.push({name: layerName, value: displayValue, width: barWidth, barClass});
                } else {
                    // Default to fusion if unclear
                    fusionLayers.push({name: layerName, value: displayValue, width: barWidth, barClass});
                }
            }
            
            // Render backward layers
            let backwardHtml = '';
            if (backwardLayers.length > 0) {
                for (const layer of backwardLayers) {
                    backwardHtml += `
                        <div class="layer-row">
                            <div class="layer-name">${layer.name}</div>
                            <div class="layer-bar-container">
                                <div class="${layer.barClass}" style="width: ${layer.width}%"></div>
                            </div>
                            <div class="layer-value">${layer.value}%</div>
                        </div>
                    `;
                }
                document.getElementById('backwardLayers').innerHTML = backwardHtml;
            } else {
                document.getElementById('backwardLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Layer</div>';
            }
            
            // Render forward layers
            let forwardHtml = '';
            if (forwardLayers.length > 0) {
                for (const layer of forwardLayers) {
                    forwardHtml += `
                        <div class="layer-row">
                            <div class="layer-name">${layer.name}</div>
                            <div class="layer-bar-container">
                                <div class="${layer.barClass}" style="width: ${layer.width}%"></div>
                            </div>
                            <div class="layer-value">${layer.value}%</div>
                        </div>
                    `;
                }
                document.getElementById('forwardLayers').innerHTML = forwardHtml;
            } else {
                document.getElementById('forwardLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Layer</div>';
            }
            
            // Render fusion layers
            let fusionHtml = '';
            if (fusionLayers.length > 0) {
                for (const layer of fusionLayers) {
                    fusionHtml += `
                        <div class="layer-row">
                            <div class="layer-name">${layer.name}</div>
                            <div class="layer-bar-container">
                                <div class="${layer.barClass}" style="width: ${layer.width}%"></div>
                            </div>
                            <div class="layer-value">${layer.value}%</div>
                        </div>
                    `;
                }
                document.getElementById('fusionLayers').innerHTML = fusionHtml;
            } else {
                document.getElementById('fusionLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Layer</div>';
            }
        }
        
        function triggerValidation() {
            fetch('/monitoring/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'trigger_validation'})
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    alert('✅ Validation wurde gestartet!');
                } else {
                    alert('❌ Fehler: ' + result.message);
                }
            })
            .catch(error => {
                alert('❌ Verbindungsfehler: ' + error);
            });
        }
        
        function updateRefreshRate() {
            const newInterval = parseInt(document.getElementById('refreshInterval').value);
            
            fetch('/monitoring/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'change_refresh', interval: newInterval})
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    currentRefreshInterval = result.interval * 1000;
                    restartAutoRefresh();
                    alert(`✅ Auto-Refresh auf ${result.interval} Sekunden gesetzt`);
                }
            })
            .catch(error => {
                alert('❌ Fehler beim Setzen: ' + error);
            });
        }
        
        function startAutoRefresh() {
            refreshTimer = setInterval(fetchAndUpdate, currentRefreshInterval);
        }
        
        function restartAutoRefresh() {
            if (refreshTimer) clearInterval(refreshTimer);
            startAutoRefresh();
        }
        
        // Initial load and start auto-refresh
        fetchAndUpdate();
        startAutoRefresh();
        
        // Load config
        fetch('/monitoring/config')
            .then(response => response.json())
            .then(config => {
                currentRefreshInterval = config.refresh_interval_seconds * 1000;
                document.getElementById('refreshInterval').value = config.refresh_interval_seconds;
                restartAutoRefresh();
            });
    </script>
</body>
</html>'''


class WebMonitoringInterface:
    """Hauptklasse für das Web-Monitoring-System"""
    
    def __init__(self, port_num=5050, refresh_seconds=5):
        self.server_port = port_num
        self.data_store = CompleteTrainingDataStore()
        self.command_inbox = Queue()
        self.http_server_instance = None
        self.server_daemon_thread = None
        
        # Setze Refresh-Intervall
        WebMonitorRequestProcessor.refresh_interval_sec = refresh_seconds
        
        # Konfiguriere Request-Handler
        WebMonitorRequestProcessor.data_repository = self.data_store
        WebMonitorRequestProcessor.action_queue = self.command_inbox
        
        self._start_http_server()
    
    def _start_http_server(self):
        """Startet HTTP-Server im Daemon-Thread"""
        try:
            self.http_server_instance = HTTPServer(
                ('0.0.0.0', self.server_port),
                WebMonitorRequestProcessor
            )
            
            self.server_daemon_thread = threading.Thread(
                target=self.http_server_instance.serve_forever,
                daemon=True
            )
            self.server_daemon_thread.start()
            
            local_ip = detect_local_ip()
            print(f"🌐 Web-Monitor aktiv:")
            print(f"   • Lokal: http://localhost:{self.server_port}/monitoring")
            print(f"   • Netzwerk: http://{local_ip}:{self.server_port}/monitoring")
            
        except OSError as err:
            if err.errno == errno.EADDRINUSE:
                print(f"⚠️  Port {self.server_port} belegt, Web-Monitor deaktiviert")
            else:
                raise
    
    def update(self, **all_metrics):
        """Aktualisiert alle Metriken im Data Store"""
        self.data_store.update_all_metrics(**all_metrics)
    
    def poll_commands(self):
        """Prüft auf ausstehende Befehle"""
        if not self.command_inbox.empty():
            return self.command_inbox.get()
        return None
    
    def terminate(self):
        """Stoppt den Web-Server"""
        if self.http_server_instance:
            self.http_server_instance.shutdown()
