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
            'adaptive_plateau_counter': 0,
            'adaptive_lr_boost_available': False,
            'adaptive_perceptual_trend': 0,  # Change since last update
            
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
    runtime_config_manager = None
    
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
        """Liefert Konfiguration (z.B. Aktualisierungsintervall) und Runtime-Config"""
        config = {
            'refresh_interval_seconds': self.refresh_interval_sec,
            'auto_refresh_enabled': True
        }
        
        # Add runtime configuration if available
        if hasattr(self, 'runtime_config_manager') and self.runtime_config_manager is not None:
            try:
                runtime_config = self.runtime_config_manager.get_all()
                config['runtime_config'] = runtime_config
                
                # Add metadata about parameter categories
                from ..systems.runtime_config import RUNTIME_SAFE_PARAMS, RUNTIME_CAREFUL_PARAMS, STARTUP_ONLY_PARAMS
                config['config_categories'] = {
                    'safe': list(RUNTIME_SAFE_PARAMS.keys()),
                    'careful': list(RUNTIME_CAREFUL_PARAMS.keys()),
                    'startup_only': list(STARTUP_ONLY_PARAMS)
                }
                config['config_ranges'] = {
                    **RUNTIME_SAFE_PARAMS,
                    **RUNTIME_CAREFUL_PARAMS
                }
            except Exception as e:
                config['runtime_config_error'] = str(e)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        self.wfile.write(json.dumps(config, indent=2).encode('utf-8'))
    
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
        
        /* Color based on absolute activity value (0-2.0 scale) */
        .layer-bar-fill.activity-low {
            /* 0.0-0.5: Green */
            background: linear-gradient(90deg, #22c55e, #10b981);
        }
        
        .layer-bar-fill.activity-moderate {
            /* 0.5-1.0: Cyan/Yellow */
            background: linear-gradient(90deg, #06b6d4, #eab308);
        }
        
        .layer-bar-fill.activity-high {
            /* 1.0-1.5: Orange */
            background: linear-gradient(90deg, #f97316, #ea580c);
        }
        
        .layer-bar-fill.activity-extreme {
            /* 1.5-2.0+: Red */
            background: linear-gradient(90deg, #ef4444, #dc2626);
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
        
        /* Training Score Card */
        .training-score-card {
            background: var(--bg-card);
            border: 3px solid var(--border-color);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .training-score-card.excellent {
            border-color: var(--accent-green);
            background: linear-gradient(135deg, rgba(63, 185, 80, 0.05), var(--bg-card));
        }
        
        .training-score-card.good {
            border-color: var(--accent-blue);
            background: linear-gradient(135deg, rgba(88, 166, 255, 0.05), var(--bg-card));
        }
        
        .training-score-card.moderate {
            border-color: var(--accent-orange);
            background: linear-gradient(135deg, rgba(210, 153, 34, 0.05), var(--bg-card));
        }
        
        .training-score-card.needs-attention {
            border-color: var(--accent-red);
            background: linear-gradient(135deg, rgba(248, 81, 73, 0.05), var(--bg-card));
        }
        
        .score-title {
            font-size: 1.3em;
            color: var(--text-secondary);
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .score-value {
            font-size: 4em;
            font-weight: 700;
            margin: 20px 0;
        }
        
        .score-label {
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .score-components {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        
        .score-component {
            font-size: 1em;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
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
        
        .control-buttons {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 0.95em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
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
        
        /* Stacked Bar Chart Styles */
        .stacked-bars-container {
            display: flex;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .bar-section {
            flex: 1;
            min-width: 300px;
        }
        
        .bar-label {
            font-size: 0.9em;
            margin-bottom: 10px;
            color: var(--text-secondary);
            font-weight: 600;
        }
        
        .stacked-bar {
            height: 60px;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            position: relative;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
        }
        
        .bar-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85em;
            font-weight: bold;
            transition: all 0.3s ease;
            position: relative;
            min-width: 3%;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }
        
        .bar-segment:hover {
            filter: brightness(1.2);
            transform: scaleY(1.05);
            z-index: 10;
        }
        
        .segment-l1 { 
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }
        .segment-ms { 
            background: linear-gradient(135deg, #f59e0b, #d97706);
        }
        .segment-grad { 
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        }
        .segment-perceptual { 
            background: linear-gradient(135deg, #06b6d4, #0891b2);
        }
        
        .legend {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
            font-size: 0.9em;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
        
        .legend-value {
            color: var(--text-secondary);
            margin-left: 5px;
        }
        
        /* Peak Activity Bar */
        .peak-bar-container {
            margin: 20px 0;
        }
        
        .peak-bar {
            height: 50px;
            background: linear-gradient(
                to right,
                #4ade80 0%,
                #4ade80 25%,
                #fbbf24 50%,
                #fb923c 75%,
                #ef4444 100%
            );
            border-radius: 8px;
            position: relative;
            margin-bottom: 10px;
            border: 2px solid var(--border-color);
        }
        
        .peak-indicator {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            font-size: 18px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.8);
            transition: left 0.5s ease;
        }
        
        .peak-scale {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }
        
        .peak-info {
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .peak-warning {
            color: var(--accent-red);
            font-weight: bold;
            margin-top: 10px;
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
            <div class="control-buttons">
                <button class="btn btn-primary" onclick="downloadDataAsJSON()">
                    📥 Download Data (JSON)
                </button>
                <button class="btn btn-success" onclick="requestValidation()">
                    🔍 Run Validation
                </button>
                <button class="btn btn-primary" onclick="openConfigPage()">
                    ⚙️ Konfiguration
                </button>
            </div>
        </div>
        
        <!-- TRAINING SCORE - Prominent Performance Indicator -->
        <div id="trainingScoreCard" class="training-score-card excellent">
            <div class="score-title">⭐ TRAINING SCORE</div>
            <div class="score-value" id="scoreValue">85.0%</div>
            <div class="score-label" id="scoreLabel">EXCELLENT</div>
            <div class="score-components">
                <div class="score-component" id="scoreTrend">Trend: Converging</div>
                <div class="score-component" id="scoreQuality">Quality: 70%</div>
                <div class="score-component" id="scoreStability">Stability: Stable</div>
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
        
        <!-- NEW: Stacked Bar Chart Visualization -->
        <div class="layer-activity-container">
            <div class="card-title" style="font-size: 1.2em; margin-bottom: 20px;">📊 Loss & Weight Distribution</div>
            
            <div class="stacked-bars-container">
                <!-- Weight Distribution -->
                <div class="bar-section">
                    <div class="bar-label">Weight Distribution (%)</div>
                    <div class="stacked-bar" id="weightBar">
                        <div class="bar-segment segment-l1" id="weightL1">
                            <span>L1: 0%</span>
                        </div>
                        <div class="bar-segment segment-ms" id="weightMS">
                            <span>MS: 0%</span>
                        </div>
                        <div class="bar-segment segment-grad" id="weightGrad">
                            <span>Grad: 0%</span>
                        </div>
                        <div class="bar-segment segment-perceptual" id="weightPerc">
                            <span>Perc: 0%</span>
                        </div>
                    </div>
                </div>
                
                <!-- Loss Value Distribution -->
                <div class="bar-section">
                    <div class="bar-label">Loss Value Distribution (relative)</div>
                    <div class="stacked-bar" id="lossBar">
                        <div class="bar-segment segment-l1" id="lossL1">
                            <span>L1: 0.000</span>
                        </div>
                        <div class="bar-segment segment-ms" id="lossMS">
                            <span>MS: 0.000</span>
                        </div>
                        <div class="bar-segment segment-grad" id="lossGrad">
                            <span>Grad: 0.000</span>
                        </div>
                        <div class="bar-segment segment-perceptual" id="lossPerc">
                            <span>Perc: 0.000</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color segment-l1"></div>
                    <span>L1 Loss <span class="legend-value" id="legendL1">0.0000</span></span>
                </div>
                <div class="legend-item">
                    <div class="legend-color segment-ms"></div>
                    <span>MS Loss <span class="legend-value" id="legendMS">0.0000</span></span>
                </div>
                <div class="legend-item">
                    <div class="legend-color segment-grad"></div>
                    <span>Gradient Loss <span class="legend-value" id="legendGrad">0.0000</span></span>
                </div>
                <div class="legend-item">
                    <div class="legend-color segment-perceptual"></div>
                    <span>Perceptual Loss <span class="legend-value" id="legendPerc">0.0000</span></span>
                </div>
                <div class="legend-item">
                    <strong>Total Loss: <span class="legend-value" id="legendTotal">0.0000</span></strong>
                </div>
            </div>
        </div>
        
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
                <div class="card-title">Plateau Counter</div>
                <div class="card-value" id="plateauCounter">0</div>
                <div class="card-subtitle" id="plateauWarning"></div>
            </div>
            
            <div class="info-card">
                <div class="card-title">LR Boost</div>
                <div class="card-value" id="lrBoostStatus">Bereit</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Perceptual Weight</div>
                <div class="card-value" id="perceptualWeightDisplay">5.0%</div>
                <div class="card-subtitle" id="perceptualTrend"></div>
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
            
            <div class="info-card">
                <div class="card-title">👁️ AdamW Momentum</div>
                <div class="card-value" id="adamMomentum">0.000</div>
                <div class="card-subtitle">Optimizer</div>
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
        
        <div class="section-header">🔥 Peak Layer Activity</div>
        
        <div class="layer-activity-container">
            <div class="peak-bar-container">
                <div class="peak-scale">
                    <span>0.0</span>
                    <span>0.5</span>
                    <span>1.0</span>
                    <span style="color: var(--accent-orange)">1.5</span>
                    <span style="color: var(--accent-red)">2.0+</span>
                </div>
                <div class="peak-bar">
                    <div class="peak-indicator" id="peakIndicator">0.00</div>
                </div>
            </div>
            
            <div class="peak-info">
                <span>Peak Layer: <strong id="peakLayer">-</strong></span>
                <span>Value: <strong id="peakValue">-</strong></span>
                <span style="color: var(--text-secondary); font-size: 0.9em;">Actual: <strong id="peakActualValue">-</strong></span>
            </div>
            <div class="peak-warning" id="peakWarning" style="display: none;"></div>
        </div>
        
        <!-- Stream Summary: Backward, Forward, Final Fusion -->
        <div class="layer-activity-container" style="margin-top: 20px;">
            <div class="card-title" style="font-size: 1.1em; margin-bottom: 15px;">📊 Stream-Übersicht (Durchschnitt)</div>
            
            <div class="layer-row">
                <div class="layer-name" style="color: var(--accent-blue);">⬅️ Backward Stream</div>
                <div class="layer-bar-container">
                    <div class="layer-bar-fill" id="backwardAvgBar" style="width: 0%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));"></div>
                </div>
                <div class="layer-value" id="backwardAvgValue">0.0%</div>
            </div>
            
            <div class="layer-row">
                <div class="layer-name" style="color: var(--accent-green);">➡️ Forward Stream</div>
                <div class="layer-bar-container">
                    <div class="layer-bar-fill" id="forwardAvgBar" style="width: 0%; background: linear-gradient(90deg, var(--accent-green), #00ff88);"></div>
                </div>
                <div class="layer-value" id="forwardAvgValue">0.0%</div>
            </div>
            
            <div class="layer-row">
                <div class="layer-name" style="color: var(--accent-orange);">🔗 Final Fusion</div>
                <div class="layer-bar-container">
                    <div class="layer-bar-fill final-fusion" id="fusionAvgBar" style="width: 0%;"></div>
                </div>
                <div class="layer-value" id="fusionAvgValue">0.0%</div>
            </div>
        </div>
        
        <div class="section-header">📊 Layer-Aktivitäten (Details)</div>
        
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
            document.getElementById('adamMomentum').textContent = data.adam_momentum_avg.toFixed(3);
            
            // Loss components with weights
            document.getElementById('l1Loss').textContent = data.l1_loss_value.toFixed(4);
            document.getElementById('l1Weight').textContent = data.l1_weight_current.toFixed(2);
            document.getElementById('msLoss').textContent = data.ms_loss_value.toFixed(4);
            document.getElementById('msWeight').textContent = data.ms_weight_current.toFixed(2);
            document.getElementById('gradLoss').textContent = data.gradient_loss_value.toFixed(4);
            document.getElementById('gradWeight').textContent = data.gradient_weight_current.toFixed(2);
            document.getElementById('percLoss').textContent = data.perceptual_loss_value.toFixed(4);
            document.getElementById('percWeight').textContent = data.perceptual_weight_current.toFixed(2);
            
            // Update stacked bar charts
            updateStackedBars(data);
            
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
            
            // Plateau counter with color coding
            const plateauCounter = data.adaptive_plateau_counter || 0;
            const plateauEl = document.getElementById('plateauCounter');
            const plateauWarning = document.getElementById('plateauWarning');
            plateauEl.textContent = plateauCounter;
            if (plateauCounter > 300) {
                plateauEl.style.color = 'var(--accent-red)';
                plateauWarning.textContent = '🚨 WARNUNG';
                plateauWarning.style.color = 'var(--accent-red)';
            } else if (plateauCounter > 150) {
                plateauEl.style.color = 'var(--accent-orange)';
                plateauWarning.textContent = '🟡 Erhöht';
                plateauWarning.style.color = 'var(--accent-orange)';
            } else {
                plateauEl.style.color = 'var(--accent-green)';
                plateauWarning.textContent = '🟢 Normal';
                plateauWarning.style.color = 'var(--accent-green)';
            }
            
            // LR Boost status
            const lrBoostStatus = document.getElementById('lrBoostStatus');
            if (data.adaptive_lr_boost_available) {
                lrBoostStatus.textContent = '⚡ Bereit';
                lrBoostStatus.style.color = 'var(--accent-green)';
            } else {
                lrBoostStatus.textContent = '⏳ Cooldown';
                lrBoostStatus.style.color = 'var(--accent-orange)';
            }
            
            // Perceptual weight with trend
            const percWeight = (data.perceptual_weight_current * 100).toFixed(1);
            const percWeightDisplay = document.getElementById('perceptualWeightDisplay');
            const percTrend = document.getElementById('perceptualTrend');
            percWeightDisplay.textContent = percWeight + '%';
            
            const trend = data.adaptive_perceptual_trend || 0;
            if (trend > 0.001) {
                percTrend.textContent = '⬆️ Steigend';
                percTrend.style.color = 'var(--accent-green)';
            } else if (trend < -0.001) {
                percTrend.textContent = '⬇️ Fallend';
                percTrend.style.color = 'var(--accent-orange)';
            } else {
                percTrend.textContent = '➡️ Stabil';
                percTrend.style.color = 'var(--text-secondary)';
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
            
            // Update Training Score (Prominent Performance Indicator)
            updateTrainingScore(data);
            
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
        
        function updateStackedBars(data) {
            // Get loss values
            const l1Loss = data.l1_loss_value || 0;
            const msLoss = data.ms_loss_value || 0;
            const gradLoss = data.gradient_loss_value || 0;
            const percLoss = data.perceptual_loss_value || 0;
            const totalLoss = l1Loss + msLoss + gradLoss + percLoss;
            
            // Get weights
            const l1Weight = data.l1_weight_current || 0;
            const msWeight = data.ms_weight_current || 0;
            const gradWeight = data.gradient_weight_current || 0;
            const percWeight = data.perceptual_weight_current || 0;
            const totalWeight = l1Weight + msWeight + gradWeight + percWeight;
            
            // Update weight bar (percentages)
            if (totalWeight > 0) {
                const l1Pct = (l1Weight / totalWeight * 100);
                const msPct = (msWeight / totalWeight * 100);
                const gradPct = (gradWeight / totalWeight * 100);
                const percPct = (percWeight / totalWeight * 100);
                
                const weightL1 = document.getElementById('weightL1');
                const weightMS = document.getElementById('weightMS');
                const weightGrad = document.getElementById('weightGrad');
                const weightPerc = document.getElementById('weightPerc');
                
                weightL1.style.width = l1Pct + '%';
                weightMS.style.width = msPct + '%';
                weightGrad.style.width = gradPct + '%';
                weightPerc.style.width = percPct + '%';
                
                weightL1.innerHTML = `<span>L1: ${l1Pct.toFixed(1)}%</span>`;
                weightMS.innerHTML = `<span>MS: ${msPct.toFixed(1)}%</span>`;
                weightGrad.innerHTML = `<span>Grad: ${gradPct.toFixed(1)}%</span>`;
                weightPerc.innerHTML = `<span>Perc: ${percPct.toFixed(1)}%</span>`;
            }
            
            // Update loss bar (relative contributions)
            if (totalLoss > 0) {
                const l1LossPct = (l1Loss / totalLoss * 100);
                const msLossPct = (msLoss / totalLoss * 100);
                const gradLossPct = (gradLoss / totalLoss * 100);
                const percLossPct = (percLoss / totalLoss * 100);
                
                const lossL1 = document.getElementById('lossL1');
                const lossMS = document.getElementById('lossMS');
                const lossGrad = document.getElementById('lossGrad');
                const lossPerc = document.getElementById('lossPerc');
                
                lossL1.style.width = l1LossPct + '%';
                lossMS.style.width = msLossPct + '%';
                lossGrad.style.width = gradLossPct + '%';
                lossPerc.style.width = percLossPct + '%';
                
                lossL1.innerHTML = `<span>L1: ${l1Loss.toFixed(4)}</span>`;
                lossMS.innerHTML = `<span>MS: ${msLoss.toFixed(4)}</span>`;
                lossGrad.innerHTML = `<span>Grad: ${gradLoss.toFixed(4)}</span>`;
                lossPerc.innerHTML = `<span>Perc: ${percLoss.toFixed(4)}</span>`;
            }
            
            // Update legend
            document.getElementById('legendL1').textContent = l1Loss.toFixed(4);
            document.getElementById('legendMS').textContent = msLoss.toFixed(4);
            document.getElementById('legendGrad').textContent = gradLoss.toFixed(4);
            document.getElementById('legendPerc').textContent = percLoss.toFixed(4);
            document.getElementById('legendTotal').textContent = totalLoss.toFixed(4);
        }
        
        function updateTrainingScore(data) {
            // Calculate training score based on multiple factors
            let scoreTotal = 0;
            let scoreMax = 0;
            let components = [];
            
            // 1. Loss trend (up to 30 points) - based on plateau counter
            const plateauCounter = data.adaptive_plateau_counter || 0;
            let lossTrendScore = 0;
            let lossTrendText = '';
            let lossTrendColor = '';
            
            if (plateauCounter < 150) {
                lossTrendScore = 30.0;
                lossTrendText = 'Converging';
                lossTrendColor = 'var(--accent-green)';
            } else if (plateauCounter < 300) {
                lossTrendScore = 20.0;
                lossTrendText = 'Plateau';
                lossTrendColor = 'var(--accent-blue)';
            } else {
                lossTrendScore = 10.0;
                lossTrendText = 'Stagnating';
                lossTrendColor = 'var(--accent-red)';
            }
            scoreTotal += lossTrendScore;
            scoreMax += 30.0;
            components.push({ name: 'Trend', text: lossTrendText, color: lossTrendColor });
            
            // 2. Quality metrics (up to 40 points) - if available
            const kiQuality = (data.quality_ki_value || 0) * 100;
            if (kiQuality > 0) {
                const qualityScore = (kiQuality / 100.0) * 40.0;
                scoreTotal += qualityScore;
                scoreMax += 40.0;
                
                const qualityColor = kiQuality >= 70 ? 'var(--accent-green)' : 
                                    kiQuality >= 50 ? 'var(--accent-blue)' : 'var(--accent-orange)';
                components.push({ name: 'Quality', text: kiQuality.toFixed(0) + '%', color: qualityColor });
            }
            
            // 3. Learning stability (up to 30 points) - based on adaptive mode
            const adaptiveMode = data.adaptive_mode || 'Stable';
            let stabilityScore = 0;
            let stabilityText = '';
            let stabilityColor = '';
            
            if (adaptiveMode === 'Stable' || plateauCounter < 150) {
                stabilityScore = 30.0;
                stabilityText = 'Stable';
                stabilityColor = 'var(--accent-green)';
            } else if (plateauCounter < 300) {
                stabilityScore = 20.0;
                stabilityText = 'Moderate';
                stabilityColor = 'var(--accent-blue)';
            } else {
                stabilityScore = 10.0;
                stabilityText = 'Unstable';
                stabilityColor = 'var(--accent-red)';
            }
            scoreTotal += stabilityScore;
            scoreMax += 30.0;
            components.push({ name: 'Stability', text: stabilityText, color: stabilityColor });
            
            // Calculate overall percentage
            const trainingScorePct = scoreMax > 0 ? (scoreTotal / scoreMax) * 100.0 : 50.0;
            
            // Determine card style and label
            let cardClass = 'training-score-card ';
            let scoreLabel = '';
            let scoreColor = '';
            let scoreIcon = '';
            
            if (trainingScorePct >= 80) {
                cardClass += 'excellent';
                scoreLabel = 'EXCELLENT';
                scoreColor = 'var(--accent-green)';
                scoreIcon = '🟢';
            } else if (trainingScorePct >= 60) {
                cardClass += 'good';
                scoreLabel = 'GOOD';
                scoreColor = 'var(--accent-blue)';
                scoreIcon = '🔵';
            } else if (trainingScorePct >= 40) {
                cardClass += 'moderate';
                scoreLabel = 'MODERATE';
                scoreColor = 'var(--accent-orange)';
                scoreIcon = '🟡';
            } else {
                cardClass += 'needs-attention';
                scoreLabel = 'NEEDS ATTENTION';
                scoreColor = 'var(--accent-red)';
                scoreIcon = '🔴';
            }
            
            // Update UI
            const scoreCard = document.getElementById('trainingScoreCard');
            scoreCard.className = cardClass;
            
            const scoreValue = document.getElementById('scoreValue');
            scoreValue.textContent = scoreIcon + ' ' + trainingScorePct.toFixed(1) + '%';
            scoreValue.style.color = scoreColor;
            
            document.getElementById('scoreLabel').textContent = scoreLabel;
            
            // Update components
            document.getElementById('scoreTrend').innerHTML = 
                `<span style="color: ${components[0].color}">${components[0].name}: ${components[0].text}</span>`;
            
            if (components.length > 1) {
                document.getElementById('scoreQuality').innerHTML = 
                    `<span style="color: ${components[1].color}">${components[1].name}: ${components[1].text}</span>`;
            }
            
            if (components.length > 2) {
                document.getElementById('scoreStability').innerHTML = 
                    `<span style="color: ${components[2].color}">${components[2].name}: ${components[2].text}</span>`;
            }
        }
        
        function updatePeakActivity(peakValue, peakLayer) {
            // Update indicator position (0-2.0 scale)
            const percentage = Math.min((peakValue / 2.0) * 100, 100);
            const indicator = document.getElementById('peakIndicator');
            indicator.style.left = percentage + '%';
            indicator.textContent = peakValue.toFixed(2);
            
            // Update info
            document.getElementById('peakLayer').textContent = peakLayer;
            document.getElementById('peakValue').textContent = peakValue.toFixed(3);
            document.getElementById('peakActualValue').textContent = peakValue.toFixed(3);
            
            // Update warning
            const warningEl = document.getElementById('peakWarning');
            if (peakValue > 2.0) {
                warningEl.textContent = '🔴 EXTREME! Check training stability!';
                warningEl.style.display = 'block';
            } else if (peakValue > 1.5) {
                warningEl.textContent = '⚠️ Unusually high activity!';
                warningEl.style.display = 'block';
            } else {
                warningEl.style.display = 'none';
            }
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
            
            // Find max value for peak detection
            let maxValue = 0;
            let peakLayerName = '-';
            for (const [layerName, activityValue] of Object.entries(activityMap)) {
                if (activityValue > maxValue) {
                    maxValue = activityValue;
                    peakLayerName = layerName;
                }
            }
            
            // Update peak activity visualization (uses absolute 0-2.0 scale)
            updatePeakActivity(maxValue, peakLayerName);
            
            // Process all layers - bars are RELATIVE to peak (100% = peak value)
            for (const [layerName, activityValue] of Object.entries(activityMap)) {
                // Store the actual value
                const actualValue = activityValue.toFixed(3);
                
                // Calculate bar width RELATIVE to peak
                // The layer with maxValue gets 100%, others are proportional
                let barWidth = maxValue > 0 ? (activityValue / maxValue) * 100 : 0;
                
                // Display value is the percentage relative to peak
                const displayValue = barWidth.toFixed(1);
                
                // Ensure bar width is valid
                if (isNaN(barWidth) || barWidth < 0) {
                    barWidth = 0;
                }
                barWidth = Math.min(100, barWidth);
                
                // Determine bar color based on ABSOLUTE value (0-2.0 scale)
                // This gives visual indication of absolute activity level
                let barClass = 'layer-bar-fill';
                
                // Add color class based on absolute activity value
                if (activityValue >= 1.5) {
                    barClass += ' activity-extreme';  // Red
                } else if (activityValue >= 1.0) {
                    barClass += ' activity-high';  // Orange
                } else if (activityValue >= 0.5) {
                    barClass += ' activity-moderate';  // Cyan/Yellow
                } else {
                    barClass += ' activity-low';  // Green
                }
                
                // Categorize layer by type (for sorting/grouping)
                if (layerName.toLowerCase().includes('backward')) {
                    backwardLayers.push({name: layerName, value: displayValue, actualValue: actualValue, width: barWidth, barClass});
                } else if (layerName.toLowerCase().includes('forward')) {
                    forwardLayers.push({name: layerName, value: displayValue, actualValue: actualValue, width: barWidth, barClass});
                } else if (layerName.toLowerCase().includes('fus')) {
                    fusionLayers.push({name: layerName, value: displayValue, actualValue: actualValue, width: barWidth, barClass});
                } else {
                    // Default to fusion if unclear
                    fusionLayers.push({name: layerName, value: displayValue, actualValue: actualValue, width: barWidth, barClass});
                }
            }
            
            // Render backward layers
            let backwardHtml = '';
            let backwardSum = 0;
            let backwardCount = 0;
            if (backwardLayers.length > 0) {
                for (const layer of backwardLayers) {
                    backwardHtml += `
                        <div class="layer-row">
                            <div class="layer-name">${layer.name}</div>
                            <div class="layer-bar-container">
                                <div class="${layer.barClass}" style="width: ${layer.width}%"></div>
                            </div>
                            <div class="layer-value">${layer.value}% <span style="color: var(--text-secondary); font-size: 0.85em;">(${layer.actualValue})</span></div>
                        </div>
                    `;
                    backwardSum += parseFloat(layer.value);
                    backwardCount++;
                }
                document.getElementById('backwardLayers').innerHTML = backwardHtml;
                
                // Update backward average
                if (backwardCount > 0) {
                    const backwardAvg = backwardSum / backwardCount;
                    document.getElementById('backwardAvgBar').style.width = Math.min(backwardAvg, 100) + '%';
                    document.getElementById('backwardAvgValue').innerHTML = `${backwardAvg.toFixed(1)}%`;
                }
            } else {
                document.getElementById('backwardLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Layer</div>';
            }
            
            // Render forward layers
            let forwardHtml = '';
            let forwardSum = 0;
            let forwardCount = 0;
            if (forwardLayers.length > 0) {
                for (const layer of forwardLayers) {
                    forwardHtml += `
                        <div class="layer-row">
                            <div class="layer-name">${layer.name}</div>
                            <div class="layer-bar-container">
                                <div class="${layer.barClass}" style="width: ${layer.width}%"></div>
                            </div>
                            <div class="layer-value">${layer.value}% <span style="color: var(--text-secondary); font-size: 0.85em;">(${layer.actualValue})</span></div>
                        </div>
                    `;
                    forwardSum += parseFloat(layer.value);
                    forwardCount++;
                }
                document.getElementById('forwardLayers').innerHTML = forwardHtml;
                
                // Update forward average
                if (forwardCount > 0) {
                    const forwardAvg = forwardSum / forwardCount;
                    document.getElementById('forwardAvgBar').style.width = Math.min(forwardAvg, 100) + '%';
                    document.getElementById('forwardAvgValue').innerHTML = `${forwardAvg.toFixed(1)}%`;
                }
            } else {
                document.getElementById('forwardLayers').innerHTML = 
                    '<div style="color: var(--text-secondary); text-align: center;">Keine Layer</div>';
            }
            
            // Render fusion layers
            let fusionHtml = '';
            let fusionSum = 0;
            let fusionCount = 0;
            if (fusionLayers.length > 0) {
                for (const layer of fusionLayers) {
                    fusionHtml += `
                        <div class="layer-row">
                            <div class="layer-name">${layer.name}</div>
                            <div class="layer-bar-container">
                                <div class="${layer.barClass}" style="width: ${layer.width}%"></div>
                            </div>
                            <div class="layer-value">${layer.value}% <span style="color: var(--text-secondary); font-size: 0.85em;">(${layer.actualValue})</span></div>
                        </div>
                    `;
                    fusionSum += parseFloat(layer.value);
                    fusionCount++;
                }
                document.getElementById('fusionLayers').innerHTML = fusionHtml;
                
                // Update fusion average
                if (fusionCount > 0) {
                    const fusionAvg = fusionSum / fusionCount;
                    document.getElementById('fusionAvgBar').style.width = Math.min(fusionAvg, 100) + '%';
                    document.getElementById('fusionAvgValue').innerHTML = `${fusionAvg.toFixed(1)}%`;
                }
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
        
        function downloadDataAsJSON() {
            // Fetch current data and trigger download
            fetch('/monitoring/data')
                .then(response => response.json())
                .then(data => {
                    // Create filename with timestamp
                    const now = new Date();
                    const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
                    const filename = `vsr_training_data_${timestamp}.json`;
                    
                    // Convert data to JSON string with pretty formatting
                    const jsonStr = JSON.stringify(data, null, 2);
                    
                    // Create blob and download link
                    const blob = new Blob([jsonStr], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    
                    // Create temporary link and trigger download
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    
                    // Cleanup
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    console.log(`✅ Downloaded: ${filename}`);
                })
                .catch(error => {
                    alert('❌ Download-Fehler: ' + error);
                    console.error('Download error:', error);
                });
        }
        
        function requestValidation() {
            triggerValidation();
        }
        
        function openConfigPage() {
            // Check if config API is available
            window.open('/config', '_blank');
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
    
    def __init__(self, port_num=5050, refresh_seconds=5, runtime_config=None):
        self.server_port = port_num
        self.data_store = CompleteTrainingDataStore()
        self.command_inbox = Queue()
        self.http_server_instance = None
        self.server_daemon_thread = None
        self.runtime_config = runtime_config
        
        # Setze Refresh-Intervall
        WebMonitorRequestProcessor.refresh_interval_sec = refresh_seconds
        
        # Konfiguriere Request-Handler
        WebMonitorRequestProcessor.data_repository = self.data_store
        WebMonitorRequestProcessor.action_queue = self.command_inbox
        WebMonitorRequestProcessor.runtime_config_manager = runtime_config
        
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
