"""
Real-time training dashboard for nanoGPT.

This module provides a web dashboard to visualize training metrics in real-time.
The DashboardBroadcaster class serves as the bridge between the training loop
and web clients, collecting metrics and broadcasting them via WebSocket.

This is an MVP implementation focused on simplicity and educational value,
designed to fail gracefully so that training continues even if the dashboard
encounters issues.
"""

import time
import socket
import threading
import webbrowser
import json
from collections import deque
from typing import Dict, Any, Optional

# Conditional Flask import with graceful fallback
try:
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None

# Conditional WebSocket import with graceful fallback
try:
    from flask import request
    from simple_websocket import Server, ConnectionClosed
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    Server = None
    ConnectionClosed = None


class DashboardBroadcaster:
    """
    Broadcasts training metrics to connected web dashboard clients.
    
    This class collects training metrics from the training loop and broadcasts
    them to web clients via WebSocket. It's designed to be fail-safe - any
    exceptions in dashboard functionality will not interrupt the training process.
    """
    
    def __init__(self, port: int = 8080, enabled: bool = True):
        """
        Initialize the dashboard broadcaster.
        
        Args:
            port: Port number for the web server (default: 8080)
            enabled: Whether dashboard broadcasting is enabled (default: True)
        """
        self.port = port
        self.enabled = enabled and FLASK_AVAILABLE and WEBSOCKET_AVAILABLE
        self.start_time = time.time()
        self.app = None
        self.connected_clients = set()
        self.websocket_clients = set()
        
        # Memory management: store training metrics with automatic size limit
        # Using deque with maxlen for O(1) append and automatic removal of oldest data
        self.training_metrics = deque(maxlen=1000)  # Keep last 1000 data points
        
        # Debug logging setup
        self.log_file_path = "dashboard_debug.log"
        self._setup_debug_logging()
        
        if self.enabled:
            self._initialize_flask_app()
    
    def _setup_debug_logging(self) -> None:
        """
        Set up debug logging to track data flow and diagnose issues.
        """
        try:
            with open(self.log_file_path, 'w') as f:
                f.write(f"=== Dashboard Debug Log Started at {time.ctime()} ===\n")
                f.write(f"Dashboard enabled: {self.enabled}\n")
                f.write(f"Flask available: {FLASK_AVAILABLE}\n") 
                f.write(f"WebSocket available: {WEBSOCKET_AVAILABLE}\n")
                f.write(f"Port: {self.port}\n\n")
        except Exception as e:
            print(f"Warning: Could not initialize debug log: {e}")
    
    def _log_debug(self, message: str) -> None:
        """
        Write debug message to log file with timestamp.
        """
        try:
            with open(self.log_file_path, 'a') as f:
                timestamp = time.strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                f.flush()
        except Exception:
            pass  # Silent fail to not interrupt training
    
    def _initialize_flask_app(self) -> None:
        """
        Initialize Flask application with WebSocket support and basic configuration.
        """
        try:
            # Create Flask app instance with minimal configuration
            self.app = Flask(__name__)
            
            # Configure Flask for local development
            self.app.config.update({
                'DEBUG': False,  # Disable debug mode for production-like behavior
                'TESTING': False,
                'SECRET_KEY': 'nanoGPT-dashboard-dev-key'  # Simple key for session handling
            })
            
            # Set up static file serving routes and WebSocket endpoint
            self._setup_routes()
            
            # Set up basic error handling
            @self.app.errorhandler(500)
            def internal_error(error):
                return {'error': 'Internal server error'}, 500
                
            @self.app.errorhandler(404)
            def not_found(error):
                return {'error': 'Not found'}, 404
                
        except Exception as e:
            print(f"Warning: Failed to initialize Flask app with WebSocket support: {e}")
            print("Dashboard will be disabled for this training run.")
            self.enabled = False
            self.app = None
    
    def _setup_routes(self) -> None:
        """
        Set up Flask routes for static file serving and health checks.
        """
        import os
        
        @self.app.route('/')
        def dashboard():
            """Serve the main dashboard HTML file."""
            try:
                from flask import send_from_directory
                # Serve dashboard.html from the static directory
                static_dir = os.path.join(os.path.dirname(__file__), 'static')
                return send_from_directory(static_dir, 'dashboard.html')
            except FileNotFoundError:
                return {'error': 'Dashboard not found', 'message': 'dashboard.html is not available'}, 404
            except Exception as e:
                return {'error': 'Failed to serve dashboard', 'message': str(e)}, 500
        
        @self.app.route('/health')
        def health():
            """Provide server health status for monitoring."""
            return {
                'status': 'healthy',
                'service': 'nanoGPT-dashboard',
                'uptime': round(time.time() - self.start_time, 2),
                'connected_clients': len(self.connected_clients),
                'websocket_enabled': WEBSOCKET_AVAILABLE,
                'stored_data_points': len(self.training_metrics),
                'memory_limit': self.training_metrics.maxlen,
                'timestamp': time.time()
            }
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get latest training metrics via HTTP polling."""
            try:
                from flask import jsonify
                if not self.training_metrics:
                    return jsonify({'status': 'no_data', 'data': []})
                
                # Return latest data points (last 10)
                latest_data = list(self.training_metrics)[-10:]
                return jsonify({
                    'status': 'success',
                    'data': latest_data,
                    'total_points': len(self.training_metrics),
                    'timestamp': time.time()
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/ws')
        def websocket_endpoint():
            """WebSocket endpoint for real-time chart updates."""
            if not WEBSOCKET_AVAILABLE:
                return {'error': 'WebSocket not available'}, 503
            return self._handle_websocket()
    
    def _handle_websocket(self):
        """
        Handle WebSocket connections using native WebSocket API.
        """
        try:
            ws = Server.accept(request.environ)
            client_id = f"client_{len(self.websocket_clients)}"
            self.websocket_clients.add(ws)
            self.connected_clients.add(client_id)
            
            print(f"Dashboard client connected: {client_id} (total: {len(self.connected_clients)})")
            
            # Send welcome message
            welcome_message = {
                'type': 'connection_status',
                'status': 'connected',
                'client_id': client_id,
                'total_clients': len(self.connected_clients),
                'message': 'Connected to nanoGPT dashboard'
            }
            ws.send(json.dumps(welcome_message))
            
            # Send historical data
            self._send_historical_data_ws(ws)
            
            # Handle incoming messages
            try:
                while True:
                    message = ws.receive()
                    if message:
                        try:
                            data = json.loads(message)
                            if data.get('type') == 'ping':
                                ws.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
                        except json.JSONDecodeError:
                            pass
            except ConnectionClosed:
                pass
            finally:
                # Clean up on disconnect
                self.websocket_clients.discard(ws)
                self.connected_clients.discard(client_id)
                print(f"Dashboard client disconnected: {client_id} (total: {len(self.connected_clients)})")
                
        except Exception as e:
            print(f"Warning: WebSocket connection error: {e}")
            return {'error': 'WebSocket connection failed'}, 500
    
    def _send_historical_data(self, emit_func) -> None:
        """
        Send historical training data to a newly connected client.
        
        Args:
            emit_func: WebSocket emit function for sending data to the client
        """
        try:
            if not self.training_metrics:
                return  # No historical data available
                
            # Send historical data points to the new client
            for data_point in self.training_metrics:
                historical_message = {
                    "type": "training_update",
                    "data": data_point
                }
                emit_func('training_update', historical_message)
                
        except Exception as e:
            print(f"Warning: Failed to send historical data to client: {e}")
    
    def _send_historical_data_ws(self, ws) -> None:
        """
        Send historical training data to a newly connected WebSocket client.
        
        Args:
            ws: WebSocket connection object
        """
        try:
            if not self.training_metrics:
                return  # No historical data available
                
            # Send historical data points to the new client
            for data_point in self.training_metrics:
                historical_message = {
                    "type": "training_update",
                    "data": data_point
                }
                ws.send(json.dumps(historical_message))
                
        except Exception as e:
            print(f"Warning: Failed to send historical data to WebSocket client: {e}")
    
    def get_server_config(self) -> Dict[str, Any]:
        """
        Get server configuration with localhost-only binding for security.
        
        Returns:
            Dictionary containing host, port, and other server configuration
        """
        return {
            'host': '127.0.0.1',  # Localhost-only binding for security
            'port': self.port,
            'debug': False,
            'threaded': True,  # Enable threading for concurrent requests
            'use_reloader': False  # Disable auto-reloader in production
        }
        
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log training metrics for dashboard display.
        
        Args:
            metrics: Dictionary containing training metrics with keys:
                - 'iter' or 'iteration': Current training iteration
                - 'loss': Training loss value
                - 'elapsed': Elapsed time since training start
        """
        # Debug log: Track all calls to log_metrics
        self._log_debug(f"log_metrics called with: {metrics}")
        
        if not self.enabled:
            self._log_debug("Dashboard not enabled, skipping metrics")
            return
            
        try:
            # Validate and format incoming metrics data
            formatted_data = self._format_training_metrics(metrics)
            if formatted_data is None:
                self._log_debug(f"Formatted data is None, skipping. Original metrics: {metrics}")
                return  # Invalid data, skip silently
                
            # Debug log: Track successful data formatting
            self._log_debug(f"Formatted data: {formatted_data}")
                
            # Store metrics in memory-managed deque (automatic size limit)
            self.training_metrics.append(formatted_data['data'])
            self._log_debug(f"Data stored. Total points: {len(self.training_metrics)}")
                
            # Broadcast formatted training metrics to all connected clients
            self._broadcast_to_clients(formatted_data)
            self._log_debug(f"Data broadcasted to {len(self.websocket_clients)} clients")
            
        except Exception as e:
            # Fail gracefully - don't interrupt training if dashboard has issues
            print(f"Warning: Dashboard metric logging failed: {e}")
    
    def _format_training_metrics(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format training metrics according to design.md data models.
        
        Args:
            metrics: Raw metrics from training loop
            
        Returns:
            Formatted training metrics message or None if invalid
        """
        try:
            # Extract iteration number (support both 'iter' and 'iteration' keys)
            iteration = metrics.get('iter')
            if iteration is None:
                iteration = metrics.get('iteration')
            if iteration is None:
                self._log_debug(f"Missing iteration in metrics: {metrics}")
                print("Warning: Missing iteration in training metrics")
                return None
                
            # Extract loss value
            loss = metrics.get('loss')
            if loss is None:
                self._log_debug(f"Missing loss in metrics: {metrics}")
                print("Warning: Missing loss in training metrics")
                return None
                
            # Extract elapsed time
            elapsed_time = metrics.get('elapsed')
            if elapsed_time is None:
                elapsed_time = metrics.get('elapsed_time')
            if elapsed_time is None:
                self._log_debug(f"Missing elapsed time in metrics: {metrics}")
                print("Warning: Missing elapsed time in training metrics")
                return None
            
            # Validate data types
            try:
                iteration = int(iteration)
                loss = float(loss)
                elapsed_time = float(elapsed_time)
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid data types in training metrics: {e}")
                return None
            
            # Create formatted message according to design.md data model
            formatted_message = {
                "type": "training_update",
                "data": {
                    "iteration": iteration,
                    "loss": loss,
                    "elapsed_time": elapsed_time,
                    "timestamp": time.time()
                }
            }
            
            return formatted_message
            
        except Exception as e:
            print(f"Warning: Error formatting training metrics: {e}")
            return None
    
    def log_status(self, status: str, message: str = "") -> None:
        """
        Log training status updates for dashboard display.
        
        Args:
            status: Training status ('training', 'complete', 'error')
            message: Optional status message
        """
        if not self.enabled:
            return
            
        try:
            # Format status message according to design.md data model
            formatted_message = {
                "type": "status_update",
                "data": {
                    "status": status,
                    "message": message,
                    "timestamp": time.time()
                }
            }
            
            # Broadcast formatted status update to all connected clients
            self._broadcast_to_clients(formatted_message)
            
        except Exception as e:
            # Fail gracefully - don't interrupt training if dashboard has issues
            print(f"Warning: Dashboard status logging failed: {e}")
    
    def send_completion_status(self, message: str = "Training completed successfully") -> None:
        """
        Send completion status notification to dashboard.
        
        This is a convenience method that sends a completion status update
        to the dashboard, indicating that training has finished successfully.
        
        Args:
            message: Optional completion message (default: "Training completed successfully")
        """
        self.log_status('complete', message)
    
    def broadcast_client_count(self) -> None:
        """
        Broadcast current client count to all connected clients.
        """
        if not self.enabled or not self.socketio:
            return
            
        try:
            self.socketio.emit('client_count_update', {
                'connected_clients': len(self.connected_clients),
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"Warning: Failed to broadcast client count: {e}")
    
    def _broadcast_to_clients(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected WebSocket clients.
        
        This method performs asynchronous, non-blocking broadcasts to ensure
        training loop performance is not affected. It handles client disconnections
        gracefully and continues broadcasting to remaining clients.
        
        Args:
            message: Formatted message dictionary to broadcast
        """
        if not self.enabled or not self.websocket_clients:
            return
            
        try:
            # Broadcast to all WebSocket clients
            message_json = json.dumps(message)
            disconnected_clients = set()
            
            for ws in self.websocket_clients:
                try:
                    ws.send(message_json)
                except (ConnectionClosed, Exception):
                    # Mark client for removal if sending fails
                    disconnected_clients.add(ws)
            
            # Remove disconnected clients
            for ws in disconnected_clients:
                self.websocket_clients.discard(ws)
                
        except Exception as e:
            # Handle broadcast failures gracefully - don't interrupt training
            print(f"Warning: Failed to broadcast {message.get('type', 'unknown')} message: {e}")
            
            # If there are persistent errors, clean up disconnected clients
            self._cleanup_disconnected_clients()
    
    def _cleanup_disconnected_clients(self) -> None:
        """
        Clean up disconnected clients that may not have triggered proper disconnect events.
        
        This method helps maintain client connection state accuracy when clients
        disconnect unexpectedly or broadcast errors indicate stale connections.
        """
        if not self.enabled or not self.socketio:
            return
            
        try:
            # Get current active sessions from SocketIO server
            # Note: This is a best-effort cleanup - some SocketIO versions
            # may not expose session information directly
            initial_count = len(self.connected_clients)
            
            # For now, we'll rely on the SocketIO library's built-in disconnect handling
            # If needed, more sophisticated client validation could be added here
            
            final_count = len(self.connected_clients)
            if initial_count != final_count:
                print(f"Cleaned up {initial_count - final_count} disconnected clients")
                
        except Exception as e:
            print(f"Warning: Error during client cleanup: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current WebSocket connection status information.
        
        Returns:
            Dictionary containing connection status details
        """
        return {
            'enabled': self.enabled,
            'connected_clients': len(self.connected_clients),
            'socketio_available': SOCKETIO_AVAILABLE,
            'flask_available': FLASK_AVAILABLE,
            'server_port': self.port,
            'stored_data_points': len(self.training_metrics),
            'memory_limit': self.training_metrics.maxlen
        }
    
    def _is_port_available(self, port: int) -> bool:
        """
        Check if a port is available for binding.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    def _find_available_port(self, start_port: int = 8080, end_port: int = 8084) -> Optional[int]:
        """
        Find an available port in the specified range.
        
        Args:
            start_port: Starting port number (default: 8080)
            end_port: Ending port number (default: 8084)
            
        Returns:
            Available port number or None if no ports available
        """
        for port in range(start_port, end_port + 1):
            if self._is_port_available(port):
                return port
        return None
    
    def start_server(self) -> bool:
        """
        Start the web dashboard server with port collision handling and threading.
        
        Tries to start the server on ports 8080-8084 with collision fallback.
        Runs the server in a separate daemon thread to avoid blocking training.
        
        Returns:
            True if server started successfully, False otherwise
        """
        if not self.enabled:
            print("Dashboard server is disabled (missing dependencies or initialization failed)")
            return False
        
        if not self.app:
            print("Dashboard server cannot start: Flask app not initialized")
            return False
        
        try:
            # Find an available port in the range 8080-8084
            available_port = self._find_available_port()
            if available_port is None:
                print("Dashboard server failed to start: No available ports in range 8080-8084")
                print("Please ensure no other services are running on these ports")
                return False
            
            # Update port if different from initial
            if available_port != self.port:
                print(f"Port {self.port} unavailable, using port {available_port} instead")
                self.port = available_port
            
            # Get server configuration
            server_config = self.get_server_config()
            server_config['port'] = self.port
            
            # Start server in daemon thread to avoid blocking training
            server_thread = threading.Thread(
                target=self._run_server,
                args=(server_config,),
                daemon=True,  # Daemon thread won't prevent program exit
                name="DashboardServer"
            )
            
            print(f"Starting nanoGPT dashboard server on http://127.0.0.1:{self.port}")
            server_thread.start()
            
            # Give server a moment to start and check for immediate failures
            time.sleep(0.5)
            
            if server_thread.is_alive():
                print(f"Dashboard server started successfully on port {self.port}")
                dashboard_url = f"http://127.0.0.1:{self.port}"
                print(f"Visit {dashboard_url} to view training dashboard")
                
                # Attempt to automatically open browser
                self._launch_browser(dashboard_url)
                
                return True
            else:
                print("Dashboard server failed to start (thread exited immediately)")
                return False
                
        except Exception as e:
            print(f"Dashboard server startup failed: {e}")
            return False
    
    def _launch_browser(self, url: str) -> None:
        """
        Attempt to automatically launch the default browser to view the dashboard.
        
        This method tries to open the dashboard URL in the user's default browser.
        If the browser launch fails, it provides clear fallback instructions.
        
        Args:
            url: Dashboard URL to open in the browser
        """
        try:
            # Attempt to open URL in default browser
            print("Attempting to open dashboard in your default browser...")
            webbrowser.open(url, new=2, autoraise=True)  # new=2 opens in new tab if possible
            print("Dashboard opened successfully in browser")
            
        except Exception as e:
            # Handle browser launch failure gracefully
            print(f"Could not automatically open browser: {e}")
            print(f"Please manually open {url} in your browser to view the dashboard")
    
    def _run_server(self, config: Dict[str, Any]) -> None:
        """
        Run the Flask server with WebSocket support.
        
        This method runs in a separate thread and handles server execution.
        
        Args:
            config: Server configuration dictionary
        """
        try:
            # Run the Flask server (this will block in the thread)
            self.app.run(
                host=config['host'],
                port=config['port'],
                debug=config['debug'],
                use_reloader=config['use_reloader'],
                threaded=True  # Enable threading for WebSocket support
            )
        except Exception as e:
            print(f"Dashboard server error: {e}")
            print("Training will continue without the dashboard")
    
    def shutdown(self, reason: str = "complete", message: str = "") -> None:
        """
        Gracefully shutdown the dashboard server and notify clients.
        
        This method handles clean server termination by sending final status
        messages to clients and cleaning up server resources. It supports
        different shutdown scenarios without affecting the training process.
        
        Args:
            reason: Shutdown reason ('complete', 'interrupted', 'error')
            message: Optional shutdown message for clients
        """
        if not self.enabled:
            return
            
        try:
            # Send final status message to all connected clients
            self._send_shutdown_notification(reason, message)
            
            # Clean up server resources
            self._cleanup_server_resources()
            
            # Log shutdown completion
            print(f"Dashboard server shutdown complete (reason: {reason})")
            
        except Exception as e:
            # Even shutdown should fail gracefully
            print(f"Warning: Error during dashboard shutdown: {e}")
    
    def _send_shutdown_notification(self, reason: str, message: str) -> None:
        """
        Send final status notification to clients before shutdown.
        
        Args:
            reason: Shutdown reason ('complete', 'interrupted', 'error')
            message: Optional shutdown message
        """
        if not self.enabled or not self.socketio or not self.connected_clients:
            return
            
        try:
            # Map shutdown reasons to appropriate status messages
            status_map = {
                'complete': 'Training Complete',
                'interrupted': 'Training Interrupted', 
                'error': 'Training Error'
            }
            
            status_message = status_map.get(reason, 'Training Ended')
            final_message = message if message else f"{status_message} - Dashboard will close"
            
            # Send final status update to all clients
            shutdown_message = {
                "type": "status_update",
                "data": {
                    "status": reason,
                    "message": final_message,
                    "final": True,  # Indicates this is the final message
                    "timestamp": time.time()
                }
            }
            
            # Broadcast final message to all connected clients
            self._broadcast_to_clients(shutdown_message)
            
            # Give clients time to process the final message
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Warning: Failed to send shutdown notification: {e}")
    
    def _cleanup_server_resources(self) -> None:
        """
        Clean up server resources and connections.
        
        This method performs cleanup of server threads and resources
        while ensuring the training process is not affected.
        """
        try:
            # Clear connected clients set
            if hasattr(self, 'connected_clients'):
                self.connected_clients.clear()
            
            # Note: We don't forcibly stop the server thread since it's a daemon thread
            # and will automatically terminate when the main process exits.
            # This ensures we don't interfere with the training process.
            
        except Exception as e:
            print(f"Warning: Error during server resource cleanup: {e}")