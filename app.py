#!/usr/bin/env python3
import os
import subprocess
import json
import re
import ipaddress
from flask import Flask, request, jsonify
from datetime import datetime
import psutil

app = Flask(__name__)

# Security settings
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max request size

# MySQL config - enable if database available
MYSQL_ENABLED = False
DB_CONFIG = {
    "host": "localhost",
    "user": "testuser",
    "password": "testpass",
    "database": "testdb"
}

TELEMETRY_LOG = "telemetry.jsonl"
MAX_FILENAME_LENGTH = 100


def validate_ip_address(ip_str):
    """Validate IP address - prevent command injection"""
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        if re.match(r'^[a-zA-Z0-9.-]{1,255}$', ip_str):
            return ip_str.count('.') <= 3
        return False

def validate_filename(filename):
    """Validate filename - prevent path traversal"""
    if not filename or len(filename) > MAX_FILENAME_LENGTH:
        return False
    return bool(re.match(r'^[a-zA-Z0-9._-]+$', filename))

def add_security_headers(response):
    """Add security headers to response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


def get_process_metrics():
    metrics = {
        "cpu_percent": 0.0,
        "memory_mb": 0.0,
        "num_fds": 0,
        "disk_read_mb": 0.0,
        "disk_write_mb": 0.0,
        "ctx_switches": 0,
        "num_children": 0
    }

    try:
        p = psutil.Process()
        metrics["cpu_percent"] = p.cpu_percent(interval=0.1)
        metrics["memory_mb"] = p.memory_info().rss / (1024 * 1024)

        try:
            metrics["num_fds"] = p.num_fds()
        except:
            metrics["num_fds"] = len(p.open_files())

        io = p.io_counters()
        metrics["disk_read_mb"] = io.read_bytes / (1024 * 1024)
        metrics["disk_write_mb"] = io.write_bytes / (1024 * 1024)

        ctx = p.num_ctx_switches()
        metrics["ctx_switches"] = ctx.voluntary + ctx.involuntary

        metrics["num_children"] = len(p.children())

    except Exception:
        pass

    return metrics


def get_mysql_metrics():
    metrics = {
        "cpu_percent": 0.0,
        "memory_mb": 0.0,
        "num_fds": 0,
        "disk_read_mb": 0.0,
        "disk_write_mb": 0.0,
        "ctx_switches": 0,
        "num_children": 0
    }

    try:
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                name = proc.info["name"]
                if name and ("mysql" in name.lower() or "mysqld" in name.lower()):

                    proc.cpu_percent(interval=0.1)

                    metrics["cpu_percent"] = proc.cpu_percent(interval=0.1)
                    metrics["memory_mb"] = proc.memory_info().rss / (1024 * 1024)

                    try:
                        metrics["num_fds"] = proc.num_fds()
                    except:
                        metrics["num_fds"] = len(proc.open_files())

                    io = proc.io_counters()
                    metrics["disk_read_mb"] = io.read_bytes / (1024 * 1024)
                    metrics["disk_write_mb"] = io.write_bytes / (1024 * 1024)

                    ctx = proc.num_ctx_switches()
                    metrics["ctx_switches"] = ctx.voluntary + ctx.involuntary

                    metrics["num_children"] = len(proc.children())

                    break

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except Exception:
        pass

    return metrics


def log_telemetry(endpoint, status):
    try:
        flask_m = get_process_metrics()
        mysql_m = get_mysql_metrics() if MYSQL_ENABLED else {}

        telemetry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "status": status,
            "remote_addr": request.remote_addr,

            "flask_cpu": flask_m["cpu_percent"],
            "flask_memory_mb": flask_m["memory_mb"],
            "flask_fds": flask_m["num_fds"],
            "flask_disk_read_mb": flask_m["disk_read_mb"],
            "flask_disk_write_mb": flask_m["disk_write_mb"],
            "flask_ctx_switches": flask_m["ctx_switches"],
            "flask_children": flask_m["num_children"],

            "mysql_cpu": mysql_m.get("cpu_percent", 0.0),
            "mysql_memory_mb": mysql_m.get("memory_mb", 0.0),
            "mysql_fds": mysql_m.get("num_fds", 0),
            "mysql_disk_read_mb": mysql_m.get("disk_read_mb", 0.0),
            "mysql_disk_write_mb": mysql_m.get("disk_write_mb", 0.0),
            "mysql_ctx_switches": mysql_m.get("ctx_switches", 0),
            "mysql_children": mysql_m.get("num_children", 0)
        }

        with open(TELEMETRY_LOG, "a") as f:
            f.write(json.dumps(telemetry) + "\n")

    except Exception as e:
        print(f"Telemetry logging error: {e}")


@app.route("/login", methods=["POST"])
def login():
    try:
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        # Input validation
        if not username or not password:
            log_telemetry("/login", "rejected_empty")
            return jsonify({"status": "failure", "message": "Invalid credentials"}), 400
        
        if len(username) > 100 or len(password) > 100:
            log_telemetry("/login", "rejected_toolong")
            return jsonify({"status": "failure", "message": "Credentials too long"}), 400

        # Simulated login - always succeeds for IDS testing
        log_telemetry("/login", "success")
        response = jsonify({"status": "success", "message": "Login successful"})
        return add_security_headers(response)

    except Exception as e:
        log_telemetry("/login", "error")
        return jsonify({"status": "error", "message": "Internal error"}), 500


@app.route("/ping", methods=["GET"])
def ping():
    try:
        host = request.args.get("host", "127.0.0.1").strip()
        
        # Input validation - prevent command injection
        if not validate_ip_address(host):
            log_telemetry("/ping", "rejected_invalid_host")
            return jsonify({"status": "error", "message": "Invalid host"}), 400
        
        # Use subprocess without shell=True for safety
        import platform
        if platform.system() == "Windows":
            cmd = ["ping", "-n", "1", host]
        else:
            cmd = ["ping", "-c", "1", host]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        output = result.stdout[:500] if result.stdout else ""
        log_telemetry("/ping", "success")

        response = jsonify({"status": "success", "output": output, "returncode": result.returncode})
        return add_security_headers(response)

    except subprocess.TimeoutExpired:
        log_telemetry("/ping", "timeout")
        return jsonify({"status": "error", "message": "Ping timeout"}), 504
    except Exception as e:
        log_telemetry("/ping", "error")
        return jsonify({"status": "error", "message": "Ping failed"}), 500


@app.route("/download", methods=["GET"])
def download():
    try:
        filename = request.args.get("file", "readme.txt").strip()
        
        # Input validation - prevent path traversal
        if not validate_filename(filename):
            log_telemetry("/download", "rejected_invalid_filename")
            return jsonify({"status": "error", "message": "Invalid filename"}), 400
        
        # Simulated download response
        log_telemetry("/download", "success")
        response = jsonify({"status": "success", "content": f"File {filename} downloaded successfully"})
        return add_security_headers(response)

    except Exception as e:
        log_telemetry("/download", "error")
        return jsonify({"status": "error", "message": "Download failed"}), 500


@app.route("/health", methods=["GET"])
def health():
    log_telemetry("/health", "success")
    response = jsonify({"status": "healthy"})
    return add_security_headers(response)


if __name__ == "__main__":
    if not os.path.exists(TELEMETRY_LOG):
        open(TELEMETRY_LOG, "w").close()

    print("Starting IDS Flask application...")
    print(f"Telemetry logging to: {TELEMETRY_LOG}")
    print(f"Max request size: 1MB")
    print(f"Security: Input validation enabled")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
