#!/usr/bin/env python3
import os
import subprocess
import json
from flask import Flask, request, jsonify
from datetime import datetime
import psutil

app = Flask(__name__)

# MySQL config commented out - using local telemetry only
# DB_CONFIG = {
#     "host": "localhost",
#     "user": "testuser",
#     "password": "testpass",
#     "database": "testdb"
# }

TELEMETRY_LOG = "telemetry.jsonl"


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
        mysql_m = get_mysql_metrics()

        telemetry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "status": status,

            "flask_cpu": flask_m["cpu_percent"],
            "flask_memory_mb": flask_m["memory_mb"],
            "flask_fds": flask_m["num_fds"],
            "flask_disk_read_mb": flask_m["disk_read_mb"],
            "flask_disk_write_mb": flask_m["disk_write_mb"],
            "flask_ctx_switches": flask_m["ctx_switches"],
            "flask_children": flask_m["num_children"],

            "mysql_cpu": mysql_m["cpu_percent"],
            "mysql_memory_mb": mysql_m["memory_mb"],
            "mysql_fds": mysql_m["num_fds"],
            "mysql_disk_read_mb": mysql_m["disk_read_mb"],
            "mysql_disk_write_mb": mysql_m["disk_write_mb"],
            "mysql_ctx_switches": mysql_m["ctx_switches"],
            "mysql_children": mysql_m["num_children"]
        }

        with open(TELEMETRY_LOG, "a") as f:
            f.write(json.dumps(telemetry) + "\n")

    except Exception:
        pass


@app.route("/login", methods=["POST"])
def login():
    try:
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        # Simulated login without DB
        log_telemetry("/login", "success")
        
        if username and password:
            return jsonify({"status": "success", "message": "Login successful"})
        else:
            return jsonify({"status": "failure", "message": "Invalid credentials"})

    except Exception as e:
        log_telemetry("/login", "error")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/ping", methods=["GET"])
def ping():
    try:
        host = request.args.get("host", "127.0.0.1")
        
        # Use appropriate ping command for platform
        import platform
        if platform.system() == "Windows":
            cmd = f"ping -n 1 {host}"
        else:
            cmd = f"ping -c 1 {host}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

        log_telemetry("/ping", "success")

        return jsonify({"status": "success", "output": result.stdout})

    except Exception as e:
        log_telemetry("/ping", "error")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/download", methods=["GET"])
def download():
    try:
        filename = request.args.get("file", "readme.txt")
        
        # Simulated download response
        log_telemetry("/download", "success")
        return jsonify({"status": "success", "content": f"File {filename} downloaded successfully"})

    except Exception as e:
        log_telemetry("/download", "error")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    log_telemetry("/health", "success")
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    if not os.path.exists(TELEMETRY_LOG):
        open(TELEMETRY_LOG, "w").close()

    print("Starting vulnerable Flask application...")
    print("WARNING: This application is intentionally vulnerable for IDS testing!")
    app.run(host="0.0.0.0", port=5000, debug=False)
