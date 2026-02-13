#!/bin/bash
# Setup script for IDS system
# No root/sudo required

set -e

echo "=========================================="
echo "IDS System Setup"
echo "=========================================="

# Check Python version
echo "[1/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "[3/6] Installing Python packages..."
pip install --upgrade pip > /dev/null
pip install flask mysql-connector-python psutil numpy > /dev/null
echo "Python packages installed"

# Setup MySQL
echo "[4/6] Setting up MySQL database..."

# Check if MySQL is running
if ! systemctl is-active --quiet mysql 2>/dev/null && ! pgrep -x mysqld > /dev/null; then
    echo "WARNING: MySQL not detected. Please start MySQL service."
    echo "On Ubuntu/Debian: sudo systemctl start mysql"
    echo "On macOS: brew services start mysql"
else
    echo "MySQL is running"
fi

# Create database and user (will prompt for MySQL root password if needed)
echo "Creating test database and user..."
cat > /tmp/setup_db.sql << 'EOF'
CREATE DATABASE IF NOT EXISTS testdb;
CREATE USER IF NOT EXISTS 'testuser'@'localhost' IDENTIFIED BY 'testpass';
GRANT ALL PRIVILEGES ON testdb.* TO 'testuser'@'localhost';
FLUSH PRIVILEGES;

USE testdb;
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(50) NOT NULL
);

INSERT INTO users (username, password) VALUES 
    ('admin', 'admin123'),
    ('user', 'password')
ON DUPLICATE KEY UPDATE username=username;
EOF

# Try to execute SQL setup
mysql -u root -p < /tmp/setup_db.sql 2>/dev/null || {
    echo "Note: If MySQL prompts for password, enter your MySQL root password"
    mysql -u root -p < /tmp/setup_db.sql
}

rm /tmp/setup_db.sql
echo "Database setup complete"

# Create test files directory
echo "[5/6] Creating test files..."
mkdir -p /tmp/files
echo "This is a test file" > /tmp/files/readme.txt
echo "Secret data" > /tmp/files/secret.txt
echo "Test files created in /tmp/files/"

# Create necessary files
echo "[6/6] Initializing system files..."
touch telemetry.jsonl
touch drift_log.csv
echo "System files initialized"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Generate baseline: python3 generate_baseline.py"
echo "3. Start Flask app: python3 app.py"
echo "4. Start monitor: python3 online_monitor.py"
echo "5. Run attacks: bash attack_simulation.sh"
echo ""
