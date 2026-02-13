#!/bin/bash
# Attack Simulation Script
# Simulates various web attacks for IDS testing

BASE_URL="http://localhost:5000"

echo "=========================================="
echo "Attack Simulation Suite"
echo "=========================================="
echo ""
echo "This script will simulate:"
echo "1. Normal traffic (baseline)"
echo "2. SQL Injection attack"
echo "3. Command Injection attack"
echo "4. Path Traversal attack"
echo ""
echo "Monitor should detect attacks in real-time."
echo "Press Ctrl+C to stop at any time."
echo ""
read -p "Press Enter to start..."

# Phase 1: Normal Traffic
echo ""
echo "=========================================="
echo "Phase 1: Normal Traffic (10 requests)"
echo "=========================================="
sleep 2

for i in {1..10}; do
    echo "[$i/10] Normal login..."
    curl -s -X POST "$BASE_URL/login" \
         -d "username=admin&password=admin123" > /dev/null
    sleep 0.5
done

echo "✓ Normal traffic complete"
sleep 3

# Phase 2: SQL Injection Attack
echo ""
echo "=========================================="
echo "Phase 2: SQL Injection Attack (5 requests)"
echo "=========================================="
echo "Attack type: ' OR '1'='1"
sleep 2

for i in {1..5}; do
    echo "[$i/5] SQL injection attempt..."
    curl -s -X POST "$BASE_URL/login" \
         -d "username=admin' OR '1'='1&password=anything" > /dev/null
    sleep 1
done

echo "✓ SQL injection complete"
sleep 3

# Phase 3: Command Injection Attack
echo ""
echo "=========================================="
echo "Phase 3: Command Injection Attack (5 requests)"
echo "=========================================="
echo "Attack type: Command chaining with ;"
sleep 2

for i in {1..5}; do
    echo "[$i/5] Command injection attempt..."
    curl -s -G "$BASE_URL/ping" \
         --data-urlencode "host=127.0.0.1; cat /etc/passwd" > /dev/null
    sleep 1
done

echo "✓ Command injection complete"
sleep 3

# Phase 4: Path Traversal Attack
echo ""
echo "=========================================="
echo "Phase 4: Path Traversal Attack (5 requests)"
echo "=========================================="
echo "Attack type: Directory traversal with ../"
sleep 2

for i in {1..5}; do
    echo "[$i/5] Path traversal attempt..."
    curl -s -G "$BASE_URL/download" \
         --data-urlencode "file=../../../../etc/passwd" > /dev/null
    sleep 1
done

echo "✓ Path traversal complete"
sleep 3

# Phase 5: Mixed Normal Traffic
echo ""
echo "=========================================="
echo "Phase 5: Return to Normal Traffic (10 requests)"
echo "=========================================="
sleep 2

for i in {1..10}; do
    echo "[$i/10] Normal ping..."
    curl -s -G "$BASE_URL/ping" \
         --data-urlencode "host=127.0.0.1" > /dev/null
    sleep 0.5
done

echo "✓ Normal traffic resumed"

echo ""
echo "=========================================="
echo "Attack Simulation Complete!"
echo "=========================================="
echo ""
echo "Check monitor output for detected attacks."
echo "Check drift_log.csv for detailed drift scores."
echo ""
