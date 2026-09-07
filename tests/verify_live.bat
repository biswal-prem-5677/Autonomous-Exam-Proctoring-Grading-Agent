@echo off
chcp 65001 >nul
echo ============================================================
echo   PHASE 1: REAL APPLICATION VERIFICATION
echo ============================================================
echo.

set PASS=0
set FAIL=0

curl -s http://localhost:5000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] Server not running. Starting it...
    start /B python -X utf8 app/api/server.py
    timeout /t 5 /nobreak >nul
)

echo -- Authentication --
echo.

echo [1/6] Register examiner...
curl -s -X POST http://localhost:5000/api/auth/register -H "Content-Type: application/json" -d "{\"username\":\"v_exam\",\"email\":\"v_exam@test.com\",\"password\":\"test123\",\"full_name\":\"Verify Examiner\",\"role\":\"examiner\"}"
echo.

echo [2/6] Register student...
curl -s -X POST http://localhost:5000/api/auth/register -H "Content-Type: application/json" -d "{\"username\":\"v_stud\",\"email\":\"v_stud@test.com\",\"password\":\"test123\",\"full_name\":\"Verify Student\",\"role\":\"student\"}"
echo.

echo [3/6] Login student...
curl -s -X POST http://localhost:5000/api/auth/login -H "Content-Type: application/json" -d "{\"username\":\"v_stud\",\"password\":\"test123\"}"
echo.

echo.
echo -- Exam Management --
echo.

echo [4/6] Create exam...
curl -s -X POST http://localhost:5000/api/exams -H "Content-Type: application/json" -d "{\"title\":\"Verify Exam\",\"description\":\"Test\",\"duration_minutes\":30,\"created_by\":\"v_exam\"}"
echo.

echo [5/6] List exams...
curl -s http://localhost:5000/api/exams
echo.

echo.
echo -- Proctoring APIs --
echo.

echo [6/6] Vision analysis...
curl -s -X POST http://localhost:5000/api/vision/analyze -H "Content-Type: application/json" -d "{\"session_key\":\"test\",\"face_present\":true,\"face_count\":1,\"face_confidence\":0.9,\"head_pose\":{\"yaw\":0,\"pitch\":0,\"roll\":0}}"
echo.

echo [7/6] Behavioral analysis...
curl -s -X POST http://localhost:5000/api/behavioral/analyze -H "Content-Type: application/json" -d "{\"session_key\":\"test\",\"face_present\":true,\"face_count\":1,\"face_confidence\":0.9,\"head_pose\":{\"yaw\":0},\"keyboard\":{\"events_per_minute\":60,\"idle_seconds\":5,\"deviation\":0.1},\"mouse\":{\"avg_speed\":100,\"total_distance\":500,\"idle_seconds\":10,\"deviation\":0.1},\"browser\":{\"tab_switches\":0},\"session_elapsed\":300}"
echo.

echo.
echo [8/6] Agent status...
curl -s "http://localhost:5000/api/agent/status?session_key=test"
echo.

echo.
echo -- Page Rendering --
echo.

echo [9/6] Page: /
curl -s -o nul -w "HTTP %%{http_code}" http://localhost:5000/
echo.

echo.
echo [10/6] Page: /exam
curl -s -o nul -w "HTTP %%{http_code}" http://localhost:5000/exam
echo.

echo.
echo [11/6] Page: /examiner
curl -s -o nul -w "HTTP %%{http_code}" http://localhost:5000/examiner
echo.

echo.
echo ============================================================
echo   VERIFICATION COMPLETE
echo ============================================================
