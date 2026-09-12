-- Initial schema for the sqli_demo database used by web_app.py

CREATE TABLE IF NOT EXISTS users (
       id INT AUTO_INCREMENT PRIMARY KEY,
       username VARCHAR(100) NOT NULL,
       password VARCHAR(255) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   INSERT INTO users (username, password) VALUES
       ('admin', 'admin123'),
       ('demo_user', 'demo123');