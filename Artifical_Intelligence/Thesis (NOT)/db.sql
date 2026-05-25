CREATE DATABASE IF NOT EXISTS warehouse_ai;
USE warehouse_ai;

CREATE TABLE IF NOT EXISTS items (
  item_id INT AUTO_INCREMENT PRIMARY KEY,
  item_type VARCHAR(20),
  correct_bin INT,
  sorted_to_bin INT,
  status VARCHAR(20), -- 'pending', 'sorted', 'wrong'
  arrival_time DATETIME DEFAULT CURRENT_TIMESTAMP,
  deadline_seconds INT
);