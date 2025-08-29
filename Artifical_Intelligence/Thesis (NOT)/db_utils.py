# db_utils.py
import pymysql
from datetime import datetime

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="warehouse",
        password="ai_hello",
        database="warehouse_ai",
        cursorclass=pymysql.cursors.DictCursor
    )

# --- Item Operations ---
def insert_item(item_type, correct_bin, deadline_seconds, drop_bin, status="pending"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO items (item_type, correct_bin, deadline_seconds, drop_bin, status, arrival_time) "
        "VALUES (%s,%s,%s,%s,%s,NOW())",
        (item_type, correct_bin, deadline_seconds, drop_bin, status)
    )
    conn.commit()
    cursor.close()
    conn.close()

def fetch_pending_items():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM items
        WHERE status = 'pending'
        ORDER BY arrival_time ASC
    """)
    items = cursor.fetchall()
    cursor.close()
    conn.close()
    return items

def update_item_status(item_id, sorted_to_bin, success):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE items
        SET status = %s, sorted_to_bin = %s
        WHERE item_id = %s
    """, ("sorted" if success else "wrong", sorted_to_bin, item_id))
    conn.commit()
    cursor.close()
    conn.close()

def fetch_bin_location(bin_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT x, y FROM bins WHERE bin_id=%s", (bin_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result  # {'x': int, 'y': int}

def mark_item_carrying(item_id, agent_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE items SET carrying=TRUE WHERE item_id=%s", (item_id,))
    conn.commit()
    cursor.close()
    conn.close()

"""
def mark_item_carrying(item_id, agent_id):
    ""
    Mark an item as being carried by an agent.
    ""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = ""
                UPDATE items
                SET status='carried', agent_id=%s, picked_at=%s
                WHERE item_id=%s AND status='pending'
            ""
            cursor.execute(sql, (agent_id, datetime.now(), item_id))
        conn.commit()
    finally:
        conn.close()
"""