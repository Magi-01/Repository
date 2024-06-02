import mysql.connector

# Establish the database connection
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='tree hogger',
    database='dimensional_transfer'
)

# Create a cursor object
cursor = conn.cursor()

# Define the query
query = "SELECT p.name, g.name AS guild_name FROM Player p JOIN Guild g ON p.guild_id = g.guild_id"

# Execute the query
cursor.execute(query)

# Fetch and print the results
results = cursor.fetchall()
for row in results:
    print(f"Player: {row[0]}, Guild: {row[1]}")

# Close the cursor and connection
cursor.close()
conn.close()
