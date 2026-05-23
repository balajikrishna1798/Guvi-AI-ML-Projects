import mysql.connector
from mysql.connector import Error

def create_connection():
    """Create a database connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='uber_data',
            user='root',
            password='password' # Placeholder, user should change this
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
    return None

def create_table(connection):
    """Create the rides table if it doesn't exist."""
    try:
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS rides (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pickup_location VARCHAR(255),
            destination VARCHAR(255),
            current_datetime DATETIME,
            vehicle_type VARCHAR(50),
            fare_estimate FLOAT,
            waiting_time_mins INT,
            drop_time_mins INT
        )
        """
        cursor.execute(create_table_query)
        connection.commit()
    except Error as e:
        print(f"Failed to create table: {e}")

def insert_ride_data(connection, data):
    """Insert a new ride record into the rides table."""
    try:
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO rides (pickup_location, destination, current_datetime, vehicle_type, fare_estimate, waiting_time_mins, drop_time_mins)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(insert_query, data)
        connection.commit()
    except Error as e:
        print(f"Failed to insert record: {e}")

if __name__ == "__main__":
    conn = create_connection()
    if conn:
        create_table(conn)
        print("Database setup complete.")
        conn.close()
