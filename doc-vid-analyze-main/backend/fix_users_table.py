import sqlite3
import os
import uuid
import datetime

# Define both database paths
DB_PATH_1 = os.path.join(os.path.dirname(__file__), "../data/user_data.db")
DB_PATH_2 = os.path.join(os.path.dirname(__file__), "data/user_data.db")

# Define the function to create users table
# Make sure the create_users_table function allows NULL for hashed_password temporarily
def create_users_table(cursor):
    """Create the users table with all required columns"""
    cursor.execute('''
    CREATE TABLE users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT DEFAULT 'temp_hash_for_migration',
        password TEXT,
        subscription_tier TEXT DEFAULT 'free',
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        api_calls_remaining INTEGER DEFAULT 10,
        last_reset_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

# Update the CREATE TABLE statement to include all necessary columns
def fix_users_table(db_path):
    # Make sure the data directory exists
    data_dir = os.path.dirname(db_path)
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(db_path):
        print(f"Database does not exist at: {os.path.abspath(db_path)}")
        return False
    
    print(f"Using database path: {os.path.abspath(db_path)}")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if cursor.fetchone():
        print("Users table exists, checking schema...")
        
        # Check columns
        cursor.execute("PRAGMA table_info(users)")
        columns_info = cursor.fetchall()
        columns = [column[1] for column in columns_info]
        
        # List of all required columns
        required_columns = ['id', 'email', 'hashed_password', 'password', 'subscription_tier', 
                           'is_active', 'created_at', 'api_calls_remaining', 'last_reset_date']
        
        # Check if any required column is missing
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"Schema needs fixing. Missing columns: {', '.join(missing_columns)}")
            
            # Dynamically build the SELECT query based on available columns
            available_columns = [col for col in columns if col != 'id']  # Exclude id as we'll generate new ones
            
            if not available_columns:
                print("No usable columns found in users table, creating new table...")
                cursor.execute("DROP TABLE users")
                create_users_table(cursor)
                print("Created new empty users table with correct schema")
            else:
                # Backup existing users with available columns
                select_query = f"SELECT {', '.join(available_columns)} FROM users"
                print(f"Backing up users with query: {select_query}")
                cursor.execute(select_query)
                existing_users = cursor.fetchall()
                
                # Drop the existing table
                cursor.execute("DROP TABLE users")
                
                # Create the table with the correct schema
                create_users_table(cursor)
                
                # Restore the users with new UUIDs for IDs
                if existing_users:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for user in existing_users:
                        user_id = str(uuid.uuid4())
                        
                        # Create a dictionary to map column names to values
                        user_data = {'id': user_id}
                        for i, col in enumerate(available_columns):
                            user_data[col] = user[i]
                        
                        # Set default values for missing columns
                        # Add a default value for hashed_password in the Set default values section
                        if 'hashed_password' not in user_data:
                            user_data['hashed_password'] = 'temp_hash_for_migration'  # Temporary hash for migration
                        if 'subscription_tier' not in user_data:
                            user_data['subscription_tier'] = 'free'
                        if 'is_active' not in user_data:
                            user_data['is_active'] = 1
                        if 'created_at' not in user_data:
                            user_data['created_at'] = current_time
                        if 'api_calls_remaining' not in user_data:
                            user_data['api_calls_remaining'] = 10
                        if 'last_reset_date' not in user_data:
                            user_data['last_reset_date'] = current_time
                        
                        # Build INSERT query with all required columns
                        insert_columns = ['id']
                        insert_values = [user_id]
                        
                        # Add values for columns that exist in the old table
                        for col in available_columns:
                            insert_columns.append(col)
                            insert_values.append(user_data[col])
                        
                        # Add default values for columns that don't exist in the old table
                        for col in required_columns:
                            # Add hashed_password to the column default values section
                            if col not in ['id'] + available_columns:
                                insert_columns.append(col)
                                if col == 'subscription_tier':
                                    insert_values.append('free')
                                elif col == 'is_active':
                                    insert_values.append(1)
                                elif col == 'created_at':
                                    insert_values.append(current_time)
                                elif col == 'api_calls_remaining':
                                    insert_values.append(10)
                                elif col == 'last_reset_date':
                                    insert_values.append(current_time)
                                elif col == 'hashed_password':
                                    insert_values.append('temp_hash_for_migration')  # Temporary hash for migration
                                else:
                                    insert_values.append(None)  # Default to NULL for other columns
                        
                        placeholders = ', '.join(['?'] * len(insert_columns))
                        insert_query = f"INSERT INTO users ({', '.join(insert_columns)}) VALUES ({placeholders})"
                        
                        cursor.execute(insert_query, insert_values)
                
                print(f"Fixed users table, restored {len(existing_users)} users")
        else:
            print("Users table schema is correct")
    else:
        print("Users table doesn't exist, creating it now...")
        create_users_table(cursor)
        print("Users table created successfully")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    return True

if __name__ == "__main__":
    print("Checking first database location...")
    success1 = fix_users_table(DB_PATH_1)
    
    print("\nChecking second database location...")
    success2 = fix_users_table(DB_PATH_2)
    
    if not (success1 or success2):
        print("\nWarning: Could not find any existing database files.")
        print("Creating a new database at the primary location...")
        # Create a new database at the primary location
        data_dir = os.path.dirname(DB_PATH_1)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH_1)
        cursor = conn.cursor()
        create_users_table(cursor)
        conn.commit()
        conn.close()
        print(f"Created new database at: {os.path.abspath(DB_PATH_1)}")