import sqlite3

import uuid
import os
import logging
from datetime import datetime, timedelta
import hashlib  # Use hashlib instead of jwt
from passlib.hash import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException, status
import jwt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('auth')

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-for-development-only")
ALGORITHM = "HS256" 
JWT_EXPIRATION_DELTA = timedelta(days=1)  # Token valid for 1 day
# Database path from environment variable or default
# Fix the incorrect DB_PATH
DB_PATH = os.getenv("DB_PATH", "data/user_data.db")


# FastAPI OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models for FastAPI
class User(BaseModel):
    id: str
    email: str
    subscription_tier: str = "free_tier"
    subscription_expiry: Optional[datetime] = None
    api_calls_remaining: int = 5
    last_reset_date: Optional[datetime] = None

class UserCreate(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

# Subscription tiers and limits
# Update the SUBSCRIPTION_TIERS dictionary
SUBSCRIPTION_TIERS = {
    "free_tier": {
        "price": 0,
        "currency": "INR",
        "features": ["basic_document_analysis", "basic_risk_assessment"],
        "limits": {
            "document_size_mb": 5,
            "documents_per_month": 3,
            "video_size_mb": 0,
            "audio_size_mb": 0,
            "daily_api_calls": 5,              # <-- Add this
            "max_document_size_mb": 5           # <-- Add this
        }
    },
    "standard_tier": {
        "price": 799,
        "currency": "INR",
        "features": ["basic_document_analysis", "basic_risk_assessment", "video_analysis", "audio_analysis", "chatbot"],
        "limits": {
            "document_size_mb": 20,
            "documents_per_month": 20,
            "video_size_mb": 100,
            "audio_size_mb": 50,
            "daily_api_calls": 100,             # <-- Add this
            "max_document_size_mb": 20          # <-- Add this
        }
    },
    "premium_tier": {
        "price": 1499,
        "currency": "INR",
        "features": ["basic_document_analysis", "basic_risk_assessment", "video_analysis", "audio_analysis", "chatbot", "detailed_risk_assessment", "contract_clause_analysis"],
        "limits": {
            "document_size_mb": 50,
            "documents_per_month": 999,
            "video_size_mb": 500,
            "audio_size_mb": 200,
            "daily_api_calls": 1000,            # <-- Add this
            "max_document_size_mb": 50          # <-- Add this
        }
    }
}

# Database connection management
def get_db_connection():
    """Create and return a database connection with proper error handling"""
    try:
        # Ensure the directory exists

        db_dir = os.path.dirname(DB_PATH)
        os.makedirs(db_dir, exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise Exception(f"Database connection failed: {e}")

# Database setup
# In the init_auth_db function, update the CREATE TABLE statement to match our schema
def init_auth_db():
    """Initialize the authentication database with required tables"""
    conn= None
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Create users table with the correct schema
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            password TEXT,
            subscription_tier TEXT DEFAULT 'free_tier',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            api_calls_remaining INTEGER DEFAULT 10,
            last_reset_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create subscriptions table
        c.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            tier TEXT,
            plan_id TEXT,
            status TEXT,
            created_at TIMESTAMP,
            expires_at TIMESTAMP,
            paypal_subscription_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create usage stats table
        c.execute('''
        CREATE TABLE IF NOT EXISTS usage_stats (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            month INTEGER,
            year INTEGER,
            analyses_used INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create tokens table for refresh tokens
        c.execute('''
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            user_id TEXT,
            token TEXT,
            expires_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        c.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in c.fetchall()]
        if "subscription_expiry" not in columns:
            c.execute("ALTER TABLE users ADD COLUMN subscription_expiry TIMESTAMP")
            conn.commit()
            logger.info("Added 'subscription_expiry' column to users table")
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Initialize the database
init_auth_db()

# Password hashing with bcrypt
# Update the password hashing and verification functions to use a more reliable method

# Replace these functions
# Remove these conflicting functions
# def hash_password(password):
#     """Hash a password using bcrypt"""
#     return bcrypt.hash(password)
# 
# def verify_password(plain_password, hashed_password):
#     """Verify a password against its hash"""
#     return bcrypt.verify(plain_password, hashed_password)

# Keep only these improved functions
def hash_password(password):
    """Hash a password using bcrypt"""
    # Use a more direct approach to avoid bcrypt version issues
    import bcrypt
    # Convert password to bytes if it's not already
    if isinstance(password, str):
        password = password.encode('utf-8')
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password, salt)
    # Return as string for storage
    return hashed.decode('utf-8')

def verify_password(plain_password, hashed_password):
    """Verify a password against its hash"""
    import bcrypt
    # Convert inputs to bytes if they're not already
    if isinstance(plain_password, str):
        plain_password = plain_password.encode('utf-8')
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    
    try:
        # Use direct bcrypt verification
        return bcrypt.checkpw(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

# User registration
def register_user(email, password):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Check if user already exists
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        if c.fetchone():
            return False, "Email already registered"
        
        # Create new user
        user_id = str(uuid.uuid4())
        
        # Add more detailed logging
        logger.info(f"Registering new user with email: {email}")
        hashed_pw = hash_password(password)
        logger.info(f"Password hashed successfully: {bool(hashed_pw)}")
        
        c.execute("""
            INSERT INTO users 
            (id, email, hashed_password, subscription_tier, api_calls_remaining, last_reset_date) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, email, hashed_pw, "free_tier", 5, datetime.now()))
        
        conn.commit()
        logger.info(f"User registered successfully: {email}")
        
        # Verify the user was actually stored
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        stored_user = c.fetchone()
        logger.info(f"User verification after registration: {bool(stored_user)}")
        
        access_token = create_access_token(user_id)
        return True, {
            "user_id": user_id,
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        logger.error(f"User registration error: {e}")
        return False, f"Registration failed: {str(e)}"
    finally:
        if conn:
            conn.close()

# User login
# Fix the authenticate_user function
# In the authenticate_user function, update the password verification to use hashed_password
def authenticate_user(email, password):
    """Authenticate a user and return user data with tokens"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get user by email
        c.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email,))
        user = c.fetchone()
        
        if not user:
            logger.warning(f"User not found: {email}")
            return None
            
        # Add debug logging for password verification
        logger.info(f"Verifying password for user: {email}")
        logger.info(f"Stored hashed password: {user['hashed_password'][:20]}...")
        
        try:
            # Check if password verification works
            is_valid = verify_password(password, user['hashed_password'])
            logger.info(f"Password verification result: {is_valid}")
            
            if not is_valid:
                logger.warning(f"Password verification failed for user: {email}")
                return None
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return None
        
        # Update last login time if column exists
        try:
            c.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                    (datetime.now(), user['id']))
            conn.commit()
        except sqlite3.OperationalError:
            # last_login column might not exist
            pass
        
        # Convert sqlite3.Row to dict to use get() method
        user_dict = dict(user)
        
        # Create and return a User object
        return User(
            id=user_dict['id'],
            email=user_dict['email'],
            subscription_tier=user_dict.get('subscription_tier', 'free_tier'),
            subscription_expiry=None,  # Handle this properly if needed
            api_calls_remaining=user_dict.get('api_calls_remaining', 5),
            last_reset_date=user_dict.get('last_reset_date')
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        return None
    finally:
        if conn:
            conn.close()

# Token generation and validation - completely replaced
def create_access_token(user_id):
    """Create a new access token for a user"""
    try:
        # Create a JWT token with user_id and expiration
        expiration = datetime.now() + JWT_EXPIRATION_DELTA
        
        # Create a token payload
        payload = {
            "sub": user_id,
            "exp": expiration.timestamp()
        }
        
        # Generate the JWT token
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        
        logger.info(f"Created access token for user: {user_id}")
        return token
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        return None


def update_auth_db_schema():
    """Update the authentication database schema with any missing columns"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Check if tier column exists in subscriptions table
        c.execute("PRAGMA table_info(subscriptions)")
        columns = [column[1] for column in c.fetchall()]
        
        # Add tier column if it doesn't exist
        if "tier" not in columns:
            logger.info("Adding 'tier' column to subscriptions table")
            c.execute("ALTER TABLE subscriptions ADD COLUMN tier TEXT")
            conn.commit()
            logger.info("Database schema updated successfully")
        
        conn.close()
    except Exception as e:
        logger.error(f"Database schema update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database schema update error: {str(e)}"
        )

# Add this to your get_current_user function
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.error("Token missing 'sub' field")
            raise credentials_exception
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise credentials_exception

    conn = get_db_connection()
    cursor = conn.cursor()
    # Fetch all relevant fields, including api_calls_remaining and last_reset_date
    cursor.execute("SELECT id, email, subscription_tier, is_active, api_calls_remaining, last_reset_date, subscription_expiry FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()

    if user_data is None:
        logger.error(f"User not found: {user_id}")
        raise credentials_exception

    user = User(
        id=user_data[0],
        email=user_data[1],
        subscription_tier=user_data[2],
        is_active=bool(user_data[3]),
        api_calls_remaining=user_data[4],
        last_reset_date=user_data[5],
        subscription_expiry=user_data[6]
    )
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get the current active user"""
    return current_user
    
def create_user_subscription(email, tier):
    """Create a subscription for a user"""
    try:
        # Get user by email
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get user ID
        c.execute("SELECT id FROM users WHERE email = ?", (email,))
        user_data = c.fetchone()
        
        if not user_data:
            return False, "User not found"
        
        user_id = user_data['id']
        
        # Check if tier is valid
        valid_tiers = ["standard_tier", "premium_tier"]
        if tier not in valid_tiers:
            return False, f"Invalid tier: {tier}. Must be one of {valid_tiers}"
        
        # Create subscription
        subscription_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=30)  # 30-day subscription
        
        # Insert subscription
        c.execute("""
            INSERT INTO subscriptions 
            (id, user_id, tier, status, created_at, expires_at) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (subscription_id, user_id, tier, "active", created_at, expires_at))
        
        # Update user's subscription tier
        c.execute("""
            UPDATE users 
            SET subscription_tier = ? 
            WHERE id = ?
        """, (tier, user_id))
        
        conn.commit()
        
        return True, {
            "id": subscription_id,
            "user_id": user_id,
            "tier": tier,
            "status": "active",
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Subscription creation error: {e}")
        return False, f"Failed to create subscription: {str(e)}"
    finally:
        if conn:
            conn.close()

def get_user(user_id: str):
    """Get user by ID"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get user
        c.execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,))
        user_data = c.fetchone()
        
        if not user_data:
            return None
            
        # Convert to User model
        user_dict = dict(user_data)
        
        # Handle datetime conversions if needed
        if user_dict.get("subscription_expiry") and isinstance(user_dict["subscription_expiry"], str):
            user_dict["subscription_expiry"] = datetime.fromisoformat(user_dict["subscription_expiry"])
        if user_dict.get("last_reset_date") and isinstance(user_dict["last_reset_date"], str):
            user_dict["last_reset_date"] = datetime.fromisoformat(user_dict["last_reset_date"])
            
        return User(
            id=user_dict['id'],
            email=user_dict['email'],
            subscription_tier=user_dict['subscription_tier'],
            subscription_expiry=user_dict.get('subscription_expiry'),
            api_calls_remaining=user_dict.get('api_calls_remaining', 5),
            last_reset_date=user_dict.get('last_reset_date')
        )
    except Exception as e:
        logger.error(f"Get user error: {e}")
        return None
    finally:
        if conn:
            conn.close()

def check_subscription_access(user: User, feature: str, file_size_mb: Optional[float] = None):
    """Check if the user has access to the requested feature and file size"""
    # Check if subscription is expired
    if user.subscription_tier != "free_tier" and user.subscription_expiry and user.subscription_expiry < datetime.now():
        # Downgrade to free tier if subscription expired
        user.subscription_tier = "free_tier"
        user.api_calls_remaining = SUBSCRIPTION_TIERS["free_tier"]["limits"]["daily_api_calls"]  # <-- FIXED
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE users 
                SET subscription_tier = ?, api_calls_remaining = ? 
                WHERE id = ?
            """, (user.subscription_tier, user.api_calls_remaining, user.id))
            conn.commit()
        
    # Reset API calls if needed
    user = reset_api_calls_if_needed(user)
    
    # Check if user has API calls remaining
    if user.api_calls_remaining <= 0:
        raise HTTPException(
            status_code=429,
            detail="API call limit reached for today. Please upgrade your subscription or try again tomorrow."
        )
    
    # Check if feature is available in user's subscription tier
    tier_features = SUBSCRIPTION_TIERS[user.subscription_tier]["features"]
    if feature not in tier_features:
        raise HTTPException(
            status_code=403,
            detail=f"The {feature} feature is not available in your {user.subscription_tier} subscription. Please upgrade to access this feature."
        )
    
    # Check file size limit if applicable
    if file_size_mb:
        max_size = SUBSCRIPTION_TIERS[user.subscription_tier]["limits"]["max_document_size_mb"]
        if file_size_mb > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the {max_size}MB limit for your {user.subscription_tier} subscription. Please upgrade or use a smaller file."
            )
    
    # Decrement API calls remaining
    user.api_calls_remaining -= 1
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE users 
            SET api_calls_remaining = ? 
            WHERE id = ?
        """, (user.api_calls_remaining, user.id))
        conn.commit()
    
    return True

def reset_api_calls_if_needed(user: User):
    """Reset API call counter if it's a new day"""
    today = datetime.now().date()
    if user.last_reset_date is None or user.last_reset_date.date() < today:
        tier_limits = SUBSCRIPTION_TIERS[user.subscription_tier]["limits"]  # <-- FIXED
        user.api_calls_remaining = tier_limits["daily_api_calls"]
        user.last_reset_date = datetime.now()
        # Update the user in the database
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE users 
                SET api_calls_remaining = ?, last_reset_date = ? 
                WHERE id = ?
            """, (user.api_calls_remaining, user.last_reset_date, user.id))
            conn.commit()
            
    return user

def login_user(email, password):
    """Login a user with email and password"""
    try:
        # Authenticate user
        user = authenticate_user(email, password)
        if not user:
            return False, "Incorrect username or password"
        
        # Create access token
        access_token = create_access_token(user.id)
        
        # Create refresh token
        refresh_token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(days=30)
        
        # Store refresh token
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO refresh_tokens VALUES (?, ?, ?)",
                 (user.id, refresh_token, expires_at))
        conn.commit()
        
        # Get subscription info
        c.execute("SELECT * FROM subscriptions WHERE user_id = ? AND status = 'active'", (user.id,))
        subscription = c.fetchone()
        
        # Convert subscription to dict if it exists, otherwise set to None
        subscription_dict = dict(subscription) if subscription else None
        
        conn.close()
        
        return True, {
            "user_id": user.id,
            "email": user.email,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "subscription": subscription_dict
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        return False, f"Login failed: {str(e)}"


def get_subscription_plans():
    """
    Returns a list of available subscription plans based on SUBSCRIPTION_TIERS.
    """
    plans = []
    for tier, details in SUBSCRIPTION_TIERS.items():
        plans.append({
            "tier": tier,
            "price": details["price"],
            "currency": details["currency"],
            "features": details["features"],
            "limits": details["limits"]
        })
    return plans