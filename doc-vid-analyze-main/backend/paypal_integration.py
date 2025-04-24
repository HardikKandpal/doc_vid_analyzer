import requests
import json
import sqlite3
from datetime import datetime, timedelta
import uuid
import os
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from auth import get_db_connection
from dotenv import load_dotenv

from auth import get_subscription_plans


# PayPal API Configuration - Remove default values for production
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET")
PAYPAL_BASE_URL = os.getenv("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")

# Add validation to ensure credentials are provided
# Set up logging
LOG_DIR = os.path.abspath(os.getenv("LOG_DIR", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "paypal.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paypal_integration")

# Then replace print statements with logger calls
# For example:
if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
    logger.warning("PayPal credentials not found in environment variables")


# Get PayPal access token
# Add better error handling for production
# Create a session with retry capability
def create_retry_session(retries=3, backoff_factor=0.3):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Then use this session for API calls
# Replace get_access_token with logger instead of print
def get_access_token():
    url = f"{PAYPAL_BASE_URL}/v1/oauth2/token"
    headers = {
        "Accept": "application/json",
        "Accept-Language": "en_US"
    }
    data = "grant_type=client_credentials"
    
    try:
        session = create_retry_session()
        response = session.post(
            url, 
            auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
            headers=headers,
            data=data
        )
        
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            logger.error(f"Error getting access token: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception in get_access_token: {str(e)}")
        return None

def call_paypal_api(endpoint, method="GET", data=None, token=None):
    """
    Helper function to make PayPal API calls
    
    Args:
        endpoint: API endpoint (without base URL)
        method: HTTP method (GET, POST, etc.)
        data: Request payload (for POST/PUT)
        token: PayPal access token (will be fetched if None)
        
    Returns:
        tuple: (success, response_data or error_message)
    """
    try:
        if not token:
            token = get_access_token()
            if not token:
                return False, "Failed to get PayPal access token"
        
        url = f"{PAYPAL_BASE_URL}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        session = create_retry_session()
        
        if method.upper() == "GET":
            response = session.get(url, headers=headers)
        elif method.upper() == "POST":
            response = session.post(url, headers=headers, data=json.dumps(data) if data else None)
        elif method.upper() == "PUT":
            response = session.put(url, headers=headers, data=json.dumps(data) if data else None)
        else:
            return False, f"Unsupported HTTP method: {method}"
        
        if response.status_code in [200, 201, 204]:
            if response.status_code == 204:  # No content
                return True, {}
            return True, response.json() if response.text else {}
        else:
            logger.error(f"PayPal API error: {response.status_code} - {response.text}")
            return False, f"PayPal API error: {response.status_code} - {response.text}"
            
    except Exception as e:
        logger.error(f"Error calling PayPal API: {str(e)}")
        return False, f"Error calling PayPal API: {str(e)}"

def create_paypal_subscription(user_id, tier):
    """Create a PayPal subscription for a user"""
    try:
        # Get the price from the subscription tier
        from auth import SUBSCRIPTION_TIERS
        
        if tier not in SUBSCRIPTION_TIERS:
            return False, f"Invalid tier: {tier}"
        
        price = SUBSCRIPTION_TIERS[tier]["price"]
        currency = SUBSCRIPTION_TIERS[tier]["currency"]
        
        # Create a PayPal subscription (implement PayPal API calls here)
        # For now, just return a success response
        return True, {
            "subscription_id": f"test_sub_{uuid.uuid4()}",
            "status": "ACTIVE",
            "tier": tier,
            "price": price,
            "currency": currency
        }
    except Exception as e:
        logger.error(f"Error creating PayPal subscription: {str(e)}")
        return False, f"Failed to create PayPal subscription: {str(e)}"

        
# Create a product in PayPal
def create_product(name, description):
    """Create a product in PayPal"""
    payload = {
        "name": name,
        "description": description,
        "type": "SERVICE",
        "category": "SOFTWARE"
    }
    
    success, result = call_paypal_api("/v1/catalogs/products", "POST", payload)
    if success:
        return result["id"]
    else:
        logger.error(f"Failed to create product: {result}")
        return None

# Create a subscription plan in PayPal
# Update create_plan to use INR instead of USD
def create_plan(product_id, name, price, interval="MONTH", interval_count=1):
    """Create a subscription plan in PayPal"""
    payload = {
        "product_id": product_id,
        "name": name,
        "billing_cycles": [
            {
                "frequency": {
                    "interval_unit": interval,
                    "interval_count": interval_count
                },
                "tenure_type": "REGULAR",
                "sequence": 1,
                "total_cycles": 0,  # Infinite cycles
                "pricing_scheme": {
                    "fixed_price": {
                        "value": str(price),
                        "currency_code": "INR"
                    }
                }
            }
        ],
        "payment_preferences": {
            "auto_bill_outstanding": True,
            "setup_fee": {
                "value": "0",
                "currency_code": "INR"
            },
            "setup_fee_failure_action": "CONTINUE",
            "payment_failure_threshold": 3
        }
    }
    
    success, result = call_paypal_api("/v1/billing/plans", "POST", payload)
    if success:
        return result["id"]
    else:
        logger.error(f"Failed to create plan: {result}")
        return None

# Update initialize_subscription_plans to use INR pricing
def initialize_subscription_plans():
    """
    Initialize PayPal subscription plans for the application.
    This should be called once to set up the plans in PayPal.
    """
    try:
        # Check if plans already exist
        existing_plans = get_subscription_plans()
        if existing_plans and len(existing_plans) >= 2:
            logger.info("PayPal plans already initialized")
            return existing_plans
        
        # First, create products for each tier
        products = {
            "standard_tier": {
                "name": "Standard Legal Document Analysis",
                "description": "Standard subscription with document analysis features",
                "type": "SERVICE",
                "category": "SOFTWARE"
            },
            "premium_tier": {
                "name": "Premium Legal Document Analysis",
                "description": "Premium subscription with all document analysis features",
                "type": "SERVICE",
                "category": "SOFTWARE"
            }
        }
        
        product_ids = {}
        for tier, product_data in products.items():
            success, result = call_paypal_api("/v1/catalogs/products", "POST", product_data)
            if success:
                product_ids[tier] = result["id"]
                logger.info(f"Created PayPal product for {tier}: {result['id']}")
            else:
                logger.error(f"Failed to create product for {tier}: {result}")
                return None
        
        # Define the plans with product IDs - Changed currency to USD
        plans = {
            "standard_tier": {
                "product_id": product_ids["standard_tier"],
                "name": "Standard Plan",
                "description": "Standard subscription with basic features",
                "billing_cycles": [
                    {
                        "frequency": {
                            "interval_unit": "MONTH",
                            "interval_count": 1
                        },
                        "tenure_type": "REGULAR",
                        "sequence": 1,
                        "total_cycles": 0,
                        "pricing_scheme": {
                            "fixed_price": {
                                "value": "799",
                                "currency_code": "INR"
                            }
                        }
                    }
                ],
                "payment_preferences": {
                    "auto_bill_outstanding": True,
                    "setup_fee": {
                        "value": "0",
                        "currency_code": "INR"
                    },
                    "setup_fee_failure_action": "CONTINUE",
                    "payment_failure_threshold": 3
                }
            },
            "premium_tier": {
                "product_id": product_ids["premium_tier"],
                "name": "Premium Plan",
                "description": "Premium subscription with all features",
                "billing_cycles": [
                    {
                        "frequency": {
                            "interval_unit": "MONTH",
                            "interval_count": 1
                        },
                        "tenure_type": "REGULAR",
                        "sequence": 1,
                        "total_cycles": 0,
                        "pricing_scheme": {
                            "fixed_price": {
                                "value": "1499",
                                "currency_code": "INR"
                            }
                        }
                    }
                ],
                "payment_preferences": {
                    "auto_bill_outstanding": True,
                    "setup_fee": {
                        "value": "0",
                        "currency_code": "INR"
                    },
                    "setup_fee_failure_action": "CONTINUE",
                    "payment_failure_threshold": 3
                }
            }
        }
        
        # Create the plans in PayPal
        created_plans = {}
        for tier, plan_data in plans.items():
            success, result = call_paypal_api("/v1/billing/plans", "POST", plan_data)
            if success:
                created_plans[tier] = result["id"]
                logger.info(f"Created PayPal plan for {tier}: {result['id']}")
            else:
                logger.error(f"Failed to create plan for {tier}: {result}")
        
        # Save the plan IDs to a file
        if created_plans:
            save_subscription_plans(created_plans)
            return created_plans
        else:
            logger.error("Failed to create any PayPal plans")
            return None
    except Exception as e:
        logger.error(f"Error initializing subscription plans: {str(e)}")
        return None

# Update create_subscription_link to use call_paypal_api helper
def create_subscription_link(plan_id):
    # Get the plan IDs
    plans = get_subscription_plans()
    if not plans:
        return None
    
    # Use environment variable for the app URL to make it work in different environments
    app_url = os.getenv("APP_URL", "http://localhost:8500")
    
    payload = {
        "plan_id": plans[plan_id],
        "application_context": {
            "brand_name": "Legal Document Analyzer",
            "locale": "en_US",
            "shipping_preference": "NO_SHIPPING",
            "user_action": "SUBSCRIBE_NOW",
            "return_url": f"{app_url}?status=success&subscription_id={{id}}",
            "cancel_url": f"{app_url}?status=cancel"
        }
    }
    
    success, data = call_paypal_api("/v1/billing/subscriptions", "POST", payload)
    if not success:
        logger.error(f"Error creating subscription: {data}")
        return None
    
    try:
        return {
            "subscription_id": data["id"],
            "approval_url": next(link["href"] for link in data["links"] if link["rel"] == "approve")
        }
    except Exception as e:
        logger.error(f"Exception processing subscription response: {str(e)}")
        return None

# Fix the webhook handler function signature to match how it's called in app.py
def handle_subscription_webhook(payload):
    """
    Handle PayPal subscription webhooks
    
    Args:
        payload: The full webhook payload
        
    Returns:
        tuple: (success, result)
            - success: True if successful, False otherwise
            - result: Success message or error message
    """
    try:
        event_type = payload.get("event_type")
        resource = payload.get("resource", {})
        
        logger.info(f"Received PayPal webhook: {event_type}")
        
        # Handle different event types
        if event_type == "BILLING.SUBSCRIPTION.CREATED":
            # A subscription was created
            subscription_id = resource.get("id")
            if not subscription_id:
                return False, "Missing subscription ID in webhook"
                
            # Update subscription status in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE subscriptions SET status = 'pending' WHERE paypal_subscription_id = ?",
                (subscription_id,)
            )
            conn.commit()
            conn.close()
            
            return True, "Subscription created successfully"
            
        elif event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            # A subscription was activated
            subscription_id = resource.get("id")
            if not subscription_id:
                return False, "Missing subscription ID in webhook"
                
            # Update subscription status in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE subscriptions SET status = 'active' WHERE paypal_subscription_id = ?",
                (subscription_id,)
            )
            conn.commit()
            conn.close()
            
            return True, "Subscription activated successfully"
            
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            # A subscription was cancelled
            subscription_id = resource.get("id")
            if not subscription_id:
                return False, "Missing subscription ID in webhook"
                
            # Update subscription status in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE subscriptions SET status = 'cancelled' WHERE paypal_subscription_id = ?",
                (subscription_id,)
            )
            conn.commit()
            conn.close()
            
            return True, "Subscription cancelled successfully"
            
        elif event_type == "BILLING.SUBSCRIPTION.SUSPENDED":
            # A subscription was suspended
            subscription_id = resource.get("id")
            if not subscription_id:
                return False, "Missing subscription ID in webhook"
                
            # Update subscription status in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE subscriptions SET status = 'suspended' WHERE paypal_subscription_id = ?",
                (subscription_id,)
            )
            conn.commit()
            conn.close()
            
            return True, "Subscription suspended successfully"
            
        else:
            # Unhandled event type
            logger.info(f"Unhandled webhook event type: {event_type}")
            return True, f"Unhandled event type: {event_type}"
            
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}")
        return False, f"Error handling webhook: {str(e)}"
# Add this function to update user subscription
def update_user_subscription(user_email, subscription_id, tier):
    """
    Update a user's subscription status
    
    Args:
        user_email: The email of the user
        subscription_id: The PayPal subscription ID
        tier: The subscription tier
        
    Returns:
        tuple: (success, result)
            - success: True if successful, False otherwise
            - result: Success message or error message
    """
    try:
        # Get user ID from email
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_email,))
        user_result = cursor.fetchone()
        
        if not user_result:
            conn.close()
            return False, f"User not found: {user_email}"
        
        user_id = user_result[0]
        
        # Update the subscription status
        cursor.execute(
            "UPDATE subscriptions SET status = 'active' WHERE user_id = ? AND paypal_subscription_id = ?",
            (user_id, subscription_id)
        )
        
        # Deactivate any other active subscriptions for this user
        cursor.execute(
            "UPDATE subscriptions SET status = 'inactive' WHERE user_id = ? AND paypal_subscription_id != ? AND status = 'active'",
            (user_id, subscription_id)
        )
        
        # Update the user's subscription tier
        cursor.execute(
            "UPDATE users SET subscription_tier = ? WHERE email = ?",
            (tier, user_email)
        )
        
        conn.commit()
        conn.close()
        
        return True, f"Subscription updated to {tier} tier"
        
    except Exception as e:
        logger.error(f"Error updating user subscription: {str(e)}")
        return False, f"Error updating subscription: {str(e)}"

# Add this near the top with other path definitions
# Update the PLAN_IDS_PATH definition to use the correct path
PLAN_IDS_PATH="data/plan_ids.json"
os.makedirs(os.path.dirname(PLAN_IDS_PATH), exist_ok=True)
# Make sure the data directory exists

# Add this debug log to see where the file is expected
logger.info(f"PayPal plans will be stored at: {PLAN_IDS_PATH}")

# Add this function if it's not defined elsewhere
def get_db_connection():
    """Get a connection to the SQLite database"""
    DB_PATH="data/user_data.db"
    # Make sure the data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

# Add this function to create subscription tables if needed
def initialize_database():
    """Initialize the database tables needed for subscriptions"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if subscriptions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='subscriptions'")
    if cursor.fetchone():
        # Table exists, check if required columns exist
        cursor.execute("PRAGMA table_info(subscriptions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Check for missing columns and add them if needed
        if "user_id" not in columns:
            logger.info("Adding 'user_id' column to subscriptions table")
            cursor.execute("ALTER TABLE subscriptions ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
        
        if "created_at" not in columns:
            logger.info("Adding 'created_at' column to subscriptions table")
            cursor.execute("ALTER TABLE subscriptions ADD COLUMN created_at TIMESTAMP")
        
        if "expires_at" not in columns:
            logger.info("Adding 'expires_at' column to subscriptions table")
            cursor.execute("ALTER TABLE subscriptions ADD COLUMN expires_at TIMESTAMP")
        
        if "paypal_subscription_id" not in columns:
            logger.info("Adding 'paypal_subscription_id' column to subscriptions table")
            cursor.execute("ALTER TABLE subscriptions ADD COLUMN paypal_subscription_id TEXT")
    else:
        # Create subscriptions table with all required columns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            tier TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            expires_at TIMESTAMP,
            paypal_subscription_id TEXT
        )
        ''')
        logger.info("Created subscriptions table with all required columns")
    
    # Create PayPal plans table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paypal_plans (
        plan_id TEXT PRIMARY KEY,
        tier TEXT NOT NULL,
        price REAL NOT NULL,
        currency TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialization completed")


def create_user_subscription_mock(user_email, tier):
    """
    Create a mock subscription for testing
    
    Args:
        user_email: The email of the user
        tier: The subscription tier
        
    Returns:
        tuple: (success, result)
    """
    try:
        logger.info(f"Creating mock subscription for {user_email} at tier {tier}")
        
        # Get user ID from email
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_email,))
        user_result = cursor.fetchone()
        
        if not user_result:
            conn.close()
            return False, f"User not found: {user_email}"
        
        user_id = user_result[0]
        
        # Create a mock subscription ID
        subscription_id = f"mock_sub_{uuid.uuid4()}"
        
        # Store the subscription in database
        sub_id = str(uuid.uuid4())
        start_date = datetime.now()
        
        cursor.execute(
            "INSERT INTO subscriptions (id, user_id, tier, status, created_at, expires_at, paypal_subscription_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sub_id, user_id, tier, "active", start_date, start_date + timedelta(days=30), subscription_id)
        )
        
        # Update user's subscription tier
        cursor.execute(
            "UPDATE users SET subscription_tier = ? WHERE id = ?",
            (tier, user_id)
        )
        
        conn.commit()
        conn.close()
        
        # Use environment variable for the app URL
        app_url = os.getenv("APP_URL", "https://testing-hdq7qxb3k-hardikkandpals-projects.vercel.app/")
        
        # Return success with mock approval URL that matches the real PayPal URL pattern
        return True, {
            "subscription_id": subscription_id,
            "approval_url": f"{app_url}/subscription/callback?status=success&subscription_id={subscription_id}",
            "tier": tier
        }
        
    except Exception as e:
        logger.error(f"Error creating mock subscription: {str(e)}")
        return False, f"Error creating subscription: {str(e)}"

# Add this at the end of the file
def initialize():
    """Initialize the PayPal integration module"""
    try:
        # Create necessary directories
        os.makedirs(os.path.dirname(PLAN_IDS_PATH), exist_ok=True)
        
        # Initialize database
        initialize_database()
        
        # Initialize subscription plans
        plans = get_subscription_plans()
        if plans:
            logger.info(f"Subscription plans initialized: {plans}")
        else:
            logger.warning("Failed to initialize subscription plans")
            
        return True
    except Exception as e:
        logger.error(f"Error initializing PayPal integration: {str(e)}")
        return False

# Call initialize when the module is imported
initialize()

# Add this function to get subscription plans
def get_subscription_plans():
    """
    Get all available subscription plans with correct pricing
    """
    try:
        # Check if we have plan IDs saved in a file
        if os.path.exists(PLAN_IDS_PATH):
            try:
                with open(PLAN_IDS_PATH, 'r') as f:
                    plans = json.load(f)
                    logger.info(f"Loaded subscription plans from {PLAN_IDS_PATH}: {plans}")
                    return plans
            except Exception as e:
                logger.error(f"Error reading plan IDs file: {str(e)}")
                return {}
        
        # If no file exists, return empty dict
        logger.warning(f"No plan IDs file found at {PLAN_IDS_PATH}. Please initialize subscription plans.")
        return {}
        
    except Exception as e:
        logger.error(f"Error getting subscription plans: {str(e)}")
        return {}

# Add this function to create subscription tables if needed
def initialize_database():
    """Initialize the database tables needed for subscriptions"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create subscriptions table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subscriptions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        tier TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        expires_at TIMESTAMP,
        paypal_subscription_id TEXT
    )
    ''')
    
    # Create PayPal plans table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paypal_plans (
        plan_id TEXT PRIMARY KEY,
        tier TEXT NOT NULL,
        price REAL NOT NULL,
        currency TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()


def create_user_subscription(user_email, tier):
    """
    Create a real PayPal subscription for a user
    
    Args:
        user_email: The email of the user
        tier: The subscription tier (standard_tier or premium_tier)
    
    Returns:
        tuple: (success, result)
            - success: True if successful, False otherwise
            - result: Dictionary with subscription details or error message
    """
    try:
        # Validate tier
        valid_tiers = ["standard_tier", "premium_tier"]
        if tier not in valid_tiers:
            return False, f"Invalid tier: {tier}. Must be one of {valid_tiers}"
        
        # Get the plan IDs
        plans = get_subscription_plans()
        
        # Log the plans for debugging
        logger.info(f"Available subscription plans: {plans}")
        
        # If no plans found, check if the file exists and try to load it directly
        if not plans:
            if os.path.exists(PLAN_IDS_PATH):
                logger.info(f"Plan IDs file exists at {PLAN_IDS_PATH}, but couldn't load plans. Trying direct load.")
                try:
                    with open(PLAN_IDS_PATH, 'r') as f:
                        plans = json.load(f)
                        logger.info(f"Directly loaded plans: {plans}")
                except Exception as e:
                    logger.error(f"Error directly loading plans: {str(e)}")
            else:
                logger.error(f"Plan IDs file does not exist at {PLAN_IDS_PATH}")
                
            # If still no plans, return error
            if not plans:
                logger.error("No PayPal plans found. Please initialize plans first.")
                return False, "PayPal plans not configured. Please contact support."
        
        # Check if the tier exists in plans
        if tier not in plans:
            return False, f"No plan found for tier: {tier}"
            
        # Use environment variable for the app URL
        app_url = os.getenv("APP_URL", "https://testing-hdq7qxb3k-hardikkandpals-projects.vercel.app/")
        
        # Create the subscription with PayPal
        payload = {
            "plan_id": plans[tier],
            "subscriber": {
                "email_address": user_email
            },
            "application_context": {
                "brand_name": "Legal Document Analyzer",
                "locale": "en-US",  # Changed from en_US to en-US
                "shipping_preference": "NO_SHIPPING",
                "user_action": "SUBSCRIBE_NOW",
                "return_url": f"{app_url}/subscription/callback?status=success",
                "cancel_url": f"{app_url}/subscription/callback?status=cancel"
            }
        }
        
        # Make the API call to PayPal
        success, subscription_data = call_paypal_api("/v1/billing/subscriptions", "POST", payload)
        if not success:
            return False, subscription_data  # This is already an error message
        
        # Extract the approval URL
        approval_url = next((link["href"] for link in subscription_data["links"] 
                            if link["rel"] == "approve"), None)
        
        if not approval_url:
            return False, "No approval URL found in PayPal response"
        
        # Get user ID from email
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_email,))
        user_result = cursor.fetchone()
        
        if not user_result:
            conn.close()
            return False, f"User not found: {user_email}"
        
        user_id = user_result[0]
        
        # Store pending subscription in database
        sub_id = str(uuid.uuid4())
        start_date = datetime.now()
        
        cursor.execute(
            "INSERT INTO subscriptions (id, user_id, tier, status, created_at, expires_at, paypal_subscription_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sub_id, user_id, tier, "pending", start_date, None, subscription_data["id"])
        )
        
        conn.commit()
        conn.close()
        
        # Return success with approval URL
        return True, {
            "subscription_id": subscription_data["id"],
            "approval_url": approval_url,
            "tier": tier
        }
        
    except Exception as e:
        logger.error(f"Error creating user subscription: {str(e)}")
        return False, f"Error creating subscription: {str(e)}"

# Add a function to cancel a subscription
def cancel_subscription(subscription_id, reason="Customer requested cancellation"):
    """
    Cancel a PayPal subscription
    
    Args:
        subscription_id: The PayPal subscription ID
        reason: The reason for cancellation
        
    Returns:
        tuple: (success, result)
            - success: True if successful, False otherwise
            - result: Success message or error message
    """
    try:
        # Cancel the subscription with PayPal
        payload = {
            "reason": reason
        }
        
        success, result = call_paypal_api(
            f"/v1/billing/subscriptions/{subscription_id}/cancel", 
            "POST", 
            payload
        )
        
        if not success:
            return False, result
        
        # Update subscription status in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE subscriptions SET status = 'cancelled' WHERE paypal_subscription_id = ?",
            (subscription_id,)
        )
        
        # Get the user ID for this subscription
        cursor.execute(
            "SELECT user_id FROM subscriptions WHERE paypal_subscription_id = ?",
            (subscription_id,)
        )
        user_result = cursor.fetchone()
        
        if user_result:
            # Update user to free tier
            cursor.execute(
                "UPDATE users SET subscription_tier = 'free_tier' WHERE id = ?",
                (user_result[0],)
            )
        
        conn.commit()
        conn.close()
        
        return True, "Subscription cancelled successfully"
        
    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        return False, f"Error cancelling subscription: {str(e)}"

def verify_subscription_payment(subscription_id):
    """
    Verify a subscription payment with PayPal
    
    Args:
        subscription_id: The PayPal subscription ID
        
    Returns:
        tuple: (success, result)
            - success: True if successful, False otherwise
            - result: Dictionary with subscription details or error message
    """
    try:
        # Get subscription details from PayPal using our helper
        success, subscription_data = call_paypal_api(f"/v1/billing/subscriptions/{subscription_id}")
        if not success:
            return False, subscription_data  # This is already an error message
        
        # Check subscription status
        status = subscription_data.get("status", "").upper()
        
        if status not in ["ACTIVE", "APPROVED"]:
            return False, f"Subscription is not active: {status}"
        
        # Return success with subscription data
        return True, subscription_data
        
    except Exception as e:
        logger.error(f"Error verifying subscription: {str(e)}")
        return False, f"Error verifying subscription: {str(e)}"

def verify_paypal_subscription(subscription_id):
    """
    Verify a PayPal subscription
    
    Args:
        subscription_id: The PayPal subscription ID
        
    Returns:
        tuple: (success, result)
    """
    try:
        # Skip verification for mock subscriptions
        if subscription_id.startswith("mock_sub_"):
            return True, {"status": "ACTIVE"}
            
        # For real subscriptions, call PayPal API
        success, result = call_paypal_api(f"/v1/billing/subscriptions/{subscription_id}", "GET")
        
        if success:
            # Check subscription status
            if result.get("status") == "ACTIVE":
                return True, result
            else:
                return False, f"Subscription is not active: {result.get('status')}"
        else:
            logger.error(f"PayPal API error: {result}")
            return False, f"Failed to verify subscription: {result}"
    except Exception as e:
        logger.error(f"Error verifying PayPal subscription: {str(e)}")
        return False, f"Error verifying subscription: {str(e)}"

# Add this function to save subscription plans
def save_subscription_plans(plans):
    """
    Save subscription plans to a file
    
    Args:
        plans: Dictionary of plan IDs by tier
    """
    try:
        with open(PLAN_IDS_PATH, 'w') as f:
            json.dump(plans, f)
        logger.info(f"Saved subscription plans to {PLAN_IDS_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving subscription plans: {str(e)}")
        return False
