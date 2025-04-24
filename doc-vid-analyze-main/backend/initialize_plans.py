import os
import sys
from dotenv import load_dotenv
from paypal_integration import initialize_subscription_plans

# Load environment variables
load_dotenv()

def main():
    """Initialize PayPal subscription plans"""
    print("Initializing PayPal subscription plans...")
    plans = initialize_subscription_plans()
    
    if plans:
        print("✅ Plans initialized successfully:")
        for tier, plan_id in plans.items():
            print(f"  - {tier}: {plan_id}")
        return True
    else:
        print("❌ Failed to initialize plans. Check the logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)