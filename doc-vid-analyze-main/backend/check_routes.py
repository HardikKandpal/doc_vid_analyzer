import os
import sys
import importlib.util

def check_fastapi_routes():
    """Check the FastAPI routes in the app.py file."""
    print("Checking FastAPI routes...")
    
    # Store current directory
    current_dir = os.getcwd()
    
    try:
        # Change to the backend directory
        os.chdir("doc-vid-analyze-main")
        print(f"Current directory: {os.getcwd()}")
        
        # Check if app.py exists
        if not os.path.exists("app.py"):
            print(f"❌ Error: app.py not found in {os.getcwd()}")
            return False
        
        # Load the app.py module
        print("Loading app.py module...")
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        sys.modules["app"] = app_module
        spec.loader.exec_module(app_module)
        
        # Check if app is defined
        if not hasattr(app_module, "app"):
            print("❌ Error: 'app' object not found in app.py")
            return False
        
        app = app_module.app
        
        # Print app information
        print("\n📋 FastAPI App Information:")
        print(f"App title: {app.title}")
        print(f"App version: {app.version}")
        print(f"App description: {app.description}")
        
        # Print routes
        print("\n📋 FastAPI Routes:")
        for route in app.routes:
            print(f"Route: {route.path}")
            print(f"  Methods: {route.methods}")
            print(f"  Name: {route.name}")
            print(f"  Endpoint: {route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else route.endpoint}")
            print()
        
        return True
            
    except Exception as e:
        print(f"❌ Error checking routes: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(current_dir)

if __name__ == "__main__":
    check_fastapi_routes()