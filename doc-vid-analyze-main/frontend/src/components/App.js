import React, { useEffect, useState } from 'react';
import { checkBackendConnection } from '../api';

function App() {
  const [backendConnected, setBackendConnected] = useState(true);
  const [connectionChecked, setConnectionChecked] = useState(false);
  
  useEffect(() => {
    const checkConnection = async () => {
      try {
        console.log("Checking backend connection...");
        const isConnected = await checkBackendConnection();
        console.log("Connection check result:", isConnected);
        setBackendConnected(isConnected);
      } catch (error) {
        console.error("Error checking backend connection:", error);
        setBackendConnected(false);
      } finally {
        setConnectionChecked(true);
      }
    };
    
    // Initial check
    checkConnection();
    
    // Set up periodic connection checks
    const intervalId = setInterval(checkConnection, 30000); // Check every 30 seconds
    
    return () => clearInterval(intervalId); // Clean up on unmount
  }, []);
  
  return (
    <div className="App">
      {connectionChecked && !backendConnected && (
        <div style={{ 
          backgroundColor: '#fff3cd', 
          color: '#856404', 
          padding: '10px', 
          textAlign: 'center',
          borderRadius: '4px',
          margin: '10px'
        }}>
          Could not connect to backend server. Some features may not work.
          Please make sure the backend server is running at http://localhost:8500.
        </div>
      )}
      
      {/* Rest of your app */}
    </div>
  );
}

export default App;